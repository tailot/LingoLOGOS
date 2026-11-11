# Technical Documentation: Neuro-Symbolic Grammar Induction and Rule Synthesis Engine

This document provides complete engineering and theoretical guidelines for maintaining, extending, configuring, and formally verifying the differentiable grammar induction and neuro-symbolic synthesis engine.

---

## 1. Internal Architecture and Module Breakdown

The codebase implements a hybrid neural-symbolic architecture structured as follows:

```
src/neuro_symbolic/
├── __init__.py          # Public interface and module exports
├── parser.py            # DifferentiableCYKParser: Inside CYK (log semiring), family gate, Viterbi AST
├── grammar.py           # PCFGrammar, ProductionRule, Symbol, pruning + renormalization, derive(), exporters
├── trainer.py           # DyckDataset (train/test split), GrammarTrainer (NLL + mixture entropy, annealing)
├── reference.py         # Gold Dyck-2 CNF, gold prior initialization, negative controls, gold rule F1
├── tokenizer.py         # Byte-pair vocabulary learned from scratch: atomic algebra alphabet, lossless round trip
├── masking.py           # Symbolic decoding mask: prefix automaton of the legal algebra, then token-level logits mask
├── verification.py      # GrammarVerifier: Lark, PCFG invariant, CYK coverage, negative controls, Z3 inclusion
└── synthesize.py        # End-to-end synthesis, acceptance gates (G1-G4) and artifact export pipeline
```

### 1.1 `parser.py` - DifferentiableCYKParser
- **Parameter Representations**:
  - `rule_logits` ($\mathbb{R}^{K \times K \times K}$): unnormalized log-odds for binary non-terminal productions $A \to B \ C$.
  - `term_logits` ($\mathbb{R}^{K \times |V_T|}$): unnormalized log-odds for terminal emissions $A \to a$.
  - `family_logits` ($\mathbb{R}^{K}$): logit $\phi_A$ of the **Bernoulli family gate** $\pi_A = \sigma(\phi_A)$ that splits the probability budget of $A$ between the binary and the terminal production family.
- **Joint Normalization (Distribution Invariant)**:
  $$P(A \to B\,C) = \pi_A \cdot \mathrm{softmax}_{\theta}(r_A)_{B,C}, \qquad P(A \to a) = (1 - \pi_A) \cdot \mathrm{softmax}_{\theta}(t_A)_a$$
  so that, for every $A \in V_N$,
  $$\sum_{B,C} P(A \to B\,C) + \sum_{a} P(A \to a) = 1 .$$
  `get_log_rules()` adds $\log \pi_A$ and `get_log_terms()` adds $\log(1-\pi_A)$; because the softmax blocks are normalized separately, the identity holds analytically for any temperature $\tau$ (verified numerically by `production_mass_per_lhs()`, error $\sim 10^{-7}$).
  > Historical defect: before this revision the two families were normalized independently, so each left-hand side carried mass $2$ and marginal string probabilities were systematically inflated (each derivation picked up a spurious factor $2^{\#\text{nodes}}$). `tests/test_parser.py` locks the invariant against regression.
- **Logarithmic Soft Semiring**:
  Probability multiplications are mapped to log-space additions, and summation over alternative derivation paths is computed via the numerically stable, differentiable `LogSumExp` operator. The chart is indexed by span length (`chart[b, L-1, i, A]` = inside score of $x_i \dots x_{i+L-1}$ under $A$).
- **Deterministic Viterbi CYK Parsing**:
  Instead of `logsumexp`, `parse_viterbi` tracks `argmax` choices across split points $k$ and non-terminal pairs $(B, C)$ to extract the optimal Abstract Syntax Tree (AST) in polynomial time $O(N^3 \cdot K^3)$. Used as constructive evidence (derivation trees) by the verifier and by the counterexample search.
- **Entropy of the Production Mixture**:
  `production_entropy_per_lhs()` returns $H(A) = H_b(\pi_A) + \pi_A H(P(A \to \cdot\,\cdot)) + (1-\pi_A) H(P(A \to \cdot))$, the regularizer actually used by the trainer (`rule_entropy()`).

### 1.2 `grammar.py` - Symbolic Abstraction and Serialization
- **Core Data Structures**:
  - `Symbol`: Models terminal and non-terminal symbols.
  - `ProductionRule`: Models context-free productions $A \to \alpha$ with probability $P \in [0, 1]$ plus `raw_probability`, the pre-pruning mass kept for audit (`probability_pre_pruning` in the JSON/YAML export).
  - `PCFGrammar`: Encapsulates the formal grammar tuple $G = (V_N, V_T, R, S, P)$.
- **Deterministic Rule Pruning and Renormalization**:
  `PCFGrammar.from_model(parser_model, nt_labels, vocab_labels, start_symbol="S", prune_threshold=0.01, temperature=1.0, renormalize=True)` drops every production with raw mass $P(r) < \tau_{\text{prune}}$. Pruning destroys the distribution invariant, so the remaining productions are **renormalized per left-hand side** ($\sum_\alpha P(A \to \alpha) = 1$ again) unless `renormalize=False` is requested for diagnostics. Skipping this step was the second distributional defect of the previous implementation (post-pruning masses measured up to $1.998$).
- **Grammar Sanity and Derivability**:
  - `production_mass_per_lhs()` / `is_valid_pcfg(tolerance=1e-6)`: numerical check of the invariant on the extracted grammar.
  - `productions_by_lhs()`: discrete production index used by the verifier.
  - `derive(sequence)`: **discrete CYK** membership decision that returns the derivation tree (nodes as `{"type": ..., "rule"/"token": ..., "children": [...]}`) or `None`. It is the engine behind corpus coverage, negative-control rejection and witness search — a yes-answer is a constructive proof, not a probabilistic estimate.
- **Export Formats**:
  - `to_json(...)`: Structured JSON grammar serialization.
  - `to_yaml(...)`: Clean YAML configuration format.
  - `to_gf(...)`: Abstract and concrete grammar definitions for **Grammatical Framework** (`.gf`).
  - `to_lark(...)`: EBNF specification for Lark parser validation.

### 1.3 `trainer.py` - Training and Optimization
- **Dyck-2 Generator and Phrase-Level Split**:
  `DyckDataset` generates balanced sentences over parentheses `()` and brackets `[]` with bounded recursion depth and de-duplicated samples. `split(test_fraction=0.25, seed=42)` returns **disjoint sentence sets** (no shared string between train and test), so hold-out metrics are not contaminated by memorization.
- **Regularized Objective** (mixture entropy, consistent with the joint gate of §1.1):
  $$\mathcal{L}(\theta) = -\frac{1}{|D|} \sum_{X \in D} \log P(S \stackrel{*}{\Rightarrow} X \mid \theta) + \lambda_1 \sum_{A \in V_N} H(A) + \lambda_2 \|\theta\|_2^2$$
  $$H(A) = H_b(\pi_A) + \pi_A\, H\!\left(P(A \to \cdot\,\cdot)\right) + (1-\pi_A)\, H\!\left(P(A \to \cdot)\right)$$
  The penalty pushes the parameters toward low-entropy, near-deterministic productions; penalizing $H_b(\pi_A)$ as well prevents the model from hiding probability mass in the family gate.
- **Optimization**: AdamW with L2 weight decay, batches grouped by sentence length (homogeneous CYK charts), gradient clipping at `max_grad_norm = 1.0` (`clip_grad_norm_` after `backward()`). Linear **temperature annealing** $\tau : \tau_{\text{start}} \rightarrow \tau_{\text{end}}$ is applied per epoch by `synthesize.py`, and the final $\tau$ is the one passed to extraction, so the pruned grammar reflects the distribution that was actually optimized.
- **Validation Metrics** (returned by `evaluate()`):
  | Key | Definition |
  | :--- | :--- |
  | `mean_nll_per_symbol` | $-\frac{1}{N}\sum_X \log P(S \Rightarrow^\ast X)$, $N$ = total symbols. |
  | `ppl_per_symbol` | $\exp(\text{mean\_nll\_per\_symbol})$: **marginal** perplexity per symbol. It is *not* an autoregressive conditional perplexity ($P(x_i \mid x_{<i})$ is not available from a chart parser), and cross-entropy on a corpus has no upper bound — so no closed-form PPL cap may be asserted. |
  | `model_coverage` | Fraction of sentences with non-zero inside probability (a *dense* PCFG gets 1.0 on any bracket string, so this measures nothing about selectivity). |
  | `rule_sparsity` | Fraction of the $K^3 + K|V_T|$ possible productions with mass below $10^{-4}$. |
  | `max_lhs_mass_error` | $\max_A |\sum_\alpha P(A \to \alpha) - 1|$: the distribution invariant of §1.1. |
  
  The former `perplexity` / `parsing_f1` keys were removed: `parsing_f1` equalled `model_coverage` and was therefore vacuous. Selectivity is now measured by `reference.generate_negative_controls` + `GrammarVerifier.verify_negative_controls`, and structural quality by `reference.gold_rule_f1`.

### 1.4 `reference.py` - Ground Truth and Negative Controls
- `GOLD_RULE_TRIPLES` / `gold_dyck2_grammar()`: the ground-truth Dyck-2 CNF with **5 non-terminals**
  $$S \to S\,S \mid O_R C_R \mid O_S C_S,\quad C_R \to S\,C_R \mid \texttt{)},\quad C_S \to S\,C_S \mid \texttt{]},\quad O_R \to \texttt{(},\quad O_S \to \texttt{[}$$
  The previously shipped 4-non-terminal "prior" was **not** a Dyck-2 grammar: it forced one closing non-terminal to emit both `)` and `]`, so it generated `(]`. That is why the induced grammar could never be sound.
- `apply_gold_prior(parser, nt_labels)`: soft-logit initialization used by `--init prior` (logit $3.0$ on gold rules, `family_logits` $\pm 4.0$). It is a disclosed inductive bias, not a hidden oracle for the metrics: gold rule F1 is computed afterwards against the same gold set, and `--init random` is reported as an ablation (§4.4).
- `generate_negative_controls(positives, max_controls=200, seed=17)`: perturbations of corpus sentences (bracket-type swaps, insertions, deletions, reversals, random words), each filtered by `is_not_in_dyck` so only genuinely out-of-language strings enter the control set.
- `rule_set(grammar, probability_floor)` and `gold_rule_f1(induced, probability_floor=1e-6)`: precision / recall / $F_1$ of the induced rule set against the gold CNF, maximized over the **injective assignments of induced non-terminals into gold non-terminals**
  (`permutations(GOLD_NON_TERMINALS, len(induced.non_terminals))`, exhaustive: $5! = 120$ for $K = 5$;
  when $K_{\text{induced}} > K_{\text{gold}}$ it falls back to all assignments and refuses above
  $4 \times 10^{5}$ combinations). The alignment direction matters: permuting only the induced names —
  as an earlier revision did — makes the metric collapse to $0$ whenever the induced labels differ from
  the gold ones. `tests/test_reference.py::test_gold_rule_f1_alignes_renamed_non_terminals` locks it: an
  isomorphic copy of the gold CNF with renamed non-terminals must score $F_1 = 1.0$.

### 1.5 `verification.py` - Formal Testing and Model Checking
- `verify_lark_compilation()`: the exported EBNF compiles as a Lark Earley grammar. **Scope: compilability only** — it proves nothing about the generated language, and the old claim "zero syntactic hallucinations" derived from it was unfounded.
- `verify_pcfg_distribution()`: distribution invariant of the extracted grammar (mass $= 1$ per LHS within `1e-6`).
- `verify_corpus_coverage(sequences, engine="cyk")`: for each sentence, `PCFGrammar.derive` builds a derivation proof; `engine="lark"` keeps the Earley recognizer as a cross-check.
- `verify_negative_controls(sequences)`: rejection rate of the negative controls (a string is rejected iff `derive` returns `None`).
- `find_counterexample(max_length)`: exhaustive enumeration of all words up to `max_length` over $V_T$, generated-by-$G$ (with derivation tree) and ill-bracketed → the strongest possible refutation, an observable counterexample.
- `verify_language_soundness(search_bound)`: **grammar-level** soundness, two phases.
  1. *Witness search* (as above). A hit returns `STATUS_REFUTED` with the word and its tree.
  2. *Inductive certificate over a counter abstraction*. For each non-terminal $A$, ask Z3 for a signature $(\delta_r, \delta_s, m_r, m_s)_A$, where $\delta_r/\delta_s$ are the net round/square bracket deltas and $m_r/m_s$ the minimum prefix balance produced by $A$. Transfer constraints mirror the grammar:
     - $A \to a$: $\delta = e(a)$, $m = e(a)$ with $e$ the one-token effect ($e(\texttt{(})=(1,0)$, $e(\texttt{)})=(-1,0)$, $e(\texttt{[})=(0,1)$, $e(\texttt{]})=(0,-1)$);
     - $A \to B\,C$: $\delta = \delta_B + \delta_C$ and the prefix-minimum transfer $m = \min(m_B, \delta_B + m_C)$ (encoded as a conditional `If` that takes the true minimum).
     If the system is satisfiable and the start symbol satisfies $\delta = 0 \wedge m \ge 0$, no sentential form derivable from $S$ can violate the Dyck prefix invariant — by induction on derivation depth, for **arbitrary** length — hence $L(G) \subseteq \mathrm{Dyck}\text{-}2$ and the report is `STATUS_PROVED` (`UNSAT`).
  3. If no certificate exists and no bounded witness was found, the verdict is `STATUS_INCONCLUSIVE`: the certificate is *sound but incomplete* (a single signature per non-terminal under-approximates the set of contexts; some correct grammars admit none). It can never yield a false `PROVED`.
- `counter_signature(tokens)` / `is_well_bracketed(tokens)`: the counter abstraction and its invariant, shared by the certificate and the witness check.
- `check_word_balance_z3(tokens)`: diagnostic only, for a single string. The prefix invariant sits in the **thesis** and premise consistency is guarded (`INCONSISTENT_PREMISES`), because the previous encoding put the invariant among the *premises*: for `())`, `)(` and `][` the premise set was inconsistent, so `UNSAT` was returned vacuously and the check reported "sound" for malformed strings. `tests/test_verification.py` locks this regression.

### 1.6 `tokenizer.py` - Vocabulary Learned From the Corpus, Not Imported

The pure model track (see `doc/PURE_MODEL_ROADMAP.md` and section 11 of `README.md`) needs a
subword vocabulary that is **trained on this corpus** with no pretrained artifact anywhere.
`tokenizer.py` implements that vocabulary, format id `lingologos-bpe-1`, in plain Python: no
`from_pretrained` call exists in this repository and the file has no dependency outside the
standard library.

- **Special tokens**: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<RULE_START>`, `<RULE_END>`, the six
  entries of `SPECIAL_TOKENS`; ids are reserved first so that they never collide with learned pieces.
- **Hybrid (protected-atom) byte-pair encoding**. Pre-tokenization is the reversible split
  `_UNIT_PATTERN = \w+ | \s+ | [^\w\s]`, so `" ".join(pre_tokenize(text)) == text` holds by
  construction and no character is ever dropped or invented. A unit is **atomic** when it is a
  whitespace run, a run of digits, one of the four algebra brackets, a declared operator or leaf
  name, or any other unit observed in the corpus that the caller declared atomic. Atomic units are
  put in the vocabulary whole and are **never touched by a merge**: merges only ever combine pieces
  inside one word unit. This keeps the formal alphabet exactly one token per symbol, which is what
  the decoding mask of section 1.7 assumes.
- **Vocabulary construction**: specials, then the four brackets, then the declared atomic symbols
  and every atomic unit observed in the training texts, then the base characters (space, newline,
  carriage return, tab, digits). Merges are then greedy and deterministic: the pair with the highest
  frequency, ties broken by the pair itself in ascending order, is appended repeatedly until the
  target size is reached, no mergeable pair is left, or the best pair falls below `min_frequency`.
  `stop_reason` records which case happened.
- **Measured behaviour on the shipped corpus** (`python datasets/multilingual/tools/train_tokenizer.py`, gate
  output in `datasets/multilingual/checkpoints/tokenizer/metrics.json`): 72,840 training texts and 5,214,389
  pre-tokenized units give 1,578 word types and 3,040,076 atomic units (268 atomic types), and the
  vocabulary **saturates at 3,921 entries with 3,578 merges** (`stop_reason: no mergeable pair
  left`). A target of 8,000 to 32,000 pieces, as usually written for text models, is not reachable
  on this corpus: the requested range is above saturation, so the honest number is the saturation
  point. Gates `T1` reversibility, `T2` zero unknown tokens (`0` over 5,786,067 produced tokens),
  `T3` every formal symbol atomic (267 of 267), `T4` algebra symbols of the held-out splits already
  in the vocabulary, `T5` bracket-word round trip (`13,490` checked, `0` broken) all pass.
- **Serialization**: `save` / `load` write and read a single UTF-8 JSON document with the format id,
  the vocabulary, the merge list and the statistics; `load` refuses a document whose format id is
  not `lingologos-bpe-1`, so a foreign vocabulary cannot be mistaken for this one.

### 1.7 `masking.py` - The Symbolic Constraint That Replaces the Penalty Term

The specification asks for a symbolic term in the loss and for masked logits
$L_{t,i} = -\infty$ when token $i$ is not admissible after prefix $t$. `masking.py` makes that
constraint *exact* rather than soft, and it is deliberately **model-free**: it contains no neural
module, so it is importable and testable before any network exists.

- **Two levels.** `AlgebraMask` is a stack automaton over the *symbol* alphabet (four brackets, the
  53 operator names, the leaf tags and leaf payloads). `TokenStreamMask` lifts it to the vocabulary
  of section 1.6, emitting the canonical protocol `<RULE_START> s1 s2 ... sn <RULE_END> <EOS>` with
  exactly one space token between two symbols.
- **Why the automaton is prefix-deterministic.** The arities live in `datasets/multilingual/templates/operators.json`
  and every operator name has fixed arity, so after the symbol that follows `(` the number of
  remaining children is known and with it every closing bracket of that subtree. A leaf is `[`
  followed by a tag and payload tokens whose admissible lengths come from a *shape profile* learned
  from the corpus, so the position of `]` is decidable from the prefix alone. Consequence: the set of
  admissible next symbols is a function of the prefix, no backtracking needed, and legality can be
  decided in one left-to-right pass.
- **Admissible language.** A stream is accepted exactly when it is the canonical serialization
  `logic_map.bracket_ast(tree)` of a well-typed tree: root is always a round node, brackets are
  properly matched (Dyck-2 with `(` `)` and `[` `]`), a round node holds exactly `arity(name)`
  children, a leaf never nests, and the leaf payload length is in the observed profile. `validate_stream`
  returns the tree, so acceptance is constructive; `sample_stream` draws uniformly over the legal
  continuations with a depth budget, which yields guaranteed-legal training data.
- **Token-level mask.** `mask_logits(logits, prefix_ids)` fills the rows outside the admissible set
  with $-\infty$ and leaves the admissible scores untouched, so `softmax` is a distribution over the
  legal choices only and the entropy of the policy is measured on real alternatives. A vector that is
  narrower than the vocabulary, or a batch of logits, is a hard error rather than a silent slice.
  `<RULE_END>` is offered *only* when the automaton is complete, and after it only `<EOS>`; the
  automaton never offers a second root.
- **Measured behaviour** (`python datasets/multilingual/tools/verify_masking.py --refine`, gate output in
  `datasets/multilingual/checkpoints/masking/masking_metrics.json`): mask alphabet 53 operators, 136 leaf tags,
  75 payload tokens, leaf length profile `{1: 135, 2: 1}`, 268 vocabulary tokens for the masked
  alphabet. `M1` held-out algebra: 1,350 of 1,350 accepted. `M2` negative controls: 16,350 mutated
  streams, 16,350 refused against an **independent** oracle that reuses `logic_map.is_well_bracketed`,
  the registry arities and the reader of `logic_map` instead of the mask itself; mutations the oracle
  judges legal are discarded instead of being counted, which is why `unknown_leaf` reports 0 kept out
  of 2,700 attempts (the held-out leaves are all inside the training alphabet). `M3` 300 sampled
  streams, 300 valid trees and 300 well bracketed. `M4` 52 distinct operator/arity pairs observed in
  the corpus agree with the registry, 0 mismatch (of 53 registered operators, one never occurs in the
  corpus). `M5` token protocol: 67,190 gold decisions on the held-out split, 67,190 accepted, closing
  marker forced at completeness 1,350 times, premature closing marker offered 0 times. With
  `--refine` the extra restriction "which child kind sits at which position" is learned from the
  training split only (83 refined `(operator, position)` pairs) and the held-out split is still fully
  accepted, so the refinement is sound and not over-fitted to the test data.
- **The constraint is portable.** `to_spec()` serializes the automaton: operator arities, the leaf forms
  with their payload token ids, the leaf tag tokens, the observed child kinds keyed `NAME <position>`,
  and the leaf length bound. `AlgebraMask.from_spec` rebuilds an identical automaton from that JSON and
  refuses a specification that is missing a key or that carries a malformed child-kind key, so
  `mask_spec.json` in a release reconstructs the same mask, token ids included, without the templates
  and without the corpus.
- **Consequence for the loss.** With the hard mask in place during both training and inference, the
  symbolic term $\lambda L_{\mathrm{Symbolic}}$ of the specification becomes redundant rather than
  wrong: the invalid region of the output space has zero probability mass by construction, so the
  cross-entropy alone already operates on legal continuations. The mask is therefore the primary
  mechanism, and a soft penalty stays optional.

### 1.8 `model.py` - The Network, Written From Scratch

`model.py` is the whole neural side of the pure-model track: one `ModelConfig`, four layer classes and
one `EncoderDecoderModel`. There is **no** `from_pretrained` call, no external weight, no tokenizer or
embedding imported from a hub; the checkpoint metadata states `"external_weights": "none"` and the
test suite asserts that an archive whose format name is not `lingologos-transformer-1` is refused.

- **Geometry** (the specification, kept): 6 encoder blocks plus 6 decoder blocks, 8 heads of
  $d_{head} = 64$, $d_{model} = 512$, $d_{ff} = 2048$, dropout 0.1, learned absolute positions capped
  at 96 in both directions (measured maxima 68 encoder and 92 decoder protocol tokens, so nothing is
  ever truncated). `ModelConfig.__post_init__` rejects a vocabulary smaller than 2, non-positive
  widths or position budgets, $d_{model}$ not divisible by the head count, fewer than one layer on a
  side, and dropout outside $[0, 1)$, raising `ModelError` rather than letting a shape error surface
  300 lines later; `from_dict` refuses unknown or missing keys, so a checkpoint cannot quietly carry a
  renamed hyperparameter.
- **Blocks.** Pre-norm (LayerNorm before attention and before the feed-forward, residual after),
  attention through `F.scaled_dot_product_attention` with the causal or padding mask passed as an
  additive float mask and dropout applied only in training mode. The decoder block carries an optional
  cross-attention sub-layer against the encoder memory.
- **Embeddings.** Encoder and decoder keep **separate** token tables (the encoder reads free text, the
  decoder emits the protocol alphabet), each scaled by $\sqrt{d_{model}}$, with `padding_idx` on the id
  of `<PAD>` and that row pinned to zero. The decoder input table is tied to the output projection,
  which saves exactly $V \cdot d_{model} = 2{,}007{,}552$ parameters. Total measured count:
  **48,253,952** trainable parameters, from `parameter_breakdown`: encoder blocks 18,914,304, decoder
  blocks 25,224,192, token tables 4,015,104, position tables 98,304, final layernorms 2,048.
- **Initialization, per the specification and measured.** `apply_initialization(model, seed)` is
  deterministic in the seed: LayerNorm gains 1 and biases 0; embeddings and position tables
  $N(0, 0.02)$; every linear matrix gets Kaiming (He) if it is the ReLU-side inner projection of a
  feed-forward block and Xavier otherwise, biases uniform in $\pm 1/\sqrt{fan_{in}}$; the projection is
  re-tied afterwards. In PyTorch 2.14 the He branch applies the ReLU gain, so the measured bound is
  $\sqrt{6/fan_{in}}$ (0.3062 at $fan_{in} = 64$) and not the smaller $\sqrt{5}$-adjusted bound; the
  tests encode the measured distinction. `initialization_report` at the shipped configuration returns
  100 matrices, standard deviation 0.044993, no non-finite parameter, matrix rows 96 to 3,921.
- **Masks are shapes, not conventions.** `causal_mask(T)` is $[1,1,T,T]$ and `additive_padding_mask`
  is $[B,1,1,S]$, both additive with $-\infty$; the tests assert those shapes and assert that the
  decoder output at position $t$ does not change when a later position changes (causality checked in
  `eval()` mode, where dropout cannot mask the effect).
- **Persistence for training.** `save_model` writes a `torch.save` archive with the state dict, the
  config dict, the initialization statistics, the mask and vocabulary reports, the hardware probe and
  the training settings; `load_model` reads it with `weights_only=True` and refuses an archive from
  another track. That archive remains the internal working format, because it also carries the optimizer
  state a resumed run needs.
- **Persistence for release: `safetensors`.** Published weights ship as `model.safetensors`, written by
  `save_model_safetensors`. It hands the whole `nn.Module` to `safetensors.torch.save_model`, so a tied
  tensor is stored **once**: at the shipped configuration the file holds 260 tensor entries for the 261
  names declared by `state_dict`, and the write path raises when the file does not contain exactly as
  many tensors as the model has distinct storages. The header carries four string fields: the format tag
  `lingologos-transformer-1`, the configuration as JSON, the padding id, and caller metadata.
  `load_model_safetensors` reads that header, rebuilds the architecture, and loads with `strict=True`,
  which re-establishes the tie; `read_model_safetensors_header` exposes the header without the weights
  and rejects a file whose tag is not this track or that has no configuration. Measured: 193,044,672
  bytes for 48,253,952 distinct parameters, and a reload that reproduces every tensor and every logit of
  a fixed probe batch exactly.
- **Tests**: `tests/test_model.py`, 20 checks (config guards, config round trip, forward shapes, budget
  refusal, causality, mask shapes, invariance to encoder padding, zero `<PAD>` row, tying and the
  untied variant, seed determinism of initialization, LayerNorm neutrality, the He versus Xavier
  distinction, bias bounds, gradients on every parameter, the parameter breakdown identity, save and
  load round trip with `torch.equal`, and refusal of a foreign archive).

### 1.9 `training.py` - Masked Supervision, Schedules and Constrained Decoding

`training.py` connects the corpus, the automaton and the network. It holds no hyperparameter policy of
its own: the script `datasets/multilingual/tools/train_model.py` owns the command line, this module owns the
arithmetic.

- **Examples and labels.** `AlgebraExample(identifier, source_text, symbols, language, domain,
  logic_family)` is one supervised pair; `protocol_of` builds `<RULE_START> ... <EOS>`;
  `label_allowed_sets` walks the automaton **once** per target and returns the admissible token set of
  every predicted position, so a target of length $n$ yields $n-1$ labels, $n-1$ admissible-set rows
  and the boolean `free` flag telling which of those positions had a real choice. On the shipped
  corpus that share is **0.1966**: 80.3 percent of supervised positions are singletons the automaton
  forces (spaces, the closing marker, `<EOS>`, the payload continuation of a leaf).
- **Admissible sets as a table, not a loop.** `AllowedSetTable` interns every distinct admissible set
  into an integer row (the corpus needs **13** rows: widest 136 tokens, narrowest 1, 8 single-choice
  rows) and materializes one dense boolean matrix $[13, V]$ on the device. Batching gathers the rows
  for the recorded states, so masking costs one `masked_fill_` over $[B, T, V]$ instead of $B \cdot T$
  automaton calls. Padding writes `rule_start_row` and the start marker so a padded label never
  receives an all-$-\infty$ row; `position_mask` removes those positions from the loss.
- **The loss and its metrics.** `masked_logits` clones the logits and fills inadmissible entries with
  $-\infty$, so cross-entropy is computed over legal alternatives only and perplexity is perplexity
  over those alternatives. `score_batch` reports loss, perplexity, exact target-sequence accuracy
  computed on the **masked** argmax, free-position accuracy, the count of generated sequences, and
  `illegal_unmasked_argmax_rate`: how often the raw, unmasked argmax falls outside the admissible set.
  The last number is not a bug report but a measurement of the division of labour: at convergence the
  model scores an inadmissible token highest at 83.4 percent of positions while its masked accuracy is
  99.9 percent, because masked cross-entropy never penalizes an inadmissible logit.
- **Optimization.** `build_optimizer` is AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.98$,
  $\epsilon = 10^{-9}$, weight decay 0.01 on weight matrices and zero on biases and layernorm gains
  (two parameter groups, asserted by the tests). `cosine_schedule` is a `LambdaLR` that ramps linearly
  over the warmup steps and then decays to `final_fraction` of the peak rate; a warmup longer than the
  total is repaired rather than producing a zero learning rate.
- **Constrained decoding.** `generate_batch` decodes a whole batch under the automaton: each row keeps
  its own prefix, the position of the next symbol is `len(prefix) - 1`, the encoder memory is gathered
  by the active rows, and every emitted token is drawn from the admissible ids of that row's state, so
  an illegal continuation is unreachable rather than improbable. Greedy and temperature sampling are
  both seeded; `max_steps` is checked against the decoder position budget instead of silently
  truncating.
- **Tests**: `tests/test_training.py`, 19 checks, including a **non-vacuous** overfit: 12 sentence and
  stream pairs trained for 130 epochs (390 updates) reach loss below 0.05, exact-sequence accuracy 1.0,
  and greedy constrained decoding reproduces all 12 streams byte for byte. Legality of generated
  prefixes is re-checked with `AlgebraMask.replay` on untrained weights, and every corrupted stream is
  refused by `validate_stream`.

### 1.10 `asic_chip/` - The Hardware Track

The same three formal mechanisms the software induces and verifies are also described as a
chip: a fixed function ASIC in SystemVerilog targeting the SkyWater `sky130A` PDK through
OpenLane. The tree is not hand written where it can be generated. Three JSON artifacts under
`asic_chip/artifacts/` hold the design data, and `asic_chip/scripts/lingologos_compiler.py`
is the only description of the datapath: it emits the three generated modules, the golden
vector table, the core self test table and a manifest with the digest of every file. The
RTL is checked in, so the checkout is readable, and `--check` re-derives every byte of it
from the artifacts, which is what makes a hand edited generated file fail instead of drift.

| Artifact | Holds | Feeds |
| --- | --- | --- |
| `model_weights.json` | a 4 by 4 matrix of signed 8 bit weights, schema `lingologos-weight-mesh-1` | the constant weight mesh |
| `gf_rules.json` | field polynomial `0x11D`, initial state, state budget 255, and the four bracket symbols with their multipliers | the bracket automaton |
| `smt_constraints.json` | eight input bits mapped to variables `A` to `H`, four result lanes, and the clause list in the constraint DSL | the combinational clause evaluator |

**The weight mesh.** The weights are compile time constants, so a product `w * X_c` is never
a multiplier: it is a shift-add tree built from the non-adjacent form of `w`, which has the
fewest non-zero digits and no two adjacent ones, so the tree alternates additions and
subtractions of shifted copies of the single operand. Each row sums four such trees into a
signed 20 bit accumulator. The generator attaches to every product a partial-sum bound of
the form `B * |X_c|` and gate `H1` proves the accumulated bound of a row fits the
accumulator: measured maximum shift 7, worst row bound 35072 against the limit 524287,
headroom ratio 14.95. Every wire carries that bound in its comment, and the gate suite
recomputes the bound from those comments and compares it with the datasheet.

**The bracket automaton.** Bracket matching is a product in `GF(2^8)` modulo the irreducible
polynomial `0x11D`: an opener multiplies the state by `0x02`, a closer by its inverse `0x8E`,
and the word is balanced exactly when the state returns to the initial value. With one state
byte the machine has 255 reachable non-zero states and order 255 for the generator `0x02`, so
the transition table is total: 255 states times 4 alphabet symbols is 1020 defined arms plus
one default arm that resets. The generator expands the whole table into
`case` arms of a single combinational block, which makes the step one clock with no
multiplier and no iteration, and the limitation is a property of the one byte budget rather
than of the table: round and square brackets share the same two multipliers, so the machine
recognizes the commutative image of Dyck-2 and accepts `(]`.

**The clause evaluator.** The constraint DSL has prefix `NOT`, prefix and infix `MUX`, and
`AND`, `OR`, `XOR` as named functions, with precedence `MUX` lower than `OR`, `OR` lower than
`XOR`, `XOR` lower than `AND`. `select ? set : clear` is the semantics of `MUX`, and each
clause compiles to a mask over the 8 input bits, so it becomes a pure tree of one bit gates:
wires only, no `always` block, no latch, no clock. Depth is a property of the tree, measured
on the dependency graph of the generated wires: 3 levels maximum for the shipped clause set,
with 3, 2, 4, 3 nodes per clause.

**The core.** An RV32E machine, embedded variant, single cycle, with the minimal implemented
subset documented in the header of `rtl/core/rv32e_core.sv` and in section 6.1 of the
integration guide: `LUI`, `AUIPC`, `JAL`, `JALR`, the six branches, the five loads and three
stores, the immediate and register arithmetic, logic and shift forms, and `FENCE` as a no-op.
The coprocessor is reached through the custom-0 opcode `0b0001011` with five instructions,
`LLSMT` for the four clause results of a byte, `LLGFS` for one field step, `LLRDM` for a
signed readback of a mesh row, `LLCTL` for the access counter and the sticky illegal flag,
and `LLHALT`; each one drives the request for a single cycle and takes its result in the same
cycle, which is what zero wait states means here. Anything the subset does not implement
traps: a 5 bit register field of 16 or above, which is the RV32E register-space rule, any
64 bit operation, any system or privileged instruction, `FENCE.I`, and a load or store that
is not naturally aligned. The trap is sticky and reaches bit 4 of the coprocessor status
word. The firmware is 97 words assembled into the instruction ROM as an initializer, word 79
is a deliberate illegal probe `ADDI x1, x16, 0`, and the self test writes the mesh result, the
clause lane word and the GF state back through the window. There is no bus and no external
memory: program and data live in on-chip arrays, 128 instruction words and 64 data words.

**Reproducibility.** `rtl/generated/MANIFEST.json` records the schema id, the digest of each
input artifact, the digest of each generated file, and the measurements of the five generator
gates `H1` to `H5`. The generator writes LF line endings and UTF-8 without a byte order mark
on every platform, so the recorded digest of a generated file equals what `sha256sum` prints
for the file in the checkout, and gate 7 of the suite verifies exactly that over the raw
bytes. `.gitattributes` pins `text eol=lf` for the whole `asic_chip/` tree, because a Windows
checkout with `core.autocrlf` would otherwise rewrite the input artifacts, the generated RTL and
the memory images to CRLF: the digests would stop matching the bytes on disk and the generated
files would look stale on that host alone.

**Status of this track, stated without inflation.** Python carries the generator, its five
gates, and 23 tests that recompute the datapath, the field, the clause masks and the
instruction semantics from an independent model. Verilator 5.054, run from the image
`verilator/verilator:latest`, linted the design clean under `-Wall` and replayed all 304
golden rows plus the core self test in two independent paths, 0 mismatches, and along the way
it reported defects in the RTL and in the checkers, items 7 to 16 of section 4 of the report,
all fixed. The first place and route attempt, on 2026-11-09, reached the 25th step of the
OpenLane 2 Classic flow: Yosys mapped the design into `sky130_fd_sc_hd` at 11,015 cells on
10,938 nets and 149,098 square micrometers of cell area, the lint, the checkers, the floorplan,
the tap and cap rows, the power distribution network, the global placement and the pin placement
completed, and the step that places the pins from `openlane/pin_order.cfg` stopped the run on a
file written in the format of OpenLane 1. That step reads the file with a grammar of its own,
which has no comment token and matches anchored regular expressions against the bit level names
of the netlist, and section 5 of `asic_chip/doc/build_instructions.md` now records that grammar
and the rewritten file. After that file was rewritten, the same command ran all 78 steps of the
same flow twice more on the same day, in 16 min 59 s and in 12 min 12 s, and both produced a layout:
three GDSII files, a DRC report at zero violations in Magic and in KLayout, an LVS report reading
`Final result: Circuits match uniquely.`, a static IR drop report, and a static timing report over nine
corners. The first of those two, at the 8.0 ns bring-up period then in force, left a signoff summary
line saying `Antenna Failed` on 54 pins and 46 nets, and worst setup slack of -4.4409 ns in the typical
corner and -15.7205 ns in the slow one. The second, at a 30.0 ns gate, is the run whose manufacturability
report reads `Antenna Passed`, `LVS Passed` and `DRC Passed`; it still exited non zero, on setup in one
corner at -0.4906 ns in `max_ss_100C_1v60` and on hold in seven corners at a worst -0.1540 ns, every
violating path from an input port to a register. The hold side of that second run is what moved two
resizer margins: the resizer closed hold at +0.100201 ns in `nom_tt_025C_1v80` and the same corner read
-0.058548 ns at signoff, because detailed routing and extraction took about 0.16 ns out of those paths
and the Classic flow repairs hold nothing after them. No second simulator has been run, no run here has
closed signoff, and no maximum operating frequency is
claimed anywhere in this tree: the 4.0 ns clock period in the SDC is a declared budget, and what is
measured is the slack a layout produced at a stated period and corner. The section 5 script runs a
second, deliberately looser constraint set by default, 32.0 ns in `config.bringup.json` and
`lingologos.bringup.sdc` with `PL_RESIZER_HOLD_SLACK_MARGIN` and `GRT_RESIZER_HOLD_SLACK_MARGIN` at
0.40 ns, selected against the declared one with `LL_OPENLANE_CONFIG`, and a one
command setup script installs the toolchain that makes any of that runnable. Three things those
complete runs taught about the configuration files, and all three are corrected in them: `PNR_SDC_FILE` and
`SIGNOFF_SDC_FILE` were undefined, which sends the placement and routing steps to a generic
constraint file instead of this design's, and both files now name the design constraint file under
those keys; `QUIT_ON_TIMING_VIOLATIONS` and `RUN_CVC` are keys OpenLane 2.3.10 does not read at all,
so they never gated anything and are gone, while what does gate timing is the corner list, and both
files now set `SETUP_VIOLATION_CORNERS` and `HOLD_VIOLATION_CORNERS` to every corner instead of
inheriting the `["*tt*"]` that the PDK compatibility layer installs; and the OpenLane 1
`DIODE_INSERTION_STRATEGY` key is accepted, deleted and recompiled into the native antenna knobs,
which for the value 3 these files carried disabled the heuristic diode insertion and the diodes on
ports the same files asked for, so the native knobs are written directly now. That script also selects the interpreter the toolchain is installed on, 3.12 at
best, because the published wheel sets of the two compiled dependencies of the `openlane` package
stop at cp313 for `klayout` 0.29.12 and at cp312 for `libparse` 0.3.1, and both source
distributions failed to compile here. When the package manager cannot provide such an interpreter
the script obtains one itself, first from the archive Ubuntu defers to for older Pythons and then
as a standalone build under `~/.local/share/lingologos`. That interpreter is also required to
import `tkinter`, which the `openlane` package imports while starting up and which Debian and
Ubuntu package apart from the interpreter. The flow entry point then looks for the installed
`openlane` in `LL_OPENLANE_BIN`, in `~/openlane-venv/bin` and on `PATH`, in that order, and invokes
one it finds in that virtual environment with `--dockerized`, because the package brings no tools
of its own and the container image does. `asic_chip/doc/verification_report.md`
separates executed from declared claim by claim.

---

## 2. Step-by-Step Configuration and Training Guide

### 2.1 System Requirements and Dependencies
- Python 3.10+ (tested up to Python 3.12).
- Core packages:
  - `torch >= 2.0.0`
  - `z3-solver >= 5.0.0`
  - `lark >= 1.2.0`
  - `pyyaml >= 6.0`
  - `numpy >= 1.24.0`
  - `safetensors >= 0.4.0` - release format of the pure-model weights, section 1.8
  - `httpx >= 0.24.0` - client of the federated HTTP transport of section 13.5, used only when a node
    declares `"transport": "http"`
  - `pytest >= 8.0.0`
- The federated Git transport of section 13.5 shells out to `git`, an external program that must be on
  the `PATH` for a node declaring `"transport": "git"`; no other Python package is involved.
- Optional, used only when publishing: `huggingface_hub >= 0.23.0`, installed with
  `pip install -e ".[publish]"`. `datasets/multilingual/tools/publish_to_hub.py` imports it inside the upload step and
  nowhere else, so building a release folder needs no Hub package and no network access.
- The `asic_chip/` track of section 1.10 adds no dependency at all: its generator uses only the Python
  standard library and its gates run inside `pytest`. The tools that would take it further, Verilator,
  Icarus Verilog, Yosys and the sky130A PDK, are external programs and none is installed in the
  checkout where this track was written. OpenLane is the exception in form: `pip install openlane`
  gets version 2, which drives the tools of every step inside containers, and its compiled
  dependencies limit the interpreter of that environment to 3.12 or older on this platform. None of
  them is installed here. Section 2.5 gives the commands and marks what was actually run.

Installation:
```bash
pip install -r requirements.txt
pip install -e .
```

### 2.2 Running Training and Synthesis Pipeline
To train the model, extract the discrete grammar, and run the four acceptance gates:
```bash
python3 src/neuro_symbolic/synthesize.py
```
This command generates the following files in `artifacts/` (the directory is resolved against the
repository root, so the command also works from a different working directory):
1. `artifacts/grammar_weights.pt`: PyTorch checkpoint of `rule_logits`, `term_logits`, `family_logits` and the run configuration.
2. `artifacts/grammar.json`: JSON CNF grammar specification (with `probability` and `probability_pre_pruning`).
3. `artifacts/grammar.yaml`: YAML grammar specification.
4. `artifacts/DynamicGrammarIta.gf`: Grammatical Framework abstract/concrete code.
5. `artifacts/metrics.json`: machine-readable report with sections `config` (all hyperparameters and seeds),
   `metrics_train` / `metrics_test`, `grammar` (rule counts, `valid_pcfg`, `lhs_masses`, `lark_compiles`),
   `language` (coverage, negative-control rejection, gold rule precision/recall/$F_1$),
   `soundness` (status, certificate, counterexample, search bound, human-readable English `notes`) and `gates` (`G1`–`G4` booleans).

**Acceptance gates** (`G1` grammar validity, `G2` 100% corpus coverage, `G3` negative-control rejection
plus gold rule $F_1$, `G4` Z3 soundness `UNSAT`): the process exits with a **non-zero code** if any gate
fails, which is what CI enforces; `--lenient` downgrades that to a warning. `--list-only` re-loads an
existing checkpoint/artifacts and re-runs verification without training. All hyperparameters are
exposed as flags (`--init`, `--epochs`, `--lr`, `--tau`, `--tau-start`, `--entropy-penalty`,
`--prune`, `--search-bound`, `--num-nt`, `--test-fraction`, `--seed`, …); the full matrix is in
`README.md` §5 and `--help` is authoritative. Every piece of runtime prose is English ASCII, as
required by `AGENTS.md` §2: the `--help` text, the progress lines, the gate labels (`G1a` Lark
compilation, `G1b` valid PCFG, `G2` corpus coverage, `G3` negative controls, `G3b` gold rule $F_1$,
`G4` SMT inclusion check) and the final verdict line `FINAL VERDICT: PASS|FAIL`. The `SoundnessReport`
notes and the assertion messages of the test suite follow the same rule (formal operators are spelled
out, for example `P AND NOT G`, so that Python sources stay ASCII only).

### 2.3 Running Regression Tests and CI
Execute all unit and integration tests locally:
```bash
pytest -v tests/
```
The repository includes `.github/workflows/ci.yml` running pytest and the synthesis pipeline on GitHub Actions across Python versions 3.10, 3.11, and 3.12.

### 2.4 The `datasets/multilingual/` Companion Folder and its Release

`datasets/multilingual/` is a companion folder: it holds a synthetic multilingual corpus (Italian, English, French, Spanish, German) built from 50 sentence templates over five everyday domains, plus the small scripts that generate, validate, and learn from it. Nothing under `src/` reads it while the engine runs, and nothing in it changes the Dyck-2 engine: the bracket algebra it stores uses the same terminal alphabet as `src/neuro_symbolic`, which is why the trainer entry points reuse the existing parser, trainer, grammar and verifier modules unchanged.

Corpus and tools are versioned in this repository. The packaged files are published as GitHub Release assets under incrementing tags: the first release is `dataset-v1`, the next ones continue as `dataset-v2`, `dataset-v3` and so on (`https://github.com/tailot/LingoLOGOS/releases/latest` always points at the newest). A release carries `lingologos-dataset.zip` (data, chat pairs, corpus, configs, templates, tools), `grammar.json`, `grammar.yaml`, `DatasetGrammar.gf`, `grammar-metrics.json`, `tokenizer.json`, `tokenizer-metrics.json`, `masking-metrics.json` and `dataset-stats.json`. The files are free to download. `.github/workflows/dataset.yml` produces them in three stages: consistency check plus tokenizer and mask gates, grammar induction, and the release itself.

The corpus folder sits at `datasets/multilingual/` and the orchestrator workspace keeps its own per-node folders (`datasets/<node_id>/`) beside it. The orchestrator resolves only the node ids declared in `config/topology.json`, so the corpus folder is never taken for a node. Path resolution is not positional either: every tool in `datasets/multilingual/tools/` starts from its own file and walks up until it finds the corpus marker `templates/operators.json` and the project root marker `pyproject.toml` with its `src` tree, so the corpus can be relocated without touching any source. `datasets/multilingual/tools/corpus_paths.py` implements that probe, exposes `repo_root`, `corpus_root` and `bootstrap`, and accepts the `LINGOLOGOS_REPO_ROOT` and `LINGOLOGOS_DATASET_ROOT` overrides, raising `CorpusPathsError` when an override does not carry the expected marker instead of reading another corpus silently. `tests/test_evaluation_tools.py` and `tests/test_hub_package.py` discover the tools under `datasets/*/tools` for the same reason.

The learning and checking entry points of that folder, all thin wrappers over the modules documented
above. The first four are model-free, the next three build and run the pure network of sections 1.8
and 1.9, and the last one publishes what they produced:

| entry point | role |
| --- | --- |
| `datasets/multilingual/tools/validate_dataset.py` | re-derives every bracket word, pattern and SMT text, and builds all 2,698 logical forms in Z3 |
| `datasets/multilingual/tools/train_grammar.py` | induces the PCFG over the corpus of the dataset with `DifferentiableCYKParser`, `GrammarTrainer`, `PCFGrammar` and `GrammarVerifier` |
| `datasets/multilingual/tools/train_tokenizer.py` | learns the byte-pair vocabulary of section 1.6 from the training split only, derives the protected alphabet from the operator registry, and runs gates `T1`-`T5` on all three splits |
| `datasets/multilingual/tools/verify_masking.py` | builds the automaton of section 1.7 from the registry plus the training split, and runs gates `M1`-`M5` against an independent oracle, including the negative controls and the token-level protocol check |
| `datasets/multilingual/tools/model_data.py` | shared loader: tokenizer, algebra mask, token mask, examples per split, length profile and hardware probe, so training and evaluation cannot drift apart |
| `datasets/multilingual/tools/train_model.py` | the training loop of section 1.9 on the real corpus: AdamW with one epoch of warmup and cosine decay, bf16 autocast, masked cross-entropy, per-epoch validation, constrained decoding slice, checkpoints under the git-ignored `datasets/multilingual/checkpoints/model/` |
| `datasets/multilingual/tools/evaluate_model.py` | gates `E1`, `E2`, `E3`, `E5` on a held-out split: exact-tree accuracy with the per-language, per-domain and per-family breakdown, legality through the independent oracle, the registry plus Z3 audit of the trees the model produced, the invented-number audit, and the mask-ablation pass at identical weights |
| `datasets/multilingual/tools/publish_to_hub.py` | assembles the release folder `model.safetensors`, `config.json`, `tokenizer.json`, `mask_spec.json`, `metadata.json`, `evaluation_report.json` and the model card `README.md`; refuses a checkpoint whose gate report is not `PASS` unless `--allow-ungated` is given, checks that the reloaded safetensors weights are bit-identical and produce identical logits, records the sha256 of every file, and uploads only when `--upload` is passed and the token is in the environment (`--token-env`, default `HF_TOKEN`). Tested by `tests/test_hub_package.py`, 11 checks |
| `datasets/multilingual/tools/corpus_paths.py` | not an entry point but the helper the other nine import: finds the corpus root by the `templates/operators.json` marker and the project root by the `pyproject.toml` plus `src` markers, walking up from the file so the folder needs no fixed position, puts both roots and the tools folder on `sys.path`, and accepts the `LINGOLOGOS_DATASET_ROOT` and `LINGOLOGOS_REPO_ROOT` overrides |

There is no chat fine-tuning stage and no third-party model in this repository: the assistant that
answers in natural language is the pure model described in `doc/PURE_MODEL_ROADMAP.md`. Phases 1 to 4
of that roadmap are measured, phase 5 is the evaluation script above together with the release
packaging below, and the verbalization direction is specified but not trained. The trained weights are
not in the repository either: they are published as `model.safetensors` in the Hugging Face model
repository `tailot/lingologos-pure-1`, next to the tokenizer, the mask specification, the gate report
and the model card assembled by `datasets/multilingual/tools/publish_to_hub.py`. Sizing notes for consumer cards, the
measured saturation of the vocabulary, the measured cost of a 20-epoch run and the full flag
documentation are in `datasets/multilingual/TRAINING.md` and in section 11 of `README.md`.

### 2.5 Building and Checking the `asic_chip/` Track

Every command runs from the repository root, because the file lists in `asic_chip/config.json`,
in `asic_chip/Makefile` and in `sim/sim_main.cpp` are relative to it.

| step | command | tool | status in this checkout |
| --- | --- | --- | --- |
| regenerate the generated RTL, both vector tables and the manifest | `python asic_chip/scripts/lingologos_compiler.py` | CPython, standard library only | EXECUTED, `FINAL VERDICT: PASS` |
| prove the checked in files are the current ones | `python asic_chip/scripts/lingologos_compiler.py --check` | same | EXECUTED, adds the `RTL_matches_artifacts` gate |
| run the gate suite required by `AGENTS.md` | `pytest -v tests/` | pytest | EXECUTED, 208 passed |
| lint and infer latches | `verilator --lint-only ... -DLL_DEBUG_PORTS --top-module lingologos_top` | Verilator 5.006 or later | EXECUTED in the image `verilator/verilator:latest`, Verilator 5.054, through `make -f asic_chip/Makefile lint`: exit 0, no warning left. NOT RUN on the host, where there is no Verilator and no `make` |
| replay the 304 golden rows against Verilog | `verilator --cc --exe --build ... sim/sim_main.cpp` | Verilator and a C++ compiler | EXECUTED in that image, through `make -f asic_chip/Makefile sim-verilator`: 304 rows plus the core self test, 0 mismatches, `FINAL VERDICT: PASS`, 91 clock cycles for the self test, VCD written |
| run the SystemVerilog testbench | `verilator --binary --timing ...` | Verilator | EXECUTED in that image, through `make -f asic_chip/Makefile sim-tb`: same checks, same verdict |
| run the testbench in a second simulator | `iverilog -g2012` plus `vvp` | Icarus Verilog | NOT RUN, Icarus is not installed |
| place and route, first attempt | `sh asic_chip/scripts/build_gds.sh` | OpenLane 2.3.10 plus the sky130A PDK | EXECUTED in WSL Ubuntu with Docker Desktop, 24 of 78 steps, stopped at the custom I/O placement step on `pin_order.cfg`, exit 65 |
| place and route, second attempt of the same day | the same script, after the pin plan file was rewritten | the same | EXECUTED end to end, all 78 steps in 16 min 59 s: three GDSII files, DRC zero in Magic and KLayout, LVS circuits match uniquely, IR drop and nine corner static timing reports, `Antenna Failed` on 54 pins and 46 nets, exit non zero on deferred setup and hold violations at the 8.0 ns gate of that run |
| place and route, third attempt of the same day | the same script, with the antenna knobs written natively and the bring-up period at 30.0 ns | the same | EXECUTED end to end, all 78 steps in 12 min 12 s: three GDSII files, DRC zero in Magic and KLayout, LVS circuits match uniquely, and a manufacturability report reading `Antenna Passed`, `LVS Passed` and `DRC Passed`; exit non zero on deferred setup violations in one corner at -0.4906 ns and hold violations in seven corners at a worst -0.1540 ns, all of them input port to register |

The three Verilator rows ran in the official container image with the checkout mounted at
`/work`, the route `asic_chip/doc/build_instructions.md` documents for a host that has no
Verilator: that image carries its own `make` and a C++ compiler, so the make targets work
there unchanged. Section 2.4 of `asic_chip/doc/verification_report.md` quotes the output those
runs produced, and section 1 of the same report lists which tools were present where.
The three place and route rows ran in WSL Ubuntu 26.04 with Docker Desktop, from the copy of this
tree that the setup script writes under `/home/<user>/LingoLOGOS`, with openlane 2.3.10 on the host
inside `~/openlane-venv` and its steps executed in the image
`ghcr.io/efabless/openlane2:2.3.10`; section 2.5 of the same report gives the three run directory
names and the reports each run left. The numbering of the steps inside a run directory changes
between runs, because the configuration enables steps, so the reports are named by step kind and not
by number.

The generator refuses to emit RTL when one of its own gates fails: `H1` the shift-add trees fit
the 20 bit accumulator, `H2` the field table is closed and irreducible, `H3` the clause network
is latch free with no unused lane, `H4` every vector row is in range, `H5` the core self test
agrees with the engines. `--lenient` reports the failing gate instead of stopping, and the final
verdict stays `FAIL`, so an experiment can inspect the numbers without producing RTL that
silently violates the contract. `--out-dir` and `--sim-dir` write to a scratch directory, which
is how the reproducibility gate regenerates without touching the checkout.

The tool chain beyond Python is external and is not a package dependency of this project: the
generator imports nothing outside the standard library, and the gate suite needs only the
`pytest` already declared in `requirements.txt` and `pyproject.toml`. That is why the ASIC gates
run in CI on Python 3.10, 3.11 and 3.12 even though no runner has an HDL tool. `requirements.txt`
and `pyproject.toml` are unchanged by this track, and `.gitignore` now ignores the build products
`asic_chip/sim/obj_dir`, `obj_dir_tb`, `obj_dir_vvp`, `*.vcd`, `*.vvp` and `asic_chip/openlane/runs`
while keeping `asic_chip/rtl/generated` versioned on purpose.

---

## 3. Mathematical Demonstrations and Concrete Proofs

This section provides clear, step-by-step mathematical proofs demonstrating that the parser computations and SMT verifier deliver exact, non-hallucinated results.

### Example 1: Manual Trace of Inside Soft Semiring on String `()`

Consider string $X = (\text{"("}, \text{")"})$ with length $N = 2$.
Vocabulary: $x_0 = \text{"("}$ (index 0), $x_1 = \text{")"}$ (index 1).
Non-terminals: $V_N = \{S, NT_1, NT_2\}$ ($K=3$).

Suppose the grammar parameters in log-space are configured as:
- Family gates (§1.1): $\pi_S = 1$ (only binary productions for $S$), $\pi_{NT_1} = \pi_{NT_2} = 0$ (only terminal emissions).
- Binary production: $\log P(S \to NT_1 \ NT_2) = \log(1.0) = 0$
- Terminal emissions:
  - $\log P(NT_1 \to \text{"("}) = \log(1.0) = 0$
  - $\log P(NT_2 \to \text{")"}) = \log(1.0) = 0$
All other productions have negligible probability ($\to -\infty$).

With the joint gate of §1.1 this parameterization satisfies the distribution invariant exactly:
$1.0 + 0 = 1.0$ for $S$, and $0 + 1.0 = 1.0$ for $NT_1$ and $NT_2$. Under the previous
independent-softmax encoding the same logits gave mass $2$ per non-terminal, and the likelihood
computed below would have been inflated by $2^{3} = 8$ (one factor per derivation node).

#### Step 1: Base Case Initialization ($L = 1$)
For each substring of length 1:
- Substring $x_0 = \text{"("}$ ($i=0, j=0$):
  $$I(0, 0, NT_1) = \log P(NT_1 \to x_0) = \log(1.0) = 0$$
  $$I(0, 0, S) = -\infty, \quad I(0, 0, NT_2) = -\infty$$
- Substring $x_1 = \text{")"}$ ($i=1, j=1$):
  $$I(1, 1, NT_2) = \log P(NT_2 \to x_1) = \log(1.0) = 0$$
  $$I(1, 1, S) = -\infty, \quad I(1, 1, NT_1) = -\infty$$

#### Step 2: Recursive Inside Step ($L = 2$)
We compute the log-probability for the full sequence ($i=0, j=1$, length 2) under start symbol $S$.
The only valid split point is $k = 0$.
The differentiable Inside equation is:
$$I(0, 1, S) = \mathrm{LogSumExp}_{B, C} \left( \log P(S \to B \ C) + I(0, 0, B) + I(1, 1, C) \right)$$

Evaluating for pair $(B, C) = (NT_1, NT_2)$:
$$\text{term} = \log P(S \to NT_1 \ NT_2) + I(0, 0, NT_1) + I(1, 1, NT_2) = 0 + 0 + 0 = 0$$

For any other pair $(B, C)$, at least one term is $-\infty$.
Therefore:
$$I(0, 1, S) = \log(e^0) = 0$$

The marginal sequence probability under the model is:
$$P(S \stackrel{*}{\Rightarrow} \text{"()"}) = \exp(I(0, 1, S)) = \exp(0) = 1.0$$
This demonstrates that the differentiable soft semiring reproduces exact symbolic likelihood without numerical deviation.

---

### Example 2: Grammar-Level Soundness Proof via Z3 (Inductive Counter Certificate)

We prove that **every** word generated by the induced grammar is a balanced bracket word, i.e.
$L(G) \subseteq \mathrm{Dyck}\text{-}2$. This is a property of the grammar, not of one string.

#### Counter abstraction
For a word $w$ over $V_T$, let $r(i)$ / $s(i)$ be the running round/square counters after reading the
first $i$ tokens, starting from $r(0) = s(0) = 0$. Define the **signature**
$$\sigma(w) = \big(\Delta_r,\ \Delta_s,\ \mu_r,\ \mu_s\big), \qquad \Delta_r = r(|w|),\ \Delta_s = s(|w|),\ \mu_r = \min_i r(i),\ \mu_s = \min_i s(i)$$
so that $w$ is well bracketed iff $\Delta_r = \Delta_s = 0$ and $\mu_r = \mu_s \ge 0$
(`is_well_bracketed`). Composition of counters is associative:
$$\Delta(uv) = \Delta(u) + \Delta(v), \qquad \mu(uv) = \min\big(\mu(u),\ \Delta(u) + \mu(v)\big)$$

#### Certificate and transfer constraints
`verify_language_soundness()` asks Z3 for one signature per non-terminal,
$(\delta_r, \delta_s, m_r, m_s)_A$, constrained to be a **valid over-approximation** of everything $A$
can generate:
- terminal production $A \to a$: $\delta = e(a)$ and $m = e(a)$, with $e(\texttt{(}) = (1,0)$,
  $e(\texttt{)}) = (-1,0)$, $e(\texttt{[}) = (0,1)$, $e(\texttt{]}) = (0,-1)$;
- binary production $A \to B\,C$: $\delta_A = \delta_B + \delta_C$ and
  $m_A = \min(m_B,\ \delta_B + m_C)$, encoded in Z3 as a conditional transfer so the solver selects the
  true prefix minimum:
  ```python
  solver.add(d_r == b_r + c_r, d_s == b_s + c_s)
  solver.add(m_r == z3.If(b_mr <= b_r + c_mr, b_mr, b_r + c_mr))   # A -> B C
  solver.add(m_s == z3.If(b_ms <= b_s + c_s,  b_ms, b_s + c_s))
  ```
  where $(b_r, b_s, b_{m_r}, b_{m_s})$ is the signature of $B$ and $(c_r, c_s, c_{m_r}, c_{m_s})$ that of $C$.

#### Theorem (soundness of the certificate)
If such a signature exists for every production of $G$ and the start symbol satisfies
$$\delta_S = (0, 0) \quad \wedge \quad m_S \ge (0, 0)$$
then $L(G) \subseteq \mathrm{Dyck}\text{-}2$.

*Proof.* By induction on derivation depth $d$: for $d = 1$ (production $A \to a$) the constraint sets
$\sigma(a) = (e(a), e(a))$, which is exact. For $A \Rightarrow a_1 \dots a_n$ through $A \to B\,C$ with
depths $< d$, the induction hypothesis gives $\sigma(\mathrm{yield}(B)) \ge_{\text{ovx}} (\delta_B, m_B)$
and likewise for $C$; by the composition law above, the yield of $A$ has
$\Delta = \delta_B + \delta_C = \delta_A$ and $\mu = \min(m_B, \delta_B + m_C) = m_A$. Hence
$\sigma(\mathrm{yield}(S)) = (\delta_S, m_S)$, and the hypothesis on $S$ gives $\Delta = 0$ and
$\mu \ge 0$, i.e. the yield is well bracketed. $\blacksquare$

The Z3 query answers `UNSAT` precisely because no derivation can witness $\neg G$; the report status is
`STATUS_PROVED` and the certificate is written to `artifacts/metrics.json`.

#### Certificate of the shipped grammar (seed 42 run)
| Non-terminal | $\delta_r$ | $\delta_s$ | $m_r$ | $m_s$ | reading |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `S` | 0 | 0 | 0 | 0 | balanced, never below the starting depth |
| `OPN_R` | 1 | 0 | 1 | 0 | opens a round pair |
| `CLS_R` | -1 | 0 | -1 | 0 | closes one round pair |
| `OPN_S` | 0 | 1 | 0 | 1 | opens a square pair |
| `CLS_S` | 0 | -1 | 0 | -1 | closes one square pair |

Every production of the 9-rule extracted grammar satisfies the transfer constraints, and
$\delta_S = (0,0)$, $m_S = (0,0) \ge (0,0)$: the grammar is provably sound at any derivation depth.

#### Refutation path (what happens when the grammar is wrong)
Before asking for a certificate, the verifier enumerates all words up to `--search-bound` tokens and
checks them with the discrete CYK. For the 4-non-terminal grammar shipped before this revision, which
forced one closing non-terminal to emit both `)` and `]`, the search returns the witness **`(]`**
together with its derivation tree, and the status is `STATUS_REFUTED`. The same case is locked by
`tests/test_verification.py::test_overgenerating_grammar_is_refuted`.

#### Completeness caveat
A signature per non-terminal is an *under-approximation*: some grammars generate only balanced words
but admit no such signature (the context in which a non-terminal occurs matters). The verifier then
answers `INCONCLUSIVE` rather than `UNSAT`. The failure mode is therefore conservative — it can never
produce a false "sound" verdict, which is exactly the bug the previous encoding had.

#### Note on the single-string diagnostic
`check_word_balance_z3(tokens)` encodes one word with integer states $R_0, \dots, R_n$, premises
$R_0 = 0$ and $R_{i+1} = R_i \pm 1$, and the goal **in the thesis**:
$$\big(R_n = 0\big) \wedge \big(\forall i,\ R_i \ge 0\big)$$
The old encoding placed $\forall i, R_i \ge 0$ among the **premises**; for `())`, `)(` and `][` those
premises are unsatisfiable on their own, so `check(P ∧ ¬G)` returned `UNSAT` trivially and the tool
reported "sound" for malformed words. The current version guards premise consistency
(`INCONSISTENT_PREMISES`) and returns `INVALID` for those inputs, as asserted by the regression tests.

---

## 4. Maintenance and Extension Procedures

To maintain high engineering and academic standards:

### 4.1 Extending to New Languages (e.g. Dyck-$k$, Arithmetic Expressions)
1. **Extend Vocabulary**: Update `VOCAB` and `VOCAB_MAP` in `DyckDataset`, or implement a new dataset generator in `trainer.py`.
2. **Dimension Non-Terminals**: Adjust `--num-nt`. For Dyck-$k$ in CNF the minimal skeleton needs $K = 1 + 2k$ non-terminals (one start symbol, $k$ openers, $k$ closers); Dyck-2 therefore needs $K = 5$, which is the default. A too-small $K$ cannot express the language (the previous 4-symbol "prior" could not, and generated `(]`).
3. **Mirror the Ground Truth**: Add the new CNF to `reference.py` (`GOLD_*` constants) so `--init prior`, the negative controls and `gold_rule_f1` stay meaningful; extend `BRACKET_EFFECT` / `counter_signature` in `verification.py` with the new symbol effects.
4. **Regenerate Artifacts**: Run `python3 src/neuro_symbolic/synthesize.py` to retrain and export the updated PCFG.
5. **Formal Verification**: Ensure gates `G1`–`G4` pass, i.e. `verify_lark_compilation()` + `verify_pcfg_distribution()`, 100% `verify_corpus_coverage()`, `verify_negative_controls()` and `verify_language_soundness()` returning `UNSAT`. (`verify_lark_syntax()` and `verify_smt_soundness()` no longer exist.)

### 4.2 Monitoring Training Dynamics
- `avg_entropy` is the mixture entropy $\sum_A H(A)$ of §1.3, bounded by
  $K\big(\log 2 + \max(2\log K,\ \log|V_T|)\big)$ — $19.6$ nats for $K = 5$, $|V_T| = 4$. It should decrease monotonically; if it plateaus above $\approx 8$ nats, raise `--entropy-penalty` ($0.05$ was too weak: it left the grammar dense, `rule_sparsity` $= 0$, and the extracted grammar still accepted non-Dyck words).
- `rule_sparsity` and `max_lhs_mass_error` are the health signals of the extracted distribution: target sparsity $\gtrsim 0.9$ and mass error $< 10^{-5}$.
- `grad_norm` is measured **after** `clip_grad_norm_`, so it is capped by `max_grad_norm = 1.0`; it is a diagnostic of how often clipping fires, not a stability criterion. Do not chase a threshold such as $< 10^{-4}$ on this quantity.
- Temperature annealing (`--tau-start` $\rightarrow$ `--tau`) matters more than the epoch count: a sharpened final $\tau$ with a weak entropy penalty leaves the gate diffuse, and extraction then prunes rules that the optimized distribution still needs.

### 4.3 Repository Hygiene
- Never commit `*.pyc` bytecode or `__pycache__/` folders.
- Store trained weights strictly under `artifacts/grammar_weights.pt`; keep `artifacts/metrics.json` next to the grammars so every exported artifact stays auditable. `artifacts/` is version-controlled and is overwritten by each `synthesize.py` run. `tests/test_grammar_artifacts.py` locks that state: it runs the synthesis into a temporary folder and refuses a checkout whose committed grammar JSON, YAML or GF module no longer equals what the current code exports, and it checks the four gates of the report rather than its measured digits, since two runs with the same seed differed there in the eighth significant digit. The folder is load-bearing beyond that track: gate 3 of the ASIC suite reads `artifacts/grammar.json` for the terminal set it compares with the alphabet of the generated automaton.

### 4.4 Known Limitations (measured, not asserted)
1. **Induction from random initialization is not yet solved.** With `--init random` (120 epochs, same optimizer
   and annealing schedule, `--seed 42`) the model fits the corpus but converges to an **over-generating** grammar:

   | Signal | Value |
   | :--- | :--- |
   | Extracted rules | 10 (3 binary + 7 terminal) over `S, NT_1..NT_4` |
   | Rule set | `S -> NT_4 NT_2`, `NT_4 -> S NT_4`, `NT_4 -> NT_1 S`, `NT_1 -> '(' \| '['`, `NT_2 -> ')' \| ']'`, `NT_3 -> '['`, `NT_4 -> '(' \| '['` |
   | `ppl_per_symbol` (test) | 3.864 |
   | `rule_sparsity` | 0.890 |
   | Corpus coverage | 100% |
   | Negative-control rejection | 98.5% (`(]`, `[)`, `(()]()` wrongly accepted) |
   | Gold rule $F_1$ | 0.421 ($P$ 0.400 / $R$ 0.444) |
   | Z3 verdict | `SAT`, witness `(]` derived by `S -> NT_4 NT_2` (gate `G4` **FAIL**) |

   The latent symbols collapse onto “openers/closers that mix bracket types” instead of the gold
   $O_R/C_R/O_S/C_S$ partition, which is exactly the failure mode the corrected soundness gate now catches
   (the same shape of error the shipped 4-symbol prior had). Unsupervised grammar induction on Dyck-2 within
   this budget is therefore an open problem; the passing default run relies on a disclosed structural prior.
2. **Perplexity target unmet.** Hold-out `ppl_per_symbol` is $2.86$ against the $\le 1.25$ goal inherited from the original specification; that target was reported as met previously only because the mass-$2$ bug deflated the likelihood.
3. **Certificate incompleteness.** The per-non-terminal signature proves soundness but is not complete (`INCONCLUSIVE` is possible); witness search is bounded by `--search-bound`.
4. **Model class.** A CNF PCFG with $K$ latent symbols cannot represent languages whose CNF needs more symbols, and the inside objective does not encourage disambiguation: several grammars fit the corpus, and only the extracted one is verified.
5. **No bit-exact reproducibility.** No global `torch` seed or deterministic algorithm setting is enforced inside the library; only `synthesize.py` seeds `random`/`torch`, so artifacts are reproducible per machine/build, not across PyTorch releases.
6. **The hardware track has layout evidence and no silicon evidence.** Verilator 5.054 ran in the container image `verilator/verilator:latest`, and linted and simulated the design: the RTL elaborates clean under `-Wall`, runs latch free, and reproduces all 304 golden vector rows plus the core self test with 0 mismatches in two independent paths. OpenLane 2.3.10 with the sky130A PDK completed two place and routes on 2026-11-09, all 78 steps each, which is why this tree now quotes post-layout numbers: three GDSII files per run, DRC at zero violations in Magic and in KLayout, LVS circuits match uniquely, area and utilization measured, static IR drop measured, and static timing reported over nine corners. What is still missing is a run that closed: both of them exited non zero on deferred setup and hold violations, the first also failed its antenna check on 54 pins and 46 nets while the second reported `Antenna Passed`, and both were constrained to a bring-up period, not to the declared one. There is no Icarus here and no second simulator has seen this RTL, and there is no Calibre and no signoff of that kind. The 4.0 ns period in `asic_chip/openlane/lingologos.sdc` is a declared budget and has never been routed; the layout of the first run misses setup by 4.4409 ns in the worst typical corner and by 15.7205 ns in the worst slow corner at an 8.0 ns period, and the layout of the second is 0.4906 ns short of a 30.0 ns period in that same slow corner while losing hold by up to 0.1540 ns on input paths. Power is a tool estimate under a default activity model, not a rating, and the datasheet labels every one of those figures with what it is.
7. **The hardware bracket tracker recognizes the commutative image of Dyck-2.** With one state byte, round and square brackets must share the same two multipliers, so the automaton accepts `(]` and `][`. This is the same over-generation the software soundness certificate was built to catch, and it is declared in `asic_chip/artifacts/gf_rules.json` and in the datasheet instead of being papered over.
8. **The core implements a documented subset, not a profile.** Every 64 bit operation, every system and privileged instruction, `FENCE.I`, any register field of 16 or above and any unaligned access are reported as illegal rather than performed; the 97 word firmware is an instruction-set self test of that subset, not an application.
9. **No electrical characterization.** Absolute maximum ratings, supply current ranges and junction temperature limits are not characterized, and the package entry in `asic_chip/config.json` is a declared body chosen to fit 146 signal bits plus the declared power pins, not a purchased part.

### 4.5 Coverage of the Test Suite
`tests/` (208 tests collected, 182 of them outside the two ASIC files of section 1.10) locks,
in particular: the distribution invariant for the parser and for the
extracted grammar; inside vs Viterbi agreement; post-pruning renormalization against
`probability_pre_pruning`; disjoint train/test split; the corrected semantics of
`check_word_balance_z3` (regression: `())`, `)(`, `][`, `((`, `[(]`, `(()[]` must be `INVALID`);
refutation of an over-generating grammar with the witness `(]`; the `gold_rule_f1` alignment on an
isomorphic copy of the gold CNF with renamed non-terminals; and the negative-control generator
(every control must be outside Dyck-2).

The pure-model track adds two suites. `tests/test_tokenizer.py` (16 tests) locks determinism of the
learned vocabulary, losslessness of the pre-tokenization, the no-merge rule on protected atoms, a
byte-exact round trip, zero unknown tokens including invisible whitespace and digit runs, one token
per algebra symbol, rejection of a symbol that only looks atomic (`SCHEDULEX`, `review`), the two
stop conditions of the merge loop, `encodable_symbols`, and `save` / `load` identity plus refusal of
a foreign format id. `tests/test_masking.py` (40 tests) locks the reader/writer inverse property,
refusal of fourteen malformed trees (missing and extra child, crossed brackets, leaf where an
operator name belongs, leaf shape too short or too long, unknown operator or leaf tag, payload
without a leaf, truncated stream, second root, nested leaf), the fact that the operator name alone
fixes every later closing bracket, the round-node-only root, the leaf shape profile, progress on
every proper prefix of a legal stream, seed reproducible sampling whose 200 draws are all legal
Dyck-2 trees, that the learned child-kind refinement strictly narrows the mask, and at token level
the gold protocol step by step, the closing marker available only at completeness, the forced single
space separator, `-inf` on every inadmissible logit with admissible scores untouched and zero
probability mass outside the allowed set, and hard errors on a narrower logits vector or a missing
alphabet symbol.

The release path adds `tests/test_hub_package.py` (11 tests): the safetensors round trip with the tied
projection restored and counted once, the report that refuses a file whose tensor count does not match
the distinct storages of the model, refusal of a header from another track, the algebra mask
specification round trip through JSON and its refusals, the projection of a gate report into the numbers
the model card prints, refusal of a report that does not read `PASS`, refusal of a checkpoint whose
vocabulary disagrees with the tokenizer it names, and the assembled folder: every declared artifact
present, every recorded sha256 verified against the file on disk, stray files from an earlier run left
out of the manifest.

The hardware track adds two suites and one helper. `tests/asic_reference_model.py` is not a test file:
it is an independent model of the chip, with its own dot product, its own `GF(2^8)` arithmetic over
`0x11D`, its own shunting-yard parser and mask evaluator for the clause DSL, and its own RV32E
emulator. It never imports the generator, which is what makes the cross-check non-circular.
`tests/test_asic_compiler.py` (17 tests) evaluates the generated wire netlist with a small interpreter
and compares it with that plain dot product; recomputes all 1020 automaton arms with field arithmetic
and checks the alphabet against the grammar artifact and the single default arm; derives each clause
lane word three ways, through the model masks, through the generator parser and through the generated
netlist text, for all 256 input bytes, and proves the network is latch free and that the depth quoted
in the configuration equals the depth of the generated wire dependency graph; recomputes every column
of all 304 golden vector rows without the generator; regenerates into a scratch directory and requires
byte-identical output plus a passing `--check`; recomputes the manifest digests over the raw bytes on
disk; checks the self test memory against the engines; checks that the literals baked into the
testbench and into the C++ harness equal the memory file and that the 400 cycle harness budget covers
the run, 89 cycles in the emulator and 91 as the harness itself measures, and that the row count the
testbench states equals the number of data lines in the vector file; checks that the harness touches
only declared top level ports and that the
debug group lives inside `LL_DEBUG_PORTS`; reconciles the port list, the bit counts, the datasheet pin
table and all 37 lines of the pin order plan; reconciles the SDC and the OpenLane configuration with
`asic_chip/config.json`; recomputes every quantitative claim of the datasheet including the switching
activity over the 303 transitions of the vector file and the three artifact digests it quotes; enforces
the ASCII, English and LF policy over the whole `asic_chip/` tree; and locks the `select ? set : clear`
semantics of `MUX` after a wrong mask identity in the reference model was caught by the cross-check.
`tests/test_asic_core_firmware.py` (6 tests) checks that the instruction ROM is complete and matches
the configuration, that every executed encoding is inside the documented subset, that the illegal
encoding is the documented `x16` probe, that the emulated run reproduces every entry of
`asic_chip/sim/core_self_test.mem`, that the run fits the harness cycle budget, and that data memory
readback matches the model.

---

## 13. Distributed Multi-Instance Orchestrator (RFC LingoLOGOS-MULTI-ORCH-v1)

### 13.1 Mathematical and Formal Formulation

The communication network is formalized as a directed graph:

$$G = (V, E)$$

where:
- $V = \{v_1, v_2, \dots, v_n\}$ is the finite set of autonomous node instances, each identified by an arbitrary identifier string (e.g., `agente_alfa`, `agente_beta`, `arbitro_logico`).
- $E \subseteq V \times V$ represents the directed routing relations declared in the adjacency matrix `routingMatrix`. A transmission from node $u$ to node $v$ is admissible if and only if:

$$(u, v) \in E$$

If a node $u$ attempts to route an envelope to a recipient $w \notin \{v \in V \mid (u, v) \in E\}$, the centralized message broker intercepts the payload and aborts with a `RoutingError`.

### 13.2 Volatile State and Temporal Invariants

Time in the multi-instance network is modeled as a strictly discrete, volatile tick counter:

$$\tau_k = k, \quad k \in \{1, 2, \dots, \text{maxTurns}\}$$

- **Zero Timestamp Persistence Invariant**: Envelopes and request payloads are forbidden from persisting fixed wall-clock timestamps (`created_at`, `timestamp`). All temporal context is propagated exclusively through volatile state variables:

$$\mathcal{S}_{\text{volatile}}^{(k)} = \langle \tau_k, \dots \rangle$$

- **Isolation Invariant**: Each node $v_i \in V$ runs as an isolated OS process via the CLI interface `lllayer`. No node implementation code is imported as an in-process module. Inter-process communication is performed strictly over standard I/O streams using JSON-encoded envelopes. The runner builds the environment of the node process explicitly and prepends the absolute folder that holds the `lingologos_orchestrator` package to `PYTHONPATH`: the node runs in its own working directory, where a relative entry inherited from the caller would not resolve, and a value the caller already had is kept behind that entry so an installed deployment is unaffected.

### 13.3 Formal SMT Verification with Z3

For each transmission turn $k$ from node $u$ to node $v$, the orchestrator gathers:
1. Node ontology declarations and axioms $\mathcal{O}_u$ from `datasets/{u}/ontology.smt2`.
2. SMT declarations $\mathcal{D}_k$ and assertions $\mathcal{A}_k$ from the message envelope `smtConstraints`.

The combined theory is evaluated under the Z3 SMT solver:

$$\mathcal{T}_k = \mathcal{O}_u \cup \mathcal{D}_k \cup \mathcal{A}_k$$

Before submission to Z3, symbol declarations in $\mathcal{D}_k$ are deduplicated against sort and function signatures in $\mathcal{O}_u$ to prevent re-declaration errors. The satisfiability verdict is computed:

$$\mathrm{Check}(\mathcal{T}_k) \in \{\mathrm{sat}, \mathrm{unsat}, \mathrm{unknown}\}$$

### 13.4 Termination Invariants

The orchestrator evaluates termination condition $\mathcal{C}$ at each turn $k$:

- **UNSAT Breaker**: If $\mathrm{Check}(\mathcal{T}_k) = \mathrm{unsat}$ and $\mathcal{C} \in \{\text{UNSAT\_BREAKER}, \text{Z3\_SAT\_CONVERGENCE\_OR\_UNSAT\_BREAKER}\}$, execution aborts immediately to prevent propagation of contradictory or safety-violating state transitions.
- **SAT Convergence**: If $\mathrm{Check}(\mathcal{T}_k) = \mathrm{sat}$ continuously across all active edges $|E|$, consensus is certified and the session terminates successfully with `SAT_CONVERGENCE`.
- **Turn Bound Exhaustion**: If $k = \text{maxTurns}$ is reached without convergence or breaker trip, the session halts with `MAX_TURNS_REACHED`.

### 13.5 Federated Infrastructure and Universal Transport Resolver

The orchestrator supports universal federation across heterogeneous execution environments. A
federated node adds four fields to the topology entry: `transport`, `remote_source`, `branch`,
`local_cache_path`, plus the `specs` folder holding `grammar.gf`, `ontology.smt2`, and
`seed_payload.json`.

1. **Local Transport (`local`)**: `local_cache_path` names the folder of the specification, resolved
   against `--workspace`, so the same configuration is valid from any working directory. Without the
   field the node falls back to `datasets/<node_id>/`.
2. **Git Transport (`git`)**: the cache folder, again anchored on the workspace and by convention
   `cache/nodes/<node_id>` (kept out of the index by `.gitignore`), is synchronized by Phase 01: a
   shallow clone `git clone --branch <branch> --depth 1 <remote_source> <cache>` on first use, an
   incremental `git -C <cache> pull origin <branch>` afterwards. The specification is then read from
   `<cache>/<specs.ontology>`. Because the synchronization runs inside `validate_static`, a node
   whose remote cannot be reached stops the session before a single turn is dispatched.
3. **HTTP Transport (`http`)**: the node is executed by `HttpNodeRunner`, which POSTs the input
   envelope to `endpoint_url` and parses the envelope of the response. When `auth_method` is
   `bearer_token` the request carries `Authorization: Bearer <token>`, and the token is read at
   Phase 02 from the environment variable named by `auth_token_env`; the token itself never appears
   in a topology file. A failed request raises `OrchestratorError` and aborts the session.

`NodeResolver.resolve` turns those fields into absolute paths: `grammar_path`, `ontology_path`,
`seed_path`, and `node_dir`, the folder of the ontology that Phase 02 passes to the node process as
`--node-dir`. The folder is configurable, the file names are not, because the node CLI contract of
section 13.3 requires the three specification files to sit in one directory. The seed payload of a
federated node must declare `sender` equal to the node id; a consolidated macro node is the one
exception, since the seed of its representative is relabelled to the macro id.

### 13.6 Infinite Node Scalability ($N \to \infty$)

To scale beyond static $N$-node configurations:
- **Event-Driven Message Bus (`EventBus`)**: the turn loop publishes the output envelope of every
  executed turn on the topic `channel/<sender>/<recipient>`, so a streaming subscriber observes the
  dialogue without being a routing endpoint. A subscriber on `*` receives every envelope, and
  `get_history()` returns exactly one envelope per executed turn.
- **Dynamic Node Spawner (`DynamicNodeSpawner`)**: `spawn_sub_node` emits the metadata of a
  sub-agent (generated node id, role, transport, parent id, task, and specification folder). The
  supervisor adds that entry to a federated configuration and starts a session over it; the
  orchestrator does not spawn a node in the middle of a running session.
- **Centralized SMT Solver Pool (`SMTSolverPool`)**: Phase 01 queues one `SolverTask` per node
  ontology and dispatches them through `verify_batch`, so the ontology precheck of $N$ nodes runs on
  the pooled workers instead of sequentially. Each worker binds its verifier to a dedicated
  `z3.Context()`: the implicit Z3 main context is shared process wide and is not safe to parse into
  from several threads at once, which the isolation avoids. `run_session` shuts the pool down in a
  `finally` block, so the worker threads never outlive the session.

### 13.7 Model Consolidation

The `ModelConsolidator` contracts complex $N$-node federated or multi-agent topologies. Every node is
mapped by the role classifier $\mu$ to a functional category, searching the role text for the
validator, opponent, and proponent keyword families and defaulting to proponent:
$\mu(v) \in \{\text{proponent}, \text{opponent}, \text{validator}\}$.

- **Triad Consolidation**: collapses $N$ nodes into the macro-nodes
  `macro_proponent`, `macro_opponent`, and `macro_validator`, with the projected edge set
  $E_M = \{(\mu(u), \mu(v)) : (u, v) \in E,\ \mu(u) \neq \mu(v))\}$. A category with no member is
  dropped, which degenerates the contraction to a dyad, and when the projection leaves no edge at all
  the surviving macro-nodes are wired into a cycle so the session still has a dialogue to run. Each
  macro node records a representative $r(m)$, the first physical node of its category, and runs on
  the specification folder of that node, so `--consolidate triad` needs no dataset of its own and
  stays end to end executable on any topology whose nodes have datasets.
- **Unified Consolidation**: collapses the entire topology into a single unified local execution
  macro-engine (`unified_macro_engine`). A single engine has no peer to exchange a turn with, so the
  projected routing matrix is empty and a session over it terminates immediately with
  `EMPTY_ROUTING_MATRIX`. Unified mode is the contraction used to inspect the compressed model, while
  triad mode is the consolidation that actually runs a dialogue.

---

## 14. Edge-NSAC Dynamic Board Topology (Single-Chip vs Multi-Chip)

The hardware infrastructure and routing architecture configure the computation topology dynamically, preserving and extending the neuro-symbolic engine (RTL synthesis, Z3 SMT solver, and GF grammar verification) to multi-dataset processing via partitioning, dynamic loading, and parallel mapping.

### 14.1 Operational Modes
- **Single-Chip / Single-Dataset (Standalone Mode)**:
  - External SerDes and high-speed transceivers are placed into low-power clock gating.
  - Internal loopback routes `tx_local` directly to `rx_local`.
  - The node executes SMT validation and GF syntax pipelines locally on the primary dataset.
- **Multi-Chip / Multi-Dataset (Distributed Mode)**:
  - An array of $N$ interconnected chips processes partitioned or parallel dataset shards.
  - Each chip acquires physical identity from hardware strapping pins `HARDWARE_NODE_ID[3:0]`.
  - Compute roles are allocated dynamically (e.g., Agente Alfa proponent, Agente Beta opponent, Arbitro Logico validator).

### 14.2 Hardware Router Module: `lingologos_topology_router.sv`
The switch of each die lives in `asic_chip/rtl/core/lingologos_topology_router.sv`: one
instance per die, a registered single hop switch, configured by four of the five fields of
`TOPOLOGY_CTRL` as those fields are decoded by the control and status layer.

```systemverilog
module lingologos_topology_router #(
    parameter int unsigned NODE_ID_WIDTH = 4,
    parameter int unsigned DATA_WIDTH    = 64,
    parameter int unsigned MAX_NODES     = 1 << NODE_ID_WIDTH
)(
    input  logic                                   clk,
    input  logic                                   rst_n,
    // Fields of TOPOLOGY_CTRL.
    input  logic                                   standalone,
    input  logic [NODE_ID_WIDTH-1:0]               node_id,
    input  logic [3:0]                             total_nodes,
    input  logic [MAX_NODES-1:0]                   routing_mask,
    // Injection and receive ports of the local core, valid and ready on both.
    input  logic                                   tx_valid,
    output logic                                   tx_ready,
    input  logic [NODE_ID_WIDTH-1:0]               tx_dest,
    input  logic [DATA_WIDTH-1:0]                  tx_data,
    output logic                                   rx_valid,
    output logic [DATA_WIDTH-1:0]                  rx_data,
    input  logic                                   rx_ready,
    // One channel per node id of the board, wired point to point by the board wrapper.
    output logic [MAX_NODES-1:0]                   link_tx_valid,
    input  logic [MAX_NODES-1:0]                   link_tx_ready,
    output logic [MAX_NODES-1:0][NODE_ID_WIDTH-1:0] link_tx_dest,
    output logic [MAX_NODES-1:0][DATA_WIDTH-1:0]    link_tx_data,
    input  logic [MAX_NODES-1:0]                   link_rx_valid,
    input  logic [MAX_NODES-1:0][NODE_ID_WIDTH-1:0] link_rx_dest,
    input  logic [MAX_NODES-1:0][DATA_WIDTH-1:0]    link_rx_data,
    output logic [MAX_NODES-1:0]                   link_rx_ready,
    output logic                                   routing_error
);
...
```

Four decisions are worth stating, because they are what the module does not do:

- **One hop, no relay.** `ROUTING_MASK` is a set of direct neighbours, not a path table, and
  16 bits cannot carry one. A frame whose destination is not named by the mask is refused by
  the injection port of the die that injected it, and the die reports it with one cycle pulse
  on `routing_error`. This is only sound when every destination a die can be asked to reach
  is wired to it directly, which is the case for the board `config/topology.json` declares;
  `tests/test_topology_router.py::test_every_link_the_config_declares_is_a_direct_channel_of_the_switch`
  reads the routing matrix and the negotiated masks and fails if a declared link ever stops
  being a direct channel, so a board that would need a relay cannot be introduced without
  turning that gate red.
- **A mask bit and a channel index are the same thing.** Channel $i$ of a die is the wire to
  node $i$, and the destination of a frame travels with the frame, so a link carries no
  protocol state and a board wrapper is only a wire between identically configured dies.
- **Loopback is a mode, not a mode of the mask.** With `STANDALONE` set, or with
  `TOTAL_NODES` at one, an injected frame goes to the local receive register and no link
  channel moves, whatever the destination field says.
- **`DATASET_ID` is deliberately absent from this module.** It selects the corpus shard the
  firmware of the die loads; it never reaches a datapath, and giving it a wire in the switch
  would suggest the fabric can act on a shard number.

The path from the injection port to `rx_valid` is two clock cycles when the way is free. The
fabric buffers four frames in front of a stalled destination (the receive register of the
destination die, the capture register of the channel, the send register of the source die,
the injection stage of the source die) and only the fifth frame is refused by the injection
port, which is back pressure rather than an imaginary always ready port.

What has been done to this module and what has not: it lints clean under `-Wall` and it is
simulated, in the container, through `make -f asic_chip/Makefile lint-topology` and
`make -f asic_chip/Makefile sim-topology`, whose bench drives four dies over the six wires of
the declared board and reports `FINAL VERDICT: PASS` over 42 assertions (section 2.4 of
`asic_chip/doc/verification_report.md`). It is not instantiated in `lingologos_top.sv` and it
appears in the `VERILOG_FILES` list of neither OpenLane config, so it has never been
synthesized, placed or routed: no timing, no area and no power number in this tree describes
it. `tests/test_topology_router.py::test_the_switch_is_linted_and_simulated_but_not_synthesized`
locks that boundary in both directions.


### 14.3 CSR Control Register: `TOPOLOGY_CTRL`
The 32-bit `TOPOLOGY_CTRL` register controls the router and dataset routing configuration:

| Bit Range | Field Name | Access | Description |
| :--- | :--- | :--- | :--- |
| **0** | `STANDALONE` | R/W | **1:** Internal loopback; **0:** External inter-chip mesh links. |
| **4:1** | `NODE_ID` | R | Physical node identifier read from strapping pins. |
| **8:5** | `TOTAL_NODES` | R/W | Total chips detected on board (1 for standalone, N for cluster). |
| **15:9** | `DATASET_ID` | R/W | Logical dataset shard index assigned to this chip. |
| **31:16** | `ROUTING_MASK` | R/W | Forwarding mask for adjacent nodes. |

The register exists today as a definition and a packer in `src/lingologos_orchestrator/topology_ctrl.py`,
which the orchestrator packs and hands to each `lllayer` process through
`--topology-ctrl-reg`, and as four decoded inputs of the switch described above. There is no
control and status block in `asic_chip/rtl/top/lingologos_top.sv` that holds these bits and no
strapping pin that carries `NODE_ID`, so the field layout is a contract between the software
and the switch, not a memory mapped register of the chip that was placed and routed.
`tests/test_topology_router.py::test_switch_is_sized_by_the_register_it_is_wired_to` compares
the widths of the two sides, which is where a silent disagreement would start.

### 14.4 Software Auto-Discovery and Handshake Flow (`lllayer`)
When `lllayer` initialises on a node:
1. **Strapping Read**: Obtains `NODE_ID` from hardware pins.
2. **Board Network Probing**: Sends probing packets over external links.
   - **Timeout (0 peers)**: Fallback to Standalone Mode (`TOPOLOGY_CTRL.STANDALONE = 1`, `DATASET_ID = 0`, local autonomous validation).
   - **Responses from $N-1$ peers**: Multi-Chip Mode (`TOPOLOGY_CTRL.STANDALONE = 0`, `TOTAL_NODES = N`, partition dataset shards across `DATASET_ID`, assign distributed roles).

### 14.5 Developer Environment Automation: PowerShell & WSL Bridge
The script `Activate.ps1` (`.venv-train/Scripts/Activate.ps1` and `scripts/Activate.ps1`) establishes an idempotent bridge from Windows PowerShell into the WSL Ubuntu environment:
1. Converts the Windows project path to WSL format via `wslpath -u`.
2. Checks for and idempotently creates `.venv-wsl` if missing.
3. Activates `.venv-wsl` and installs Python dependencies from `requirements.txt`.
4. Launches an interactive bash login shell ready for `lllayer` and Verilator execution.

---

## 15. In-Memory Constrained Decoding and AST Pruning Engine (`lllayer`)

### 15.1 Mathematical Formulation of Constrained Autoregressive Decoding

Let $\mathcal{V}$ be the vocabulary with cardinality $|\mathcal{V}| = N$. At each autoregressive step $t$, the runner model computes scoring function $\mathbf{z}_t = f_\theta(y_{<t}) \in \mathbb{R}^N$.

Given abstract typed formal grammar $\mathcal{G} = (\Sigma, N, S, P, \mathcal{T})$, the incremental runtime maintains state $s_t \in \mathcal{S}$ with deterministic transition function:

$$\delta: \mathcal{S} \times \mathcal{V} \to \mathcal{S} \cup \{\bot\}$$

where $\bot$ indicates an invalid transition that violates formal grammar $\mathcal{G}$ or logical constraints $C_{\text{SMT}}$ evaluated on Z3.

### 15.2 Zero-Copy In-Memory Vector Masking ($\mathcal{O}(1)$ Allocation)

The set of admitted tokens at state $s_t$ is:

$$\mathcal{A}(s_t) = \{ v \in \mathcal{V} \mid \delta(s_t, v) \neq \bot \}$$

Volatile masking vector $\mathbf{m}_t \in \mathbb{R}^N$ allocated in RAM:

$$m_{t,i} = \begin{cases} 0.0 & \text{if } i \in \mathcal{A}(s_t) \\ -\infty & \text{if } i \notin \mathcal{A}(s_t) \end{cases}$$

Logits transformation prior to Softmax is computed in-place via Hadamard addition:

$$\mathbf{\hat{z}}_t = \mathbf{z}_t + \mathbf{m}_t$$

$$\lim_{m_{t,i} \to -\infty} P(y_t = i \mid y_{<t}, s_t) = \frac{\exp(z_{t,i} + m_{t,i})}{\sum_{j=1}^N \exp(z_{t,j} + m_{t,j})} = 0, \quad \forall i \notin \mathcal{A}(s_t)$$

### 15.3 Deterministic AST Tree Pruning Operator

Given an AST $T_A = (V_A, E_A)$, deterministic synthesis $T_B = \pi(T_A)$ applies projection operator $\pi$:

$$\pi(v) = \begin{cases} v & \text{if } \mathrm{Type}(v) \in \mathcal{T}_{\text{essential}} \\ \emptyset & \text{if } \mathrm{Type}(v) \in \mathcal{T}_{\text{ancillary}} \end{cases}$$

When encountering a node whose semantic salience is not essential (such as `OPTIONAL_MODIFIER`), the engine transitions into terminal closing state $\mathcal{S}_{\text{terminal\_closing}}$, masking all tokens except closing parenthesization tokens.

### 15.4 C-FFI Native Accelerator and Pure Vector Fallback

- **C-FFI Kernel (`src/lingologos_orchestrator/native/masker.cpp`)**: Exports `masker_init`, `masker_update_allowed`, `masker_apply_in_place`, and `masker_free`. Reuses contiguous RAM float buffer across steps with zero allocations inside the loop.
- **Python Bridge (`src/lingologos_orchestrator/lllayer_masker.py`)**: `NativeASTLogitsProcessor` manages C pointers, zero-copy contiguous numpy array pass-through, and graceful fallback to numpy in memory.
- **Incremental Engine (`src/lingologos_orchestrator/lllayer_engine.py`)**: `IncrementalASTEngine` drives step-by-step decoding, AST stack tracking, and subtree pruning triggers.
- **CLI Flag (`lllayer`)**: `--constrained-decoding` activates the in-memory AST-guided logits masker for node execution.

### 15.5 Host Support for the Masking Kernel

The masking layer runs on Linux and on Windows, and the selection between the two kernels never
goes through a shell:

- `library_filename(os_name, sys_platform)` maps the host to the shared object name of the kernel:
  `masker.dll` on Windows, `libmasker.dylib` on macOS, `libmasker.so` on every other POSIX system.
  `native_library_path()` resolves that name inside `src/lingologos_orchestrator/native/`.
- `compiler_candidates()` collects GCC compatible compilers with `shutil.which`, preferring `c++`,
  `g++`, and `clang++`, and additionally the `x86_64-w64-mingw32-g++` and `aarch64-w64-mingw32-g++`
  triplets on Windows. No shell is spawned and no external probe utility is invoked, so the lookup is
  valid in PowerShell and in CMD exactly as in bash.
- `build_command()` produces the single GCC flavored invocation
  `-O3 -shared -fPIC -std=c++17 <source> -o <target>`, and the compiler output is captured to
  `DEVNULL` instead of being inherited, because the orchestrator parses node stdout as JSON and the
  node stderr must stay empty.
- `find_or_build_native_library()` returns the located or freshly compiled kernel, or `None`. Returning
  `None` is a supported outcome rather than an error: `NativeASTLogitsProcessor` then keeps the
  persistent numpy buffer and applies the identical mask.
- `LINGOLOGOS_DISABLE_NATIVE_KERNEL=1` forces the numpy branch on any host. Deployments use it to
  suppress the on-the-fly compilation, and the parity tests use it to pin the two branches.
- MSVC `cl` is not a candidate, because the invocation above is GCC flavored. Windows acceleration
  therefore needs MinGW-w64 `g++` or LLVM `clang++`; without one the numpy branch runs and the decoded
  output does not change.

| Host | Kernel actually used | Evidence measured on this checkout |
| --- | --- | --- |
| WSL Ubuntu, `c++` and `g++` present | compiled `libmasker.so` | `is_native` is `True`; over `(2.0, 1.0, 5.0, 3.0, 4.0)` with admitted set `{1, 3}` the mask yields `[-inf, 1.0, -inf, 3.0, -inf]`; the masker test file runs `12 passed` |
| Windows PowerShell, no C++ compiler installed | numpy buffer | `is_native` is `False`; the same mask vector is produced; masker file `11 passed, 1 skipped`; whole suite `243 passed, 1 skipped`, zero failures |
| Windows with MinGW-w64 or LLVM clang | `masker.dll` | Expected by symmetry with the Linux branch. Not exercised on this machine: no C++ compiler is installed on the Windows side, and the WSL distribution cannot take the sudo password needed to install a cross toolchain |

Both branches satisfy the same equivalence gate of section 15.2: `test_native_kernel_matches_numpy_reference`
compares the C kernel against the numpy buffer over six admission sets on a 64 wide vocabulary and
requires `np.array_equal`, and `test_reference_decoding_is_identical_on_both_kernels` requires the
decoded AST to be identical whichever branch decoded it. Generated binaries stay untracked, because
`.gitignore` lists `*.so`, `*.dylib`, and `*.dll` in agreement with the zero binary rule of `AGENTS.md`.

### 15.6 Cross-Platform Test Contract

`tests/test_lllayer_masker.py` must be green on both hosts, and its assertions are written so that
they are:

- `test_library_filename_matches_every_supported_host` pins the three names and checks that the host
  executing the test receives exactly the file name its loader can open.
- `test_compiler_lookup_stays_free_of_posix_shell_probes` is the regression gate for the previous
  implementation, which shelled out to `which(1)` with a redirection onto the POSIX null device. That
  probe always failed on Windows and wrote two error lines onto stderr, which a node protocol must
  never emit. The gate rejects those constructs in the module source and asserts that every resolved
  candidate is a real file.
- `test_lllayer_cli_constrained_decoding_flag` mirrors `NodeProcessRunner._child_environment`: it
  prepends the absolute folder holding the package to `PYTHONPATH` for the child interpreter and pins
  `cwd` to the checkout root, so the relative `datasets/agente_alfa` path resolves from any starting
  directory. It also asserts that the child stderr is empty. The earlier version relied on the pytest
  `pythonpath = ["src", "."]` ini option, which pytest injects into its own process only, so the child
  aborted with `ModuleNotFoundError` on every platform.
- `test_environment_switch_selects_the_numpy_branch` validates the documented switch on any host.
- The only conditional test is the native versus numpy comparison, which reports an explicit skip
  reason naming the compilers it searched for. Everything else runs identically on both platforms.

Recorded status: Windows host `243 passed, 1 skipped, 0 failed`. Linux, checked inside WSL Ubuntu with
a temporary environment holding `numpy`, `pytest`, `z3-solver`, and `httpx`: masker test file
`12 passed`, including the native comparison that skips on Windows. The `.venv-wsl` environment stored
in the checkout still holds no installed dependencies, so a Linux session run needs the setup command
of section 14 that installs `requirements.txt`.

### 15.7 Integration Boundary of the Introduced Engine

The current engine is a reference harness rather than the production decoding path, and the boundary
matters when the flag is evaluated against topologies and datasets.

- **Fixed reference grammar**: `run_constrained_ast_decoding(node_id, role, incoming_ast, vocab_size)`
  builds a hard-coded eight token micro-grammar (`ROOT`, `(`, `Transition`, `StateA`, `StateB`, `)`,
  `END`, `MODIFIER`). The incoming AST reaches `run_session` as `prompt` and is never consumed, and
  the node folder (`grammar.gf`, `ontology.smt2`, `seed_payload.json`) is never read. Measured on the
  three shipped datasets, the decoded AST is the same string
  `( Transition StateA StateB ) (Node <node_id>)` for `agente_alfa`, `agente_beta`, and
  `arbitro_logico`, regardless of the AST carried by the envelope.
- **Pruning stays dormant through the CLI**: `MockRunnerModel.get_logits` returns a constant vector
  and `sample` takes the greedy argmax, so at `NODE_TRANS` the lower indexed essential token wins over
  `MODIFIER` and the ancillary projection $\pi$ of section 15.3 is never triggered. Only the dedicated
  unit test, which restricts `NODE_TRANS` to the modifier token, exercises that branch.
- **Dataset turn payloads bypass the masker**: a folder holding `turn_<n>_payload.json`
  short-circuits `generate_node_response` before the masking branch, so recorded payloads take
  precedence over constrained decoding. No such file ships under `datasets/` today.
- **Topology handling is unaffected**: the flag changes only the payload content. `topologyCtrl` and
  `datasetId` are still attached after generation, verified with
  `--hardware-node-id 3 --probe-peers 2 --dataset-id 1`, which yields
  `standalone=0`, `node_id=3`, `total_nodes=3`, `dataset_id=1`, `raw_hex=0x00000266`.
- **Multi-node sessions cannot activate it**: `NodeProcessRunner.execute_turn` composes the fixed
  flag list `--node-id`, `--role`, `--node-dir`, `--hardware-node-id`, `--probe-peers`,
  `--dataset-id`, plus optional `--topology-ctrl-reg`, with no pass-through, and the orchestrator CLI
  exposes no equivalent switch. A session over `config/topology.json` completes with
  `Termination reason: SAT_CONVERGENCE`, `Verdict: PASS`, and no turn of the transcript carries the
  constrained decoding payload, so `--constrained-decoding` is currently an isolated single-node
  capability.
- **Two masking stacks are not unified yet**: grammar driven masking for the learned model lives in
  `src/neuro_symbolic/masking.py` through `AlgebraMask` and `TokenStreamMask`, which walk the bracket
  algebra over the tokenizer ids. The `lllayer` engine consumes neither `artifacts/grammar.json` nor
  the tokenizer, so the `lllayer_masker` kernel is a separate, smaller mechanism that a future change
  is expected to fold into that automaton.
