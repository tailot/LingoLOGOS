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
  - `pytest >= 8.0.0`

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
- Store trained weights strictly under `artifacts/grammar_weights.pt`; keep `artifacts/metrics.json` next to the grammars so every exported artifact stays auditable. `artifacts/` is version-controlled and is overwritten by each `synthesize.py` run.

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

### 4.5 Coverage of the Test Suite
`tests/` (43 tests) locks, in particular: the distribution invariant for the parser and for the
extracted grammar; inside vs Viterbi agreement; post-pruning renormalization against
`probability_pre_pruning`; disjoint train/test split; the corrected semantics of
`check_word_balance_z3` (regression: `())`, `)(`, `][`, `((`, `[(]`, `(()[]` must be `INVALID`);
refutation of an over-generating grammar with the witness `(]`; the `gold_rule_f1` alignment on an
isomorphic copy of the gold CNF with renamed non-terminals; and the negative-control generator
(every control must be outside Dyck-2).
