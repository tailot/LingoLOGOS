# Neuro-Symbolic Grammar Induction and Rule Synthesis Engine

Integrated neuro-symbolic architecture designed for differentiable grammar induction and rule synthesis from unstructured data streams. The engine combines continuous representations learned via neural networks with the formal constraints of Probabilistic Context-Free Grammars (**PCFG**) in Chomsky Normal Form (**CNF**).

---

## 1. System Architecture

The system integrates continuous representation learning via gradient descent with rigorous symbolic formal parsing:

```
   +-------------------------------------------------------------------+
   |                    Input Sequence / Data Stream                   |
   +-------------------------------------------------------------------+
                                     |
                                     v
   +-------------------------------------------------------------------+
   |             Neural Encoder (Continuous Vector Embeddings)         |
   +-------------------------------------------------------------------+
                                     |
                                     v
   +-------------------------------------------------------------------+
   |      Differentiable CYK Parsing Engine (Inside, log-semiring)     |
   +-------------------------------------------------------------------+
                    /                                 \
                   v                                   v
   +-------------------------------+   +-------------------------------+
   | Loss Optimization (Entropy &  |   | Rule Extraction & Induction   |
   | Dynamic Programming Gradients)|   | (Discrete Grammar Conversion) |
   +-------------------------------+   +-------------------------------+
```

### Main Components
1. **Neural Encoder Module**: Maps input terminal symbols into a continuous vector space $d_{model}$.
2. **Differentiable CYK Engine**: Implements the Cocke-Younger-Kasami algorithm in differentiable form using soft semiring operations (`LogSumExp`), enabling backpropagation of gradients through the parsing chart.
3. **Symbolic Extraction Layer**: Discretizes the weighted rule transition matrix via probability pruning (threshold $\tau = 0.01$) to produce a rigid formal grammar in Chomsky Normal Form (CNF).
4. **Formal Verification Loop**: Checks the PCFG distribution invariant, compiles the extracted grammar as a Lark (Earley) grammar, proves 100% corpus yield coverage with a discrete CYK, measures rejection of **negative controls**, and proves or refutes the language inclusion $L(G) \subseteq \mathrm{Dyck}\text{-}2$ with the **Z3 SMT solver**.

---

## 2. Mathematical Formulation and Theoretical Model

Grammar induction is formulated as maximizing the marginal log-likelihood of observed sequences $X = (x_1, x_2, \dots, x_N)$ with respect to grammar parameters $\theta$.

### 2.1 Formal Definition of Weighted Grammar
The PCFG is defined by the 5-tuple $G = (V_N, V_T, R, S, P)$, where:
- $V_N$: Finite set of non-terminal symbols ($|V_N| = K$).
- $V_T$: Finite set of terminal symbols.
- $R$: Set of production rules of the form $A \to B \ C$ or $A \to a$, with $A, B, C \in V_N$ and $a \in V_T$.
- $S \in V_N$: Start symbol.
- $P$: Probability distribution over production rules.

### 2.2 Differentiable Inside Tensor Equations
Only the **inside** chart is needed: gradients of the marginal log-likelihood flow directly through the
`LogSumExp` chart, so no outside pass is implemented (the "Inside-Outside" label of earlier revisions
was inaccurate).

Let $I(i, j, A)$ be the "Inside" log-probability that substring $x_i \dots x_j$ is derived from non-terminal $A$. Every $\log P(\cdot)$ below is a **jointly normalized** production log-probability as defined in §2.3 (gate $\pi_A$ included).

For substring lengths $L = j - i + 1$:

- **Base Case** ($L = 1$, terminals):
  $$I(i, i, A) = \log P(A \to x_i)$$

- **Recursive Step** ($L > 1$, non-terminals):
  $$I(i, j, A) = \mathrm{LogSumExp}_{k=i}^{j-1} \left( \mathrm{LogSumExp}_{B, C \in V_N} \left( \log P(A \to B C) + I(i, k, B) + I(k+1, j, C) \right) \right)$$

### 2.3 Joint Normalization of Productions (Distribution Invariant)
Every non-terminal $A$ competes over two production families, binary ($A \to B\,C$) and terminal ($A \to a$). The model allocates a **Bernoulli family gate** $\pi_A$ per non-terminal:

$$\pi_A = \sigma(\phi_A), \qquad
P(A \to B\,C) = \pi_A \cdot \mathrm{softmax}_{\theta}\big(r_{A}\big)_{B,C}, \qquad
P(A \to a) = (1-\pi_A) \cdot \mathrm{softmax}_{\theta}\big(t_{A}\big)_{a}$$

which enforces the **distribution invariant**

$$\sum_{B,C \in V_N} P(A \to B\,C) \; + \; \sum_{a \in V_T} P(A \to a) \; = \; 1 \qquad \forall A \in V_N$$

> Historical note: earlier revisions normalized the two families independently, giving every
> left-hand side total mass $2$. Marginal string probabilities were therefore inflated and the
> reported perplexity was an artifact. `DifferentiableCYKParser.production_mass_per_lhs()` and
> the `G1` acceptance gate now check this invariant numerically (error $< 10^{-6}$).

### 2.4 Loss Function and Neural Regularization
The global objective $\mathcal{L}(\theta)$ combines the negative log-likelihood with an entropy regularizer $H(A)$ defined on the **mixture** of the two production families, so that sharpening the gate is penalized too:

$$\mathcal{L}(\theta) = -\frac{1}{|D|} \sum_{X \in D} \log P(S \stackrel{*}{\Rightarrow} X \mid \theta) + \lambda_1 \sum_{A \in V_N} H(A) + \lambda_2 \|\theta\|_2^2$$

$$H(A) = H_b(\pi_A) + \pi_A \, H\big(P(A \to \cdot\,\cdot)\big) + (1-\pi_A)\, H\big(P(A \to \cdot)\big)$$

where $H_b$ is the binary entropy. Temperature annealing $\tau : \tau_{\text{start}} \rightarrow \tau_{\text{end}}$ is applied during training so that the discrete pruning read-out ($\tau_{\text{prune}} = 0.01$) is consistent with the distribution actually optimized.

---

## 3. Synthetic Data and Validation Metrics

Evaluation compares the induced grammar with the ground-truth grammar of the controlled formal language **Dyck-2** (balanced parentheses and brackets), on a **phrase-level train/test split** (100 unique sentences: 75 train / 25 test), plus 200 automatically generated **negative controls** (perturbations that provably fall outside Dyck-2).

| Metric | Definition | Measured (seed 42) | Acceptance Target |
| :--- | :--- | :--- | :--- |
| **Corpus coverage (CYK)** | $\frac{1}{|D|}\sum_s \mathbb{1}\!\left[\exists\, T: S \Rightarrow^\ast T,\ \mathrm{Yield}(T)=s\right]$ | **100.0%** | $100\%$ |
| **Negative-control rejection** | Fraction of out-of-language strings **not** derivable in the extracted grammar | **100.0%** (200/200) | $\ge 99.9\%$ |
| **Gold rule F1** | $F_1$ between extracted and gold rules, best non-terminal alignment | **1.000** ($P=R=1.000$) | $\ge 0.900$ |
| **Perplexity per symbol (hold-out)** | $\exp\!\left(-\frac{1}{N}\sum_{X\in D}\log P(S \Rightarrow^\ast X)\right)$ (marginal) | **2.86** | $\le 1.25$ — *not met, see notes* |
| **Rule sparsity** | $\frac{|\{r \in R_{\max} \mid P(r) < 10^{-4}\}|}{|R_{\max}|}$ | **0.917** (133 of 145 productions) | High determinism |
| **Distribution invariant error** | $\max_A \left| \sum_\alpha P(A \to \alpha) - 1 \right|$ | **1.2e-07** | $< 10^{-5}$ |
| **SMT certificate** | Verdict of $L(G) \subseteq \mathrm{Dyck}\text{-}2$ | **UNSAT** (proved) | UNSAT |

### Honest notes on these metrics
- **Marginal, not conditional perplexity.** A chart parser exposes $P(S \Rightarrow^\ast X)$, not $P(x_i \mid x_{<i})$. The reported PPL is therefore a *marginal per-symbol* figure; it is not comparable with autoregressive language-model perplexity, and the $\le 1.25$ target of the original spec has not been reached (measured 2.86 on the hold-out with the current 60-epoch budget).
- **The former "Parsing F1 = 1.000" was vacuous** and has been removed: it merely counted sentences whose inside probability was non-zero, which a dense PCFG achieves for *any* bracket string (including `]`, `(]` and `)(`). Selectivity is now measured by negative controls and gold-rule alignment.
- **Prior initialization is disclosed.** The default run seeds the logits with the gold Dyck-2 CNF (`--init prior`). The random-initialization ablation (`--init random`) is reported in `doc/TECHNICAL_DOCUMENTATION.md`: within the tested budget it does **not** recover the gold grammar (gold rule F1 $\approx 0.42$, extracted grammar over-generates `(]`).

---

## 4. Description of Generated Grammar Artifacts

The synthesis process outputs discrete formal artifacts in JSON, YAML, and Grammatical Framework (`.gf`) formats, plus a machine-readable verification report (`artifacts/metrics.json`). The structure extracted after pruning ($\tau_{\text{prune}} = 0.01$) with per-non-terminal renormalization is stored in `artifacts/`:

### YAML Grammar (`artifacts/grammar.yaml`, seed 42 run)
```yaml
grammar_metadata:
  type: Probabilistic Context-Free Grammar (PCFG)
  normal_form: Chomsky Normal Form (CNF)
  non_terminals: [S, OPN_R, CLS_R, OPN_S, CLS_S]
  terminals: ['(', ')', '[', ']']
  start_symbol: S

production_rules:
- rule: S -> S S            # probability: 0.1606   (pre-pruning 0.1605)
- rule: S -> OPN_R CLS_R    # probability: 0.4022   (pre-pruning 0.4019)
- rule: S -> OPN_S CLS_S    # probability: 0.4373   (pre-pruning 0.4370)
- rule: CLS_R -> S CLS_R    # probability: 0.2934
- rule: CLS_S -> S CLS_S    # probability: 0.3224
- rule: OPN_R -> '('        # probability: 1.0000   (pre-pruning 0.9990)
- rule: CLS_R -> ')'        # probability: 0.7066
- rule: OPN_S -> '['        # probability: 1.0000   (pre-pruning 0.9990)
- rule: CLS_S -> ']'        # probability: 0.6776
```

This is exactly the ground-truth Dyck-2 CNF skeleton
($S \to S\,S \mid O_R C_R \mid O_S C_S$, $C_R \to S\,C_R \mid \texttt{)}$, $C_S \to S\,C_S \mid \texttt{]}$, $O_R \to \texttt{(}$, $O_S \to \texttt{[}$):
9 rules out of the 145 possible productions, every left-hand side summing to $1$.
`probability_pre_pruning` records the raw model mass before the $\tau$ cut, so the renormalization applied after pruning remains auditable.

---

## 5. Training Tuning (Hyperparameters)

Defaults of `src/neuro_symbolic/synthesize.py` (all overridable from the CLI, see §10):

| Hyperparameter | Default | Search Range | Description |
| :--- | :--- | :--- | :--- |
| `--seed` | $42$ | — | Seeds dataset generation, split, initialization and negative controls. |
| `--init` | `prior` | `prior`, `random` | `prior` seeds the gold Dyck-2 CNF structure; `random` starts from $\mathcal{N}(0, 0.5)$. |
| `--num-nt` ($K$) | $5$ | $[4, 8]$ | Number of non-terminals (Dyck-2 CNF needs 5). |
| `learning_rate` | $1.0 \times 10^{-2}$ | $[10^{-4}, 3 \times 10^{-2}]$ | AdamW learning rate. |
| `tau_start -> tau` | $1.0 \rightarrow 0.5$ | $[0.1, 2.0]$ | Linear temperature annealing; the final value also drives extraction. |
| `entropy_penalty` ($\lambda_1$) | $0.5$ | $[0.01, 2.0]$ | Mixture-entropy regularizer; values $\le 0.05$ leave the grammar dense (see §3). |
| `weight_decay` ($\lambda_2$) | $1.0 \times 10^{-4}$ | $[0, 10^{-2}]$ | L2 regularization for weight stability. |
| `max_grad_norm` | $1.0$ | $[0.5, 5.0]$ | Gradient clipping (torch default in `GrammarTrainer`); without it the sharpening phase diverges. |
| `--epochs` / `--batch-size` | $60$ / $8$ | $[15, 200]$ / $[4, 32]$ | Batches are grouped by sentence length for homogeneous CYK charts. |
| `--prune` ($\tau_{\text{prune}}$) | $0.01$ | $[10^{-3}, 10^{-1}]$ | Production pruning threshold, followed by per-LHS renormalization. |
| `--search-bound` | $6$ | $[4, 10]$ | Exhaustive counterexample search length for the soundness check. |
| `max_seq_length` | $12$ (data), $64$ (engine) | $[16, 256]$ | Maximum sequence length processed in a single CYK pass. |

> Reproducibility note: no bit-exact artifact reproducibility is guaranteed across PyTorch
> versions or devices (no `torch.use_deterministic_algorithms`, cuDNN kernels are non-deterministic).
> The seed makes a run repeatable on the same machine and build; the acceptance gates in §9, not
> byte-identical artifacts, are what the CI enforces.

---

## 6. Differentiable CYK Core

The computational core lives in `src/neuro_symbolic/parser.py` (not duplicated here, so the
documentation cannot drift away from the code). Its contract:

- `get_log_rules()` / `get_log_terms()` return log-productions **including the family gate**,
  i.e. $\log \pi_A + \log \mathrm{softmax}(r_A)$ and $\log(1-\pi_A) + \log \mathrm{softmax}(t_A)$.
- `forward(x, temperature)` runs the inside CYK in the log semiring over span chart
  `chart[b, length-1, i, A]`, returning $\log P(S \Rightarrow^\ast X)$ per sentence.
- `production_mass_per_lhs()` must return $1$ for every non-terminal (invariant of §2.3).
- `parse_viterbi(tokens)` returns the most probable derivation tree (backpointer CYK), used as
  constructive evidence by the verifier.
- `rule_entropy()` is the mixture entropy $H(A)$ of §2.4.

---

## 7. Repository Layout and Generated Artifacts

```
.
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow (Python 3.10, 3.11, 3.12)
├── artifacts/
│   ├── DynamicGrammarIta.gf       # Grammatical Framework grammar (abstract + concrete)
│   ├── grammar.json               # Discrete CNF productions in JSON format
│   ├── grammar.yaml               # Discrete CNF productions in YAML format
│   ├── grammar_weights.pt         # Saved PyTorch checkpoint (rule / term / family logits)
│   └── metrics.json               # Metrics, verification report and acceptance gates
├── asic_chip/
│   ├── artifacts/                   # Weight mesh, GF rules and SMT clauses: the only source of the RTL
│   ├── config.json                  # Public ports, datapath limits, package and path declarations
│   ├── doc/                         # Datasheet, integration guide, verification report, build instructions
│   ├── env.sh                       # POSIX environment for the container flow
│   ├── Makefile                     # gen check sim-verilator sim-tb sim-iverilog lint pytest openlane openlane-spec setup-openlane clean
│   ├── openlane/                    # SkyWater sky130A configs, 4.0 ns spec and 32.0 ns bring-up, pin plan
│   ├── rtl/                         # Generated mesh, GF automaton and clause netlist, RV32E core, top level
│   ├── scripts/                     # lingologos_compiler.py, the OpenLane entry point, and the one command automatic setup (sh plus ps1)
│   └── sim/                         # C++ harness, SystemVerilog testbench, golden vectors, core self test
├── doc/
│   └── TECHNICAL_DOCUMENTATION.md # Comprehensive engineering and mathematical docs
├── src/
│   └── neuro_symbolic/
│       ├── __init__.py            # Package public interface
│       ├── parser.py              # DifferentiableCYKParser, family gate, Viterbi AST extraction
│       ├── grammar.py             # PCFGrammar, Symbol, ProductionRule, CYK derive, JSON/YAML/GF/Lark exporters
│       ├── trainer.py             # DyckDataset (train/test split) & GrammarTrainer (AdamW, annealing)
│       ├── reference.py           # Gold Dyck-2 CNF, gold prior init, negative controls, gold rule F1
│       ├── tokenizer.py           # Byte-pair vocabulary trained from scratch, protected formal alphabet
│       ├── masking.py             # Symbolic decoding mask: prefix automaton plus token-level logits mask
│       ├── model.py               # Encoder-decoder transformer from scratch: config, layers, init, torch and safetensors persistence
│       ├── training.py            # Masked supervision, AdamW plus cosine schedule, constrained batch decoding
│       ├── verification.py        # GrammarVerifier (Lark, coverage, negative controls, Z3 inclusion)
│       └── synthesize.py          # End-to-end synthesis, verification gates and export pipeline
├── tests/
│   ├── test_parser.py             # Differentiability, tensor shapes, mass invariant, Viterbi vs inside
│   ├── test_grammar.py            # Serialization, pruning, post-pruning renormalization, derive()
│   ├── test_trainer.py            # Training loop, train/test split, metric definitions
│   ├── test_reference.py          # Gold CNF, negative controls, gold rule F1 sanity
│   ├── test_tokenizer.py          # Lossless pre-tokenization, protected atoms, one token per symbol
│   ├── test_masking.py            # Legal language, refused mutations, logits mask, protocol markers
│   ├── test_model.py              # Config guards, causality, embedding tying, initialization schema, save and load
│   ├── test_training.py           # Admissible-set table, masked CE, schedules, constrained overfit proof
│   ├── test_evaluation_tools.py   # Gate arithmetic of the checkpoint evaluator (accuracy and audits)
│   ├── test_hub_package.py        # safetensors round trip, mask specification, gate refusal, release folder
│   ├── test_verification.py       # Z3 certificate (UNSAT) and refutation of over-generating grammars
│   ├── asic_reference_model.py    # Helper, not a test: independent model of the ASIC datapath, GF field and RV32E core
│   ├── test_asic_compiler.py      # Gates that recompute every numeric claim in the ASIC tree from the committed files
│   └── test_asic_core_firmware.py # RV32E firmware emulation checked against the committed self test table
├── AGENTS.md                      # Mandatory instructions for AI agents
├── pyproject.toml                 # Build metadata, packaging, and pytest configuration
├── requirements.txt               # Project dependencies
└── README.md                      # System documentation
```

The companion `dataset/` folder (corpus, chat pairs, Dyck-2 corpus, templates, and the tools that
validate it, learn its vocabulary, check its decoding mask, train the pure network on it and package the
result for publication) is documented in section 11 and in `dataset/README.md`. The learned tokenizer,
the model checkpoints and the evaluation reports live under `dataset/checkpoints/`, which is git-ignored:
the repository carries the code and the data that rebuild them, not the weights. The trained weights of
the pure model are published as `model.safetensors` on the Hugging Face Hub, and section 11 gives the
command that builds and uploads that release.

---

## 8. Final Hybrid Neural-Symbolic Artifact

The final output is an **Hybrid Neural-Symbolic Model**, comprising:
1. **Neural Weights & Embeddings Module (`artifacts/grammar_weights.pt`)**:
   Tensors `rule_logits` ($K^3$), `term_logits` ($K \times |V_T|$) and `family_logits` ($K$, the Bernoulli gate $\pi_A$ of §2.3), plus the run configuration in `metadata`.
2. **Discrete Formal Grammar (`artifacts/grammar.json`, `artifacts/grammar.yaml`, `artifacts/DynamicGrammarIta.gf`)**:
   Rigid CNF productions extracted post-pruning ($\tau_{\text{prune}} = 0.01$) and renormalized per non-terminal, with `probability_pre_pruning` kept for audit; suitable for formal compilers or symbolic parsers.
3. **Verification Report (`artifacts/metrics.json`)**:
   Train/hold-out metrics, per-non-terminal masses, gold-rule alignment, negative-control results, the Z3 verdict with its counter certificate, and the boolean acceptance gates.
4. **Parsing and Validation Runtime (`src/neuro_symbolic/`)**:
   Executable Python modules capable of parsing raw token sequences and emitting verified Abstract Syntax Trees (ASTs).

---

## 9. Formal Verification and Testing Procedure

Verification runs as four **acceptance gates** that `synthesize.py` evaluates and records in
`artifacts/metrics.json` (the exit code becomes non-zero on failure unless `--lenient` is passed):

1. **G1 — Grammar validity**:
   - `verify_lark_compilation()`: the exported EBNF compiles as a Lark Earley grammar (compilability only — it proves nothing about the generated language).
   - `verify_pcfg_distribution()`: $\left|\sum_\alpha P(A \to \alpha) - 1\right| < 10^{-6}$ for every non-terminal $A$.
2. **G2 — Corpus coverage (Coverage Rate = 100%)**:
   - `verify_corpus_coverage()` runs a **discrete CYK** (`PCFGrammar.derive`) over the corpus: a constructive derivation proof per sentence. The Lark Earley variant stays available as a cross-check (`engine="lark"`).
3. **G3 — Selectivity on negative controls**:
   - `verify_negative_controls()` measures the rejection rate over perturbed strings that provably lie outside Dyck-2 (bracket-type swaps, insertions, deletions, reversals, random words). A dense grammar scores badly here: the grammar shipped before this revision accepted `]`, `(]` and `)(`.
   - `gold_rule_f1()` adds a structural measure (precision / recall of the rule set against the gold CNF under the best non-terminal alignment).
4. **G4 — SMT language inclusion ($L(G) \subseteq \mathrm{Dyck}\text{-}2$)**:
   - `verify_language_soundness()` first searches exhaustively for a **witness**: a string generated by $G$ (derivation tree included) that violates the bracket invariant, up to `--search-bound`.
   - Otherwise it asks Z3 for a **counter certificate** $(\delta_r, \delta_s, m_r, m_s)_A$ per non-terminal, constrained by the transfer equations

$$A \to a:\ \delta = e(a),\ m = e(a) \qquad A \to B\,C:\ \delta = \delta_B + \delta_C,\ m = \min(m_B,\ \delta_B + m_C)$$

   - If that system is satisfiable and the start symbol satisfies $\delta = 0 \wedge m \ge 0$, then adding $\neg G$ makes the solver answer **UNSAT**: by induction on derivation depth no sentential form can break the prefix invariant, so no syntactic hallucination is derivable — at any length, not merely up to the search bound.
   - **Refutation** is reported when a witness exists (`SAT`, with the offending string and its tree), as asserted by `tests/test_verification.py::test_overgenerating_grammar_is_refuted`.
   - **Caveats, stated openly**: the certificate is *sound but incomplete* (a language-correct grammar may admit no per-non-terminal signature, yielding `INCONCLUSIVE`), and witness search is bounded by `--search-bound`. The former single-string check `check_word_balance_z3()` is kept only for diagnosis: the prefix invariant now sits in the **thesis** behind a premise-consistency guard, because the previous encoding put it among the premises and returned vacuous `UNSAT / sound` verdicts for `())`, `)(` and `][`.

### Verification Matrix (seed 42 run, `python src/neuro_symbolic/synthesize.py`)

| Component Under Test | Verification Tool | Acceptance Criterion | Result |
| :--- | :--- | :--- | :--- |
| **Grammar compilability** | Lark Earley (`verify_lark_compilation`) | 0 grammar exceptions | **PASS** |
| **PCFG distribution invariant** | `verify_pcfg_distribution` | Mass $=1$ per non-terminal (error $< 10^{-5}$) | **PASS (1.2e-07)** |
| **Corpus coverage** | Discrete CYK (`verify_corpus_coverage`) | 100% coverage rate | **PASS (100.0%)** |
| **Selectivity (negative controls)** | `verify_negative_controls` | 100% rejection of 200 non-Dyck strings | **PASS (200/200)** |
| **Rule induction quality** | `gold_rule_f1` | $F_1 \ge 0.90$ under NT alignment | **PASS (1.000)** |
| **Language inclusion** | Z3 SMT Solver | **UNSAT** on certificate $+ \neg G$ | **PASS (UNSAT)** |
| **Unit / integration tests** | pytest | All tests PASSED | **PASS (184 passed)** |

---

## 10. Installation and Execution Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Z3 Solver (`z3-solver`)
- Lark (`lark`)
- PyYAML (`pyyaml`)
- NumPy (`numpy`)
- Safetensors (`safetensors`), the release format of the pure-model weights
- pytest (`pytest`)

The ASIC track of section 12 needs nothing beyond CPython and pytest for generation and gates. Verilator 5.006 or later, Icarus Verilog 11 or later, Yosys 0.38 or later and OpenLane 1 or 2 with the SkyWater `sky130A` PDK are external programs, not Python packages, and none of them is committed to this tree. Verilator 5.054 ran from the image `verilator/verilator:latest`, which is how the lint and both simulation paths of section 12.3 were executed; OpenLane 2.3.10 ran the full place and route in WSL Ubuntu on 2026-11-09, with the sky130A PDK downloaded by volare and the tools taken from the image `ghcr.io/efabless/openlane2:2.3.10`, and Yosys ran inside that flow as its synthesis step; Icarus Verilog has never run here. `asic_chip/doc/build_instructions.md` says what to install and how to run the same commands in that image, and `asic_chip/doc/verification_report.md` says what that leaves unverified.

Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
pip install -e ".[publish]"   # only needed to upload a release, adds huggingface_hub
```

### End-to-End Synthesis and Artifact Export
To train the parser on Dyck-2, compute metrics, export grammar artifacts, and run formal verification:
```bash
python3 src/neuro_symbolic/synthesize.py
```

The artifacts directory is resolved against the **repository root**, so the command works from any
working directory. The process exits with a **non-zero status when any acceptance gate fails**, which
is what `.github/workflows/ci.yml` relies on; add `--lenient` to keep exit code 0 for exploratory runs.

### Command-Line Interface
```bash
python3 src/neuro_symbolic/synthesize.py --epochs 120 --lr 3e-3 --init random --search-bound 8
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--seed` | `42` | Seed for dataset, split, initialization, negative controls. |
| `--init {prior,random}` | `prior` | Seed logits with the gold Dyck-2 CNF (`reference.py`) or start from noise. |
| `--num-nt` | `5` | Latent non-terminals. |
| `--num-samples`, `--max-depth`, `--max-length` | `100`, `3`, `12` | Dyck-2 corpus generation. |
| `--test-fraction` | `0.25` | Phrase-level train/test split (disjoint sentences). |
| `--epochs`, `--batch-size`, `--lr` | `60`, `8`, `1.0e-2` | Optimization budget (batches grouped by sentence length). |
| `--tau`, `--tau-start` | `0.5`, `1.0` | Temperature annealing; `--tau` also drives extraction. |
| `--entropy-penalty`, `--weight-decay` | `0.5`, `1.0e-4` | Regularizers of §2.4. |
| `--prune` | `0.01` | Pruning threshold before per-LHS renormalization. |
| `--search-bound` | `6` | Counterexample search length for gate G4. |
| `--artifacts-dir` | `artifacts` | Output directory (relative paths resolve against the repo root). |
| `--list-only` | off | Load an existing checkpoint / artifacts and re-run verification without training. |
| `--lenient` | off | Report gate failures without a non-zero exit code. |

### Running Test Suite
Execute unit and integration tests:
```bash
pytest -v tests/
```
The suite also carries the gates of the ASIC track described in section 12: `tests/test_asic_compiler.py`
recomputes every numeric claim in `asic_chip/` from the committed files and
`tests/test_asic_core_firmware.py` emulates the core firmware, so 184 tests pass in this
checkout with no HDL tool on the host. The ASIC steps that do need an external tool are
documented in `asic_chip/doc/build_instructions.md`: Verilator 5.054 linted and simulated the
design inside the image `verilator/verilator:latest`, OpenLane 2.3.10 with the sky130A PDK ran
two complete place and routes in WSL Ubuntu, and Icarus Verilog has never run here.

### Continuous Integration (GitHub Actions)
Tests and formal verification run automatically on push or pull requests via `.github/workflows/ci.yml` across Python **3.10**, **3.11**, and **3.12**.
The companion pipeline `.github/workflows/dataset.yml` runs on changes under `dataset/`: corpus consistency, the tokenizer gates `T1`-`T5`, the mask gates `M1`-`M5`, grammar induction, and the GitHub Release that carries the artifacts.
The ASIC gates run inside `pytest -v tests/` and therefore in CI, because they are pure Python; the Verilator and OpenLane steps are not in CI, and `asic_chip/doc/verification_report.md` states that as a limit on what is verified.

---

## 11. Synthetic Multilingual Dataset and the Pure Model Track

The folder `dataset/` is independent from the system described above: it holds a corpus
plus the scripts that build and check it, and nothing in `src/` reads it while the
assistant runs. It is the training ground of a **hobby assistant built for fun on synthetic
everyday routines**, not a component of the product. The sentences were generated from 50
patterns in five domains (home automation, agenda, finances, health, productivity), in
Italian, English, French, Spanish and German: 13,490 phrases, 2,698 distinct meanings, each
meaning written as a balanced word over `( ) [ ]`, which is the alphabet this engine induces.

The assistant trained on it is a **pure model**: no pretrained checkpoint, no external
vocabulary, no third-party weights anywhere in this repository. Phases 1 to 5 of that track
are implemented and measured, the verbalization direction of phase 4 is specified but not trained, and
the whole plan with its measured corrections is `doc/PURE_MODEL_ROADMAP.md`.

### Downloading the dataset

The corpus and everything needed to rebuild it are published as assets of the GitHub
releases, free to download. The first release is tagged **`dataset-v1`**; each new
publication increments the tag (`dataset-v2`, `dataset-v3`, ...), and
`https://github.com/tailot/LingoLOGOS/releases/latest` always points at the newest one.

```bash
gh release download dataset-v1                      # all assets of that release
gh release download dataset-v1 -r lingologos-dataset.zip
# or in a browser: https://github.com/tailot/LingoLOGOS/releases/tag/dataset-v1
```

A release carries `lingologos-dataset.zip` (data shards, chat pairs, Dyck-2 corpus,
configs, templates, tools), `grammar.json`, `grammar.yaml`, `DatasetGrammar.gf`,
`grammar-metrics.json`, `tokenizer.json` and `tokenizer-metrics.json` (the vocabulary learned
from this corpus), `masking-metrics.json` (the symbolic decoding gates) and
`dataset-stats.json`.

### Running the dataset pipeline

GitHub-hosted runners have no GPU and a six hour job limit, so `.github/workflows/dataset.yml`
runs everything that fits on a CPU (corpus consistency, the tokenizer gates, the mask gates, the
grammar induction) and then publishes the files. Neural training, which is phase 4 below, belongs
to your own machine.

```powershell
python dataset/tools/validate_dataset.py            # consistency plus Z3 on every form
python dataset/tools/train_grammar.py --epochs 90 --synthetic 900
```

### Phase 1 -- a vocabulary learned from this corpus (implemented and gated)

`src/neuro_symbolic/tokenizer.py` is a byte-pair vocabulary written from scratch, with the formal
alphabet protected: brackets, operator names and leaf tags are single tokens that no merge may ever
join, while ordinary words are learned from the text. Nothing is downloaded; `load` even refuses a
vocabulary whose format id is not `lingologos-bpe-1`.

```powershell
python dataset/tools/train_tokenizer.py --vocab-size 6000
```

The tool trains on the training split only and measures train, valid and test, writing
`dataset/checkpoints/tokenizer/tokenizer.json` and `metrics.json`:

| measured on the shipped corpus | value |
| :--- | :--- |
| training texts / pre-tokenized units | 72,840 / 5,214,389 |
| word types, atomic units, atomic types | 1,578 / 3,040,076 / 268 |
| learned merges | 3,578 (`no mergeable pair left`) |
| vocabulary size | **3,921**, saturation point (`saturated_at: 3921`) |
| round trip failures, unknown tokens | 0, 0 (over 5,786,067 produced tokens) |
| formal symbols that stay one token | 267 of 267, on every split |
| bracket words re-encoded exactly | 13,490 checked, 0 broken |

Gates `T1` reversible pre-tokenization, `T2` no unknown token, `T3` atomic alphabet, `T4` alphabet
generalizes to held-out data, `T5` algebra round trip: all pass.

### Phase 2 -- the symbolic decoding mask (implemented and gated)

`src/neuro_symbolic/masking.py` builds a stack automaton over the *symbol* alphabet that admits
exactly the legal bracket algebra: the root must be a round node, brackets must be matched, a round
node must hold exactly as many children as the arity in `dataset/templates/operators.json`, and a
leaf must match the payload shape observed in the corpus. Because an operator name reveals its
arity immediately, admissibility is decidable from the prefix alone, so the mask is a function of
the prefix and can be applied to the logits at every decoding step.

```powershell
python dataset/tools/verify_masking.py --samples 300 --refine
```

The tool checks the automaton against an **independent oracle** (it reuses `logic_map.is_well_bracketed`,
the registry arities and an independent reader, never the mask itself), writing
`dataset/checkpoints/masking/masking_metrics.json`:

| gate | measured |
| :--- | :--- |
| `M1` held-out algebra accepted | 1,350 of 1,350 |
| `M2` mutated streams refused | 16,350 of 16,350 (crossed bracket, deleted or inserted bracket, truncation, unknown operator, wrong arity, oversized leaf) |
| `M3` sampled streams that are legal trees | 300 of 300 |
| `M4` registry agrees with the corpus | 52 distinct operator and arity pairs observed in the corpus all match the registry, 0 serialization mismatch |
| `M5` token-level protocol | 67,190 of 67,190 gold decisions accepted, closing marker forced at completeness 1,350 times and offered out of place 0 times |

With `--refine` the extra rule "which child kind sits at which position" is learned from the
training split only (83 refined pairs) and the held-out split is still accepted 1,350 of 1,350, so
the narrowing is sound. Two honest caveats are in the metrics file: `unknown_leaf` controls were
all discarded because every held-out leaf is inside the training alphabet, and one of the 53
registered operators never occurs in the corpus.

### What the model will speak

The target sequence is the algebra itself, framed by the special tokens:

```
<RULE_START> ( IF [ flag 1 ] [ day mon ] ) <RULE_END> <EOS>
```

one symbol per token, single spaces between symbols. Each dataset meaning is stored in four
renderings, and the other three (semantic pattern, GF abstraction, SMT assertion) are derived
deterministically from that tree by `dataset/tools/logic_map.py`, so the model never has to
generate JSON or SMT text: one object to learn, three guaranteed views of it.

### Phases 3 to 5 -- built and measured

The network is `src/neuro_symbolic/model.py`: 6 encoder plus 6 decoder blocks, 8 heads, `d_model` 512,
`d_ff` 2048, dropout 0.1, learned absolute positions, 48,253,952 parameters, initialization exactly as
specified (LayerNorm 1/0, embeddings N(0, 0.02), Kaiming on the ReLU side of the feed-forward blocks,
Xavier elsewhere, uniform biases). The training loop is `src/neuro_symbolic/training.py` driven by
`dataset/tools/train_model.py`, and the held-out evaluation is `dataset/tools/evaluate_model.py`.

```powershell
python dataset/tools/train_model.py --epochs 20 --batch-size 64 --lr 5e-4 --warmup-steps 0 --amp auto
python dataset/tools/evaluate_model.py --checkpoint dataset/checkpoints/model/model_best.pt --split test --min-exact-tree-accuracy 0.90
```

Ledger of the real 20-epoch run on the 12,140 training pairs (seed 17, batch 64, bf16, 3,800 updates,
809 seconds, peak 2.47 GB): masked cross-entropy 0.3666 to 0.00025, perplexity 1.44 to 1.000,
free-position accuracy 0.555 to 0.9993, and greedy constrained decoding of the algebra from 12.5 to
93.8 percent of a held-out slice. Only 19.66 percent of the supervised positions are real choices: the
automaton forces the other 80 percent to a single token, and the free-position figure is measured over
the hard fifth only.

Evaluation of the best checkpoint on the 675 rows of the **test** split, gates printed by the tool:

| gate | result |
| :--- | :--- |
| `E1` exact tree, held out | **0.9393** (threshold 0.90), bracket skeleton 1.0000, truncated 0 |
| `E1` by language | de 0.933, en 0.948, es 0.941, fr 0.933, it 0.941 -- 135 rows each |
| `E1` by logic family | recurrence 1.000, frequency 1.000, conditional 0.957, **threshold 0.835** |
| `E1` by domain | agenda 1.000, produttivita 1.000, finanze 0.965, salute 0.907, domotica 0.852 |
| `E2` legality of generated streams | **675 / 675** against the independent oracle |
| `E3` formal audit of generated trees | 675 built in Z3, 0 registry rejections, 0 build errors, **675 contingent, 0 satisfiable, 0 valid** |
| `E3` invented-number audit | 5 payloads absent from the input sentence, 0.74 percent |
| `E5` ablation, same weights | legality with mask 1.0000, **legality without mask 0.0000** |

The table reports the checkpoint with the best validation loss (epoch 11). The final epoch-20 weights do
better on the gate metric itself, **0.9511 exact tree (642 of 675)**, because validation loss averages
the 80 percent of positions the automaton forces and is a weak proxy for exact-tree accuracy; both
reports and both sets of numbers are in `dataset/checkpoints/evaluation/` and in section 7 of
`doc/PURE_MODEL_ROADMAP.md`.

`E5` is the sharpest number in the table: with the mask switched off at inference time, at identical
weights, the raw decoder never reaches `<RULE_END>` and degenerates into repeated tokens (`WEEKDAY_ORD
WEEKDAY_ORD ord ord ...`), and its unmasked argmax sits outside the admissible set at 83.6 percent of
the positions even though its masked accuracy is 99.3 percent. Masked cross-entropy never penalizes an
inadmissible logit, it only refuses to reward it: well-formedness is a property of the constraint, not
of the weights, and that is what the specification asks for.

The failure modes that remain are specific and honest: errors concentrate in the `threshold` family
(0.835 over 200 rows) and in `domotica` (0.852 over 155 rows), the generated bracket skeleton is always
correct, and 5 of 675 rows carry a numeric payload the input sentence never stated. Phase 5 of the
roadmap is that script, and the phase-4 verbalization direction (tree back to a sentence) is still
specified but not trained.

### Corrections to the specification, measured rather than assumed

| the specification says | what this corpus and machine give |
| :--- | :--- |
| vocabulary `V` of 8,000 to 32,000 | the byte-pair vocabulary **saturates at 3,921** (1,578 word types, 3,040,076 atomic units): any target above that is unreachable, so `V = 3,921` is the number to size the embedding on |
| warmup of 4,000 steps | one epoch over the 12,140 training pairs is 95 steps at batch 128, so 4,000 steps of warmup would be about 42 epochs of warmup; warmup has to be a fraction of an epoch, not a fixed 4,000 |
| cosine decay of the learning rate | implemented as declared in `training.py`: linear warmup over one epoch (190 steps at batch 64), then cosine down to 10 percent of the peak rate; the grammar trainer in this repository anneals temperature linearly, so the pure model loop declares its own schedule instead of inheriting a convention |
| loss `L = L_CE + lambda * L_Symbolic` with `lambda = 0.1` | with the hard mask of phase 2 the invalid region already has zero probability mass, so the penalty term is redundant: keep it as an optional ablation, not as the mechanism that enforces legality. Measured as gate `E5`: legality 1.0000 with the mask, 0.0000 without it, at identical weights |
| minimum hardware: RTX 3090 or A100, 16 GB | the development machine has an RTX 5070 Ti with 16,303 MiB and PyTorch 2.14.1+cu130 with CUDA available, which matches the 16 GB budget. Measured at `d_model` 512 and 48,253,952 parameters: 20 epochs over 12,140 pairs in 809 seconds at batch 64 with bf16 autocast, peak process memory 2,467,995,648 bytes, 341 samples per second. Memory is not the constraint; data volume is |

### Attaching dataset artifacts to a GitHub release

```powershell
Compress-Archive -Path dataset\checkpoints\tokenizer\* -DestinationPath dist\tokenizer.zip
gh release upload dataset-v1 dist\tokenizer.zip --clobber
```

### Publishing the weights as safetensors

The published weights are **`model.safetensors`**, not a pickle. `save_model_safetensors` in
`src/neuro_symbolic/model.py` hands the whole module to `safetensors.torch.save_model`, so the tied
output projection is stored **once**: 260 tensor entries for the 261 names `state_dict` declares,
193,044,672 bytes for 48,253,952 distinct parameters. The file header carries the format tag
`lingologos-transformer-1`, the configuration as JSON, the padding id and the provenance block, and
`load_model_safetensors` rebuilds the architecture from that header and loads with `strict=True`, which
re-establishes the tie.

`dataset/tools/publish_to_hub.py` assembles the release folder and, with `--upload`, pushes it to the
Hugging Face Hub:

```powershell
python dataset/tools/publish_to_hub.py --checkpoint dataset/checkpoints/model/model_best.pt --report dataset/checkpoints/evaluation/model_best-test.json --out-dir dataset/checkpoints/hub --repo tailot/lingologos-pure-1
$env:HF_TOKEN = "..."    # the token is read from the environment, never passed as an argument
python dataset/tools/publish_to_hub.py --checkpoint dataset/checkpoints/model/model_best.pt --report dataset/checkpoints/evaluation/model_best-test.json --out-dir dataset/checkpoints/hub --repo tailot/lingologos-pure-1 --upload
```

The folder holds seven files: `model.safetensors`, `config.json`, `tokenizer.json`, `mask_spec.json`,
`metadata.json`, `evaluation_report.json` and the model card `README.md`. `mask_spec.json` counts as much
as the weights: a downloaded model without the operator arities, the leaf forms and the observed child
kinds cannot be driven as a constrained generator, so `AlgebraMask.to_spec` and `AlgebraMask.from_spec`
make the symbolic constraint part of the release. The tool refuses a checkpoint whose gate report is not
`PASS`, reloads the file it has just written and requires a bit-identical round trip (261 tensors
compared, largest logit difference 0.0 on a fixed probe batch) before anything leaves the machine, and
records the sha256 of every artifact in `metadata.json`.

The model card repeats the gate table as printed by `dataset/tools/evaluate_model.py`, the breakdown by
language, domain and logic family, the `E5` mask ablation, and what the model does not do: no
verbalization direction back to a sentence, no chat ability, and a license line because the source
repository declares none.

The raw `torch.save` checkpoint stays available as a GitHub release asset for anyone who wants the
optimizer state: measured 201,112,479 bytes for `model.pt` and 201,130,518 for `model_best.pt`, within
the 2 GiB per-file limit of a release asset, while a workflow artifact of that size would expire after
90 days and is not a public download.

```powershell
git tag model-v1
git push origin model-v1
gh release upload model-v1 dataset\checkpoints\model\model_best.pt --clobber
```

Training itself stays off hosted runners: they have no GPU, the 20-epoch run is an 809-second CUDA job
here, and a verification job only needs to run `dataset/tools/evaluate_model.py`, which exits with code 1
when any gate fails and writes the per-gate `passed` flags into its report.

---

## 12. ASIC Implementation Track (`asic_chip/`)

`asic_chip/` is the hardware side of the project: a fixed function chip that runs the same
three formal mechanisms the software trains, in silicon. It is a complete RTL tree with a
generated datapath, a core, a testbench, a C++ harness, an OpenLane configuration and four
documents. The whole tree is written to be reproducible from three JSON artifacts, and every
number in its documentation is either a measured value with the command that measured it, or
it is labeled as not measured.

### 12.1 The four engines

| Block | File | What it does | Declared resource |
|---|---|---|---|
| Weight ROM mesh | `rtl/generated/weight_rom_mesh.sv` | 4 by 4 constant dot product on four signed 8 bit lanes, as shift-add trees in non-adjacent form | no multiplier: largest shift 7, worst partial sum bound 35072 in a 20 bit accumulator, headroom ratio 14.95 |
| GF(2^8) tracker | `rtl/generated/gf_automaton_1cycle.sv` | bracket matching as a field product over the polynomial `0x11D`, one cycle per symbol | 255 reachable states, 1020 defined transition arms, one default arm that resets |
| SMT evaluator | `rtl/generated/smt_evaluator.sv` | four clauses of the constraint DSL evaluated combinationally on 8 input bits | 4 result lanes, deepest cone 3 levels, wires only, no clock |
| Control core | `rtl/core/rv32e_core.sv` | RV32E, minimal implemented subset, 97 word firmware, coprocessor window and status word | 128 word instruction ROM, 9 bit program counter, 64 word data memory, illegal encodings trap |

The top level is `rtl/top/lingologos_top.sv`: 18 public ports, 51 input bits and 95 output
bits, 146 signal bits for the pad plan. Twelve further debug ports, 302 bits, exist only
under `LL_DEBUG_PORTS` for simulation and never reach synthesis or the pin plan. The GF
tracker keeps the one byte state budget the specification declares, which is why it
recognizes the commutative image of Dyck-2 and accepts `(]`; that limitation is written in
the artifact and in the datasheet rather than hidden.

### 12.2 One source of truth

```sh
python asic_chip/scripts/lingologos_compiler.py          # regenerate, then gate H1 to H5
python asic_chip/scripts/lingologos_compiler.py --check  # prove the committed files are the current ones
python -m pytest -v tests/                               # the gate suite required by AGENTS.md
```

The generator is the only description of the datapath: it emits the three generated modules,
`sim/golden_vectors.mem` (304 rows with expected results), `sim/core_self_test.mem` and
`rtl/generated/MANIFEST.json` with the sha256 digest of every artifact and every generated
file. A failing gate stops the run, so an artifact that does not fit the accumulator or a GF
alphabet that leaves the state table cannot silently produce RTL. `asic_chip/scripts/build_gds.sh`
refuses to start a place and route until `--check` passes, which is what keeps a stale
`rtl/generated` directory out of a layout.

### 12.3 Verification status, executed against not run

`python -m pytest -v tests/` in this checkout: **184 passed**, zero failures, zero errors.
Twenty six of those tests target the ASIC tree: twenty in `tests/test_asic_compiler.py`
and six in `tests/test_asic_core_firmware.py`, plus the helper model in
`tests/asic_reference_model.py` that recomputes the dot product, the GF field, the clause
masks and the RV32E semantics without importing the generator.

| Level | Content | Status in this checkout |
|---|---|---|
| L0 to L3 | artifact validity, generator gates H1 to H5, agreement of two independent models on every vector row and every input byte, byte level reproducibility and digests | EXECUTED by CPython 3.14.5 and pytest 9.1.1 |
| Core behavior | the 97 word firmware emulated by the reference model: 89 cycles, 4 coprocessor accesses, one illegal `x16` probe at ROM slot 79, final registers `x8=0x00000009`, `x10=0x00000002`, `x11=0x0000104A`, `x13=0x00000013` | CROSS-CHECKED by emulation, and confirmed by simulation: both paths below assert the same readback, and the harness measures 91 cycles for the same run |
| L4 | Verilator lint, the C++ harness over the 304 golden rows, the SystemVerilog testbench, Icarus | EXECUTED in the container image `verilator/verilator:latest` with Verilator 5.054: lint clean under `-Wall`, and both the harness and the testbench reporting `checked 304 vector rows plus the core self-test, 0 mismatches` and `FINAL VERDICT: PASS`, through the make targets of `asic_chip/Makefile`. NOT RUN on the host, NOT RUN with Icarus |
| L5 | synthesis, floorplan, placement, route, DRC, LVS, static timing, power, area, GDSII | Partly EXECUTED on 2026-11-09, in WSL Ubuntu with openlane 2.3.10 and the sky130A PDK: two runs went through all 78 steps of the Classic flow, in 16 min 59 s and in 12 min 12 s, each wrote three GDSII files, and both passed DRC with zero violations in Magic and in KLayout and LVS with `Circuits match uniquely`, each with area, utilization, IR drop and static timing over nine corners. The second reported `Antenna Passed`, which the first failed on 54 pins and 46 nets. Both exited non zero on deferred setup and hold violations at the bring-up period each was constrained to, 8.0 ns and 30.0 ns. Static timing at the declared 4.0 ns budget has never been run, and no frequency is claimed |

The verification layer found real things, and the report names all twenty three of them. The
Python gates caught the reference model using the wrong boolean identity for `MUX`, a stale
artifact digest and the wrong combinational depth in the datasheet, generated files written
with platform line endings, and switching activity figures that did not match the committed
vector file. Linting the design then caught an illegal part select in the branch and jump
cone, a combinational loop through a write through read path in the register file, blocking
writes inside a sequential block, silent width widening in the generated mesh, and a dangling
port at the top level. The two simulation paths caught a testbench that was checking 1024
vector rows instead of 304, a checker that sampled the registered outputs one delta too
early, and a coprocessor counter expectation that was off by one in both checkers. All of it
is fixed: the cases the Python gates found have a test that fails if they return, and the
others stop the lint or the build of the same flow. Running the place and route then found
three more, all of them in this tree's own configuration: a pin plan file written in a syntax
the placement parser rejects, which stopped the first run at step 25 of 78; three timing and
signoff keys that the installed OpenLane does not read at all, so the thing they appeared to
gate was gated by nothing; and a legacy diode strategy key that silently overwrote the antenna
knobs written next to it, which is one reason the antenna check still failed at the end of the
run that got through everything else. All three are corrected in the configuration files, and
`tests/test_asic_compiler.py` fails if any of them comes back. The rerun with those knobs written the
way OpenLane 2 reads them, at a 30.0 ns gate, passed the antenna check and both rule checks, with
`Antenna Passed`, `LVS Passed` and `DRC Passed` in its manufacturability report, and still exited non
zero: setup in one of nine corners at -0.4906 ns, and hold in seven corners at a worst -0.1540 ns, every
violating path from an input port to a register. The gate is 32.0 ns now and both resizers are asked for
0.40 ns of hold slack instead of the 0.05 and 0.10 ns OpenLane defaults to, because that run measured
hold repaired at +0.100201 ns and lost about 0.16 ns of it to detailed routing and extraction; that
attempt has not run yet. Read
`asic_chip/doc/verification_report.md` before repeating any claim from the datasheet.

### 12.4 Documents

| Document | Content |
|---|---|
| `asic_chip/doc/datasheet_lingologos.md` | pinout for the declared QFN-160 body, port table, per block measured properties, stimulus statistics, the declared operating point with each electrical cell either marked NOT MEASURED or labeled with the run, period and corner a measured figure came from, and the pointer to the report behind each |
| `asic_chip/doc/integration_guide.md` | how to drive the chip: reset and start protocol, the `gf_symbol_in` hazard and how to avoid it, the implemented RV32E subset as the decoder sees it, the firmware map, the coprocessor window, the four places to change together when the pin plan moves, and the deviation table |
| `asic_chip/doc/verification_report.md` | what ran, what did not, tool inventory, gate by gate results, defects caught, and what a reviewer still has to do |
| `asic_chip/doc/build_instructions.md` | prerequisites with versions, every command for generation, pytest, Verilator, Icarus and OpenLane, make target table, platform notes, troubleshooting |

No new third party dependency enters the project for this track: the generator uses only the
Python standard library, and the gate suite needs the `pytest` already declared in
`requirements.txt` and `pyproject.toml`. Verilator, Icarus, Yosys and OpenLane are external
tools the build instructions describe and that this checkout does not carry; the OpenLane runs
above happened in WSL Ubuntu, in its own virtual environment, pulling its tools from a container
image and its PDK from a download outside this tree. Each of those runs left its own directory under
`asic_chip/openlane/runs/` in the tree it ran from, and none of those directories is committed.
