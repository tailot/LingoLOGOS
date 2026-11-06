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
├── doc/
│   └── TECHNICAL_DOCUMENTATION.md # Comprehensive engineering and mathematical docs
├── src/
│   └── neuro_symbolic/
│       ├── __init__.py            # Package public interface
│       ├── parser.py              # DifferentiableCYKParser, family gate, Viterbi AST extraction
│       ├── grammar.py             # PCFGrammar, Symbol, ProductionRule, CYK derive, JSON/YAML/GF/Lark exporters
│       ├── trainer.py             # DyckDataset (train/test split) & GrammarTrainer (AdamW, annealing)
│       ├── reference.py           # Gold Dyck-2 CNF, gold prior init, negative controls, gold rule F1
│       ├── verification.py        # GrammarVerifier (Lark, coverage, negative controls, Z3 inclusion)
│       └── synthesize.py          # End-to-end synthesis, verification gates and export pipeline
├── tests/
│   ├── test_parser.py             # Differentiability, tensor shapes, mass invariant, Viterbi vs inside
│   ├── test_grammar.py            # Serialization, pruning, post-pruning renormalization, derive()
│   ├── test_trainer.py            # Training loop, train/test split, metric definitions
│   ├── test_reference.py          # Gold CNF, negative controls, gold rule F1 sanity
│   └── test_verification.py       # Z3 certificate (UNSAT) and refutation of over-generating grammars
├── AGENTS.md                      # Mandatory instructions for AI agents
├── pyproject.toml                 # Build metadata, packaging, and pytest configuration
├── requirements.txt               # Project dependencies
└── README.md                      # System documentation
```

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
| **Unit / integration tests** | pytest | All tests PASSED | **PASS (43 passed)** |

---

## 10. Installation and Execution Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Z3 Solver (`z3-solver`)
- Lark (`lark`)
- PyYAML (`pyyaml`)
- pytest (`pytest`)

Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
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

### Continuous Integration (GitHub Actions)
Tests and formal verification run automatically on push or pull requests via `.github/workflows/ci.yml` across Python **3.10**, **3.11**, and **3.12**.

---

## 11. Synthetic Multilingual Dataset, Fine-Tuning Runs and Published Files

The folder `dataset/` is independent from the system described above: it holds a corpus
plus the scripts that build and check it, and nothing in `src/` reads it while the
assistant runs. It is a **hobby model built for fun on synthetic everyday routines**, not
a component of the product. The sentences were generated from 50 patterns in five
domains (home automation, agenda, finances, health, productivity), in Italian, English,
French, Spanish and German: 13,490 phrases, 2,698 distinct meanings, each meaning written
as a balanced word over `( ) [ ]`, which is the alphabet this engine induces.

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
`grammar-metrics.json` and `dataset-stats.json`.

### Training on your own machine

GitHub-hosted runners have no GPU and a six hour job limit, so the pipeline in
`.github/workflows/dataset.yml` checks the corpus, induces the grammar and publishes the
files; the chat fine-tuning runs on your GPU.

```powershell
python -m venv --system-site-packages .venv-train
.\.venv-train\Scripts\Activate.ps1
pip install "transformers>=4.52,<5" "huggingface-hub>=0.30,<1.0" peft accelerate

python dataset/tools/validate_dataset.py            # consistency plus Z3 on every form
python dataset/tools/train_grammar.py --epochs 90 --synthetic 900
python dataset/tools/train_sft.py --model Qwen/Qwen2.5-3B-Instruct --limit 1000 --epochs 1 --batch-size 2 --grad-accum 8 --lr 2e-4 --max-seq-length 512 --grad-checkpointing --out-dir dataset/checkpoints/sft-probe
python dataset/tools/train_sft.py --model Qwen/Qwen2.5-3B-Instruct --epochs 2 --batch-size 2 --grad-accum 8 --lr 2e-4 --max-seq-length 512 --grad-checkpointing --save-every 1000
python dataset/tools/evaluate_sft.py --self-check   # scores the scorer, must be 100
python dataset/tools/evaluate_sft.py --model Qwen/Qwen2.5-3B-Instruct --adapter dataset/checkpoints/sft --limit 200
```

Sizing on a 16 GB card: a seven billion parameter model in bfloat16 needs about 15 GB on
its own, so it only fits with `--batch-size 1 --grad-accum 16 --max-seq-length 384
--grad-checkpointing`, while a three billion parameter model runs comfortably at
`--batch-size 2 --max-seq-length 512`. Time the probe run above and multiply by 24 to get
one epoch over the full 24,280 rows. Outputs land in `dataset/checkpoints/`, which stays
out of git. `dataset/TRAINING.md` documents every flag and the environment problem with
`huggingface-hub 1.33`.

### Using the trained model

`evaluate_sft.py` answers the only question that matters: how often the model produces
the exact bracket tree of a sentence (`structure`) and how often its brackets are
balanced at all (`well formed`), split by language, task and domain. `ask_model.py`
queries the model the way it was trained and then verifies the answer: bracket balance,
operator arities from `dataset/templates/operators.json`, and whether the solver finds
the conditions consistent.

```powershell
python dataset/tools/ask_model.py --model Qwen/Qwen2.5-3B-Instruct --adapter dataset/checkpoints/sft --lang it --ask "Accendi la lampada della cucina solo se il livello di CO2 supera 50 percento"
python dataset/tools/ask_model.py --model Qwen/Qwen2.5-3B-Instruct --adapter dataset/checkpoints/sft --lang de
python dataset/tools/publish_to_hub.py                # preview of the hub files
python dataset/tools/publish_to_hub.py --apply        # upload to the hub model repository
```

`publish_to_hub.py` sends only the files that make the adapter usable, to
`https://huggingface.co/tailot/LingoLOGOS`: the adapter and tokenizer files, the model
card generated from `dataset/hf_publish/MODEL_CARD.md`, the measured scores and
`ask_model.py`. It refuses large weight files, so nobody uploads a third-party base
model by accident, and it only writes to the hub when you pass `--apply` with a write
token in `HF_TOKEN`. To attach the adapter to a GitHub release instead:

```powershell
Compress-Archive -Path dataset\checkpoints\sft\* -DestinationPath dist\chat-adapter.zip
gh release upload dataset-v1 dist\chat-adapter.zip --clobber
```
