# Neuro-Symbolic Grammar Induction and Rule Synthesis Engine

Integrated neuro-symbolic architecture designed for differentiable grammar induction and rule synthesis from unstructured data streams. The engine combines continuous representations learned via neural networks with the formal constraints of Probabilistic Context-Free Grammars (**PCFG**) in Chomsky Normal Form (**CNF**).

Published resources of this project, with what they are:

| Link | Type | Category |
| :--- | :--- | :--- |
| https://huggingface.co/datasets/tailot/lingologos-math | Topology | math |
| https://huggingface.co/datasets/tailot/lingologos-geometry | Topology | geometry |
| https://huggingface.co/datasets/tailot/lingologos-physics | Topology | physics |
| https://huggingface.co/datasets/tailot/lingologos-struc | Topology | struc |
| https://huggingface.co/datasets/tailot/math-stratum | Topology | stratum |
| https://huggingface.co/datasets/tailot/geometry-stratum | Topology | stratum |
| https://huggingface.co/datasets/tailot/physics-stratum | Topology | stratum |
| https://huggingface.co/datasets/tailot/stratum-stratum | Topology | stratum |

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
5. **Desktop Workbench (optional layer, section 16 of the technical documentation)**: a PyQt6 workbench over the whole stack in `src/gui/`, with a generic node graph, the Lllayer stream, a filesystem derived board viewer, a payload manager and a SystemVerilog analyzer. It is a closed package: no engine module imports it, it imports nothing from the engine at all, not even inside a function, and it loads no web resource. The folders and commands of the host project that it uses are declared by a workspace profile, optionally overridden by a `workbench.json` at the workspace root.

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

Five files, all of them written by `python src/neuro_symbolic/synthesize.py` and version-controlled so the formal claims stay auditable without a training run. `tests/test_grammar_artifacts.py` keeps them honest: once per session it runs the synthesis into a temporary folder and fails when a committed `grammar.json`, `grammar.yaml` or `DynamicGrammarIta.gf` no longer equals what the current code exports. The two numeric files stay out of that comparison because `grammar_weights.pt` and the measured values in `metrics.json` already differed in the eighth significant digit between two runs with the same seed, so the gates recorded in the report are checked instead of the digits.

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
│   ├── gui/                       # Desktop workbench, optional and offline: see section 16 of the technical documentation
│   │   ├── app.py                 # Offline guard, command line parsing, Qt application entry point
│   │   ├── window.py              # The shell: five modules, one shared log console, cross module routes
│   │   ├── paths.py               # Workspace discovery and the artifact roles read from disk
│   │   ├── theme.py               # Local palette, local fonts, Qt standard icons, the one style sheet
│   │   ├── widgets.py             # Log console, folding cards, chips and table helpers shared by the modules
│   │   ├── model/                 # Qt free layer: topology, board, SystemVerilog, payload, stream
│   │   ├── workers/               # Threads: workspace scan, file parse, command runner, session replay
│   │   └── views/                 # Painted modules: canvas, stream, board, payload table, source
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
│   ├── test_workbench_paths.py    # Workspace discovery and the artifact roles read from the tree
│   ├── test_workbench_profile.py  # Workspace profile overrides, features switched off, YAML topologies
│   ├── test_workbench_topology_model.py # Topology dialects, extensible kinds, cycle and DAG checks
│   ├── test_workbench_hardware_model.py # Board and fault plan read from Verilog, adaptive to N dies
│   ├── test_workbench_dataset_model.py  # Payload fields, formats, pagination, byte identical values
│   ├── test_workbench_stream_model.py   # Transcript events, pacing, verdicts, model assignment files
│   ├── test_workbench_sv_model.py       # SystemVerilog symbols, mesh masks, tasks, verdict lines
│   ├── test_workbench_runner.py         # Tool plans, verdict capture, Verilator off on Windows
│   ├── test_workbench_offline_guard.py  # No web toolkit, no network call, no remote asset in the GUI
│   ├── test_workbench_isolation.py      # The engine never imports the GUI, the GUI imports nothing of the project
│   ├── test_workbench_shell.py          # The five modules, their routes, their painting, offscreen
│   ├── test_workbench_layout.py         # Folding cards, node card sizing, canvas and schematic zoom
│   ├── asic_reference_model.py    # Helper, not a test: independent model of the ASIC datapath, GF field and RV32E core
│   ├── test_asic_compiler.py      # Gates that recompute every numeric claim in the ASIC tree from the committed files
│   └── test_asic_core_firmware.py # RV32E firmware emulation checked against the committed self test table
├── config/
│   └── topology.json              # Example topology definition and routing matrix
├── topologies/                    # Ignored by default, the shipped document is re-included
│   └── federated_mesh.json        # Example federated mesh over the three sample nodes
├── datasets/                      # Ignored by default, the shipped folders are re-included
│   ├── agente_alfa/               # Example corpus, SMT ontology, and GF grammar for node
│   ├── agente_beta/
│   ├── arbitro_logico/
│   └── <track>/                   # Example dataset tracks: own README, own tools, own numbers
├── src/
│   ├── lingologos_orchestrator/   # Distributed multi-instance orchestrator (RFC LingoLOGOS-MULTI-ORCH-v1)
│   │   ├── cli.py                 # CLI entrypoint for lingologos-orchestrator
│   │   ├── lllayer.py             # Isolated CLI node wrapper process
│   │   ├── node_runner.py         # Subprocess process isolation manager
│   │   ├── orchestrator.py        # Centralized message broker & 4-phase lifecycle
│   │   ├── schema.py              # Topology, envelope, and validation dataclasses
│   │   └── verifier.py            # Z3 SMT solver and ontology verifier
│   └── neuro_symbolic/
├── tests/
│   ├── test_orchestrator.py       # Orchestrator schema, routing, process isolation, and Z3 tests
├── AGENTS.md                      # Mandatory instructions for AI agents
├── pyproject.toml                 # Build metadata, packaging, and pytest configuration
├── requirements.txt               # Project dependencies
└── README.md                      # System documentation
```

The material that ships under `datasets/` and `topologies/` is **example material**, not a component
of the running system and not user data. The three folders `datasets/agente_alfa/`,
`datasets/agente_beta/` and `datasets/arbitro_logico/` hold the sample payload a federated node has to
provide: a corpus of sentences, an SMT ontology and a GF grammar, kept small on purpose so that the
orchestrator, the federation tests and the workbench always have the same fixed thing to route.
`config/topology.json` and `topologies/federated_mesh.json` are examples in the same sense: the first is
the three node routing matrix the local sessions run with, the second is the same three nodes reached
over the local and the Git transports, and neither one is a recommended deployment. Alongside them the
repository carries self contained example dataset tracks under `datasets/`, one folder per track: each
track documents itself in the `README.md` inside its own folder, states the commands that rebuild it, the
numbers that were measured and the limits that were found, and nothing outside the folder needs to know
about it. The point of all this material is that every gate, metric and command line in this file has
something real to run against; the system itself is the engine, the ASIC track and the workbench.

`datasets/` and `topologies/` are also the two folders where a user writes their own material, a corpus
for a new node or a topology document saved from the workbench canvas, so `.gitignore` keeps both folders
out of the index by default and re-includes only the shipped entries above. The rule is written with the
exclusion technique: `datasets/*` and `topologies/*` ignore everything, and one negation line per shipped
folder, plus the line for the one shipped topology document, brings the versioned material back. A dataset
or a topology you create is therefore never tracked and never shows up in `git status`, with no edit of
the file needed; shipping an example dataset or topology with the repository takes one extra negation
line. Section 4.3 of `doc/TECHNICAL_DOCUMENTATION.md` states the same policy and the two ordering
constraints it depends on.

An example track is self contained on purpose. Its tools resolve the corpus and the engine by walking up
from their own file instead of assuming where the folder sits, so a track can be moved, renamed, published
on its own or dropped, and whatever it learns or regenerates stays inside its own folder, out of git: the
repository carries the code and the data that rebuild those outputs, never the outputs themselves. The
orchestrator sits on the other side of that boundary: it builds paths only for the node ids declared in
`config/topology.json`, so it never reads a track folder while the assistant runs.

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
| **Unit / integration tests** | pytest | All tests PASSED | **PASS (464 passed, 10 skipped)** |

---

## 10. Installation and Execution Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Z3 Solver (`z3-solver`)
- Lark (`lark`)
- PyYAML (`pyyaml`)
- NumPy (`numpy`)
- Safetensors (`safetensors`), the release format of the trained model weights
- HTTP client for federated HTTP nodes (`httpx`)
- pytest (`pytest`)
- `tomli`, only on Python 3.10: the isolation gate of section 16 reads `pyproject.toml`, and its reader
  `tomllib` is standard library only from 3.11 on, so `requirements.txt` and the `dev` extra carry
  `tomli>=2.0.0; python_version < "3.11"` and the test falls back to it
- PyQt6 (`PyQt6`), optional and used only by the desktop workbench of section 16; the engine, the
  trainer and the orchestrator run without it, and nothing in `src/neuro_symbolic` or
  `src/lingologos_orchestrator` imports Qt

The federated Git transport of section 12 also needs `git` as an external program on the `PATH`; the
orchestrator calls it to shallow clone or pull the specification cache of a Git node.

The ASIC track of section 11 needs nothing beyond CPython and pytest for generation and gates. Verilator 5.006 or later, Icarus Verilog 11 or later, Yosys 0.38 or later and OpenLane 1 or 2 with the SkyWater `sky130A` PDK are external programs, not Python packages, and none of them is committed to this tree. Verilator 5.054 ran from the image `verilator/verilator:latest`, which is how the lint and both simulation paths of section 11.3 were executed; OpenLane 2.3.10 ran the full place and route in WSL Ubuntu on 2026-11-09, with the sky130A PDK downloaded by volare and the tools taken from the image `ghcr.io/efabless/openlane2:2.3.10`, and Yosys ran inside that flow as its synthesis step; Icarus Verilog has never run here. `asic_chip/doc/build_instructions.md` says what to install and how to run the same commands in that image, and `asic_chip/doc/verification_report.md` says what that leaves unverified.

Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
pip install -e ".[publish]"   # only needed to upload a release, adds huggingface_hub
pip install -e ".[gui]"       # only needed for the desktop workbench of section 16, adds PyQt6 and PyYAML
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

### Launching the Desktop Workbench
The workbench in `src/gui/` is a desktop application over this checkout: not a service, not a web
page. It finds the workspace by walking up from the current directory until it meets a `workbench.json`
profile file or the pair `pyproject.toml` and `src`, so it can be started from any folder inside the
repository:
```bash
lingologos-workbench                       # console script installed by pip install -e ".[gui]"
PYTHONPATH=src python3 -m gui              # from the repository root, with no install step
python3 -m gui --workspace PATH            # open another checkout than the current directory
python3 -m gui --verbose                   # also print the artifact role inventory at start up
python3 -m gui --version                   # workbench, PyQt and Qt versions, then exit
```
On Windows PowerShell the second line reads `$env:PYTHONPATH = "src"; python -m gui`. The five
modules are the generic topology canvas, the Lllayer stream, the board and hardware viewer, the
payload manager and the SystemVerilog source analyzer, and section 16 of the technical documentation
says what each of them reads.
Every view is filled from the filesystem: the tree, the schematic, the payload tables and the source
list all come from what is on disk under `config/`, `datasets/`, `export_artifacts/` and
`asic_chip/`, never from a value typed into the interface.

The space the modules give each other is liquid, not a set of fixed boxes. Every sub-block of a module
is a `widgets.Card` that folds with its own button or with a click on its title bar, and a folded card
keeps nothing but the title bar and hands its share of the column to the cards left open. A node card
on the canvas takes the size its own content needs: the title stays inside the header band, port names
get their own band and are trimmed with the full name in the tool tip, and the card grows downward,
never past the layout step, as ports and attributes are added.

The canvas and the board schematic both carry Zoom out, Zoom in and a fit button in their tool row,
plus a chip reading the current zoom. The view each module opens with is still the fit to the window it
always had. On the schematic the wheel zooms around the pointer between 25 and 800 percent, dragging
over empty space with the left button, or anywhere with the middle button, walks between blocks, the
arrow keys move the view, `+`, `-` and `0` zoom in, zoom out and fit it again, and `Center selected die`
brings the chosen block to the middle. When a board has more blocks than the drawing area can show at a
readable size, the drawing area grows and each block keeps a minimum size, so at the default fit every
block is still a box you can read rather than a dot.

Nothing in the interface fetches anything: no CDN, no remote font, no remote icon, no embedded
browser. The style sheet is a local string, the fonts are those Qt ships with and the host provides,
and the icons are the standard pixmaps of the Qt style. The start up prints that rule and the process
exits with status 3 if a web toolkit turns out to be importable in the same environment, which
`tests/test_workbench_offline_guard.py` also asserts, together with the fact that no source of the
package contains a URL, a network call or a remote asset token.

The workbench is a closed package. The engine, the trainer and the orchestrator never import `gui`,
and `gui` imports nothing from them, at module level or inside a function, and never edits
`sys.path`. It reaches the project through files it reads and commands it launches as separate
processes. Which folders and which commands, is data: a built in workspace profile, overridden by an
optional `workbench.json` at the workspace root (keys and defaults in section 16.3 of the technical
documentation). A key set to an empty string switches that feature off, and the button says why. The
`[gui]` extra is optional in `pyproject.toml`, it carries PyQt6 and PyYAML, and `PyQt6` appears in
`requirements.txt` with that same limit. `tests/test_workbench_isolation.py` enforces all of it,
including that the names of the host project appear in code only inside the profile module.

The topology canvas opens and saves JSON and YAML documents (`.json`, `.yaml`, `.yml`). PyYAML is
imported only when a YAML document is read or written, so a JSON only session never needs it.

The Verilator run buttons of the SystemVerilog module are disabled on Windows, where this repository
drives Verilator through WSL or Docker; the tooltip and the log console name the command to run in
that shell instead:
```sh
make -f asic_chip/Makefile lint-topology
make -f asic_chip/Makefile sim-topology
```
Off Windows, with `verilator` on the `PATH`, the same buttons plan and run the command and the
`FINAL VERDICT: PASS` or `FINAL VERDICT: FAIL` line of the output is captured in the log console.

### Running Test Suite
Execute unit and integration tests:
```bash
pytest -v tests/
```
The suite also carries the gates of the ASIC track described in section 11: `tests/test_asic_compiler.py`
recomputes every numeric claim in `asic_chip/` from the committed files and
`tests/test_asic_core_firmware.py` emulates the core firmware, and `tests/test_grammar_artifacts.py` re-runs the synthesis and compares the committed formal outputs, so 448 tests pass in this
checkout with no HDL tool on the host (473 collected, one skipped for a missing optional artifact). The ASIC steps that do need an external tool are
documented in `asic_chip/doc/build_instructions.md`: Verilator 5.054 linted and simulated the
design inside the image `verilator/verilator:latest`, OpenLane 2.3.10 with the sky130A PDK ran
two complete place and routes in WSL Ubuntu, and Icarus Verilog has never run here. The
topology switch of section 13 is covered by `make -f asic_chip/Makefile lint-topology` and
`sim-topology`, in the same image; it is not part of either place and route.

### Continuous Integration (GitHub Actions)
Tests and formal verification run automatically on push or pull requests via `.github/workflows/ci.yml` across Python **3.10**, **3.11**, and **3.12**.
The workbench tests build real Qt widgets, so that pipeline installs the native libraries Qt needs to load (`libegl1`, `libgl1`, `libglib2.0-0`, font config, `libxkbcommon-x11-0` and the `libxcb-*` pieces) and sets `QT_QPA_PLATFORM=offscreen`, which paints them in memory and never opens a window. On a host where the Qt runtime still cannot load, the modules that build widgets skip with the reason they hit, and the filesystem, model and isolation tests of the workbench keep running.
The matrix reaches down to 3.10 while `tests/test_workbench_isolation.py` reads `pyproject.toml`, and `tomllib` only exists from 3.11 on: that module imports it through a fallback to `tomli`, and `requirements.txt` plus the `dev` extra carry `tomli` under a `python_version < "3.11"` marker, so 3.10 installs it and newer interpreters do not.
The companion pipeline `.github/workflows/dataset.yml` belongs to one example dataset track rather than to the engine: it runs the gates of that track (corpus consistency, the vocabulary gates, the decoding mask gates, grammar induction) and publishes the release that carries its artifacts, and the README inside that track states when it runs and what it publishes.
The ASIC gates run inside `pytest -v tests/` and therefore in CI, because they are pure Python; the Verilator and OpenLane steps are not in CI, and `asic_chip/doc/verification_report.md` states that as a limit on what is verified.

---

## 11. ASIC Implementation Track (`asic_chip/`)

`asic_chip/` is the hardware side of the project: a fixed function chip that runs the same
three formal mechanisms the software trains, in silicon. It is a complete RTL tree with a
generated datapath, a core, a testbench, a C++ harness, an OpenLane configuration and four
documents. The whole tree is written to be reproducible from three JSON artifacts, and every
number in its documentation is either a measured value with the command that measured it, or
it is labeled as not measured.

### 11.1 The four engines

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

### 11.2 One source of truth

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

### 11.3 Verification status, executed against not run

`python -m pytest -v tests/` in this checkout: **464 passed, 10 skipped**, zero failures, zero errors. The ten skips are optional extras and checkpoint folders that `.gitignore` keeps out of the index, so they skip on a fresh clone as well. The synthesis that `tests/test_grammar_artifacts.py` performs once per session accounts for about 15 s of that wall time.
Twenty six of those tests target the ASIC tree: twenty in `tests/test_asic_compiler.py`
and six in `tests/test_asic_core_firmware.py`, plus the helper model in
`tests/asic_reference_model.py` that recomputes the dot product, the GF field, the clause
masks and the RV32E semantics without importing the generator.

| Level | Content | Status in this checkout |
|---|---|---|
| L0 to L3 | artifact validity, generator gates H1 to H5, agreement of two independent models on every vector row and every input byte, byte level reproducibility and digests | EXECUTED by CPython 3.14.5 and pytest 9.1.1 |
| Core behavior | the 97 word firmware emulated by the reference model: 89 cycles, 4 coprocessor accesses, one illegal `x16` probe at ROM slot 79, final registers `x8=0x00000009`, `x10=0x00000002`, `x11=0x0000104A`, `x13=0x00000013` | CROSS-CHECKED by emulation, and confirmed by simulation: both paths below assert the same readback, and the harness measures 91 cycles for the same run |
| L4 | Verilator lint, the C++ harness over the 304 golden rows, the SystemVerilog testbench, Icarus | EXECUTED in the container image `verilator/verilator:latest` with Verilator 5.054: lint clean under `-Wall`, and both the harness and the testbench reporting `checked 304 vector rows plus the core self-test, 0 mismatches` and `FINAL VERDICT: PASS`, through the make targets of `asic_chip/Makefile`. NOT RUN on the host, NOT RUN with Icarus |
| Topology fabric | the switch of section 13, four dies wired as `config/topology.json` declares: loopback, one hop delivery, refusal of a destination the mask does not name, back pressure through four buffer stages, a wire that disagrees with the mask | EXECUTED in the same container image, on 2026-11-10, through `lint-topology` and `sim-topology`: lint clean under `-Wall`, the bench reporting `checked 42 topology assertions on four dies, 0 mismatches` and `FINAL VERDICT: PASS`. NOT SYNTHESIZED: the module is in no top level and in no place and route |
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

### 11.4 Documents

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

---

## 12. Distributed Multi-Instance Orchestrator (RFC LingoLOGOS-MULTI-ORCH-v1)

The multi-instance orchestrator coordinates $N$ concurrent, autonomous LingoLOGOS instances communicating over a directed graph $G = (V, E)$. Formal language generation and algebraic reasoning are constrained by Grammatical Framework (GF) concrete grammars and the Z3 SMT solver.

### Architectural Metadata
- **RFC Identifier**: `LingoLOGOS-MULTI-ORCH-v1`
- **Interface Layer**: `lllayer` (CLI process wrapper)
- **Logical Engine**: Z3 SMT Solver (algebraic constraint satisfiability)
- **Formal Grammar**: Grammatical Framework (GF) (syntactic trees)

### Architectural Principles
1. **Node Decoupling and Isolated Processes**:
   Nodes are never imported as internal Python modules. The orchestrator spawns and interacts with each node strictly as an isolated subprocess via standard I/O streams using the `lllayer` CLI wrapper. Node identifiers (e.g., `agente_alfa`, `agente_beta`, `arbitro_logico`) are fully arbitrary and never hardcoded in source code.
2. **Volatile State and Time Management**:
   Time and contextual state parameters are managed dynamically via discrete volatile ticks (`internalTimeTick`) passed in the envelope. No persistent or fixed timestamps are stored in request payloads.
3. **Centralized Routing via Directed Graph**:
   Nodes never communicate peer-to-peer directly. The orchestrator acts as a centralized asynchronous message broker, enforcing the `routingMatrix` defined in `config/topology.json`. Any route not in $E$ is rejected.
4. **Filesystem Convention and Seed Payloads**:
   Each declared node has an isolated workspace directory under `datasets/{node_id}/` containing:
   - `ontology.smt2`: Node-specific SMT-LIB 2 sorts and declarations.
   - `grammar.gf`: Grammatical Framework concrete syntax rules.
   - `seed_payload.json`: Turn 0 seed envelope.

### Envelope Protocol (Neuro-Symbolic Envelope)
Communication between nodes uses a structured JSON envelope:
```json
{
  "sender": "agente_alfa",
  "recipient": "agente_beta",
  "turn": 1,
  "volatileState": {
    "internalTimeTick": 1
  },
  "payload": {
    "naturalLanguage": "State transition proposal under formal verification.",
    "abstractSyntaxTree": "Transition (StateA StateB)",
    "smtConstraints": {
      "declarations": [
        "(declare-sort State)",
        "(declare-fun safe_transition (State State) Bool)"
      ],
      "assertions": [
        "(assert (safe_transition s1 s2))"
      ],
      "z3Status": "sat"
    }
  }
}
```

### Lifecycle Phases
- **Phase 01: Static Validation**: Validates `topology.json` schema, resolves the transport of every federated node (a Git source is shallow cloned or pulled into `cache/nodes/<node_id>`, a local node reads `local_cache_path`), checks presence of the specification files (`ontology.smt2`, `grammar.gf`, `seed_payload.json`), and checks SMT syntax on the pooled Z3 workers.
- **Phase 02: Process Initialization**: Maps node IDs and prepares isolated subprocess runners via `lllayer`, or HTTP runners for remote nodes.
- **Phase 03: Turn Loop ($k = 1 \dots \text{maxTurns}$)**: Dispatches messages across authorized edges $(u, v) \in E$, executes node $u$ via subprocess stdin/stdout, publishes the resulting envelope on the event bus topic `channel/<sender>/<recipient>`, and verifies algebraic constraints against the node ontology using Z3.
- **Phase 04: Termination**: Concludes session upon reaching logical convergence (SAT convergence), tripping an UNSAT breaker, or reaching `maxTurns`, then releases the Z3 worker pool.

### CLI Usage
Run the orchestrator:
```bash
lingologos-orchestrator --config config/topology.json --datasets datasets/ --runtime-state runtime_state/
```

Run with federated mesh topology:
```bash
lingologos-orchestrator --config topologies/federated_mesh.json --datasets datasets/
```
The shipped `topologies/federated_mesh.json` mixes transports over the datasets of this repository:
`agente_alfa` and `arbitro_logico` are local nodes reading `datasets/<node_id>/`, and `agente_beta`
is a Git node, so the first run clones the project repository into the ignored folder
`cache/nodes/agente_beta` and reads the specification from `datasets/agente_beta/` inside that clone.
A second run only pulls the cache. Because that node needs the network the first time, run the
plain command above to reproduce the same session without any remote fetch.

Add a remote HTTP node with bearer authentication by declaring the environment variable that holds
the token; the token itself never enters the topology file:
```json
{
  "node_id": "http_api_node",
  "role": "validator",
  "transport": "http",
  "endpoint_url": "https://api.example.net/v1/evaluate",
  "auth_method": "bearer_token",
  "auth_token_env": "LINGOLOGOS_HTTP_TOKEN",
  "outbound_channels": ["agente_alfa"]
}
```
```bash
export LINGOLOGOS_HTTP_TOKEN="<token issued by the endpoint>"
```

Run with topology consolidation into a verification triad:
```bash
lingologos-orchestrator --config topologies/federated_mesh.json --datasets datasets/ --consolidate triad
```
Triad consolidation contracts the graph into `macro_proponent`, `macro_opponent`, and
`macro_validator`, and each macro node runs on the specification folder of the physical node it
represents, so the command needs no extra dataset. `--consolidate unified` collapses everything into
a single macro engine; with no peer to talk to the session reports `EMPTY_ROUTING_MATRIX`, which
makes unified mode the graph contraction used for analysis rather than a runnable dialogue graph.

Execute an isolated node CLI turn directly:
```bash
lllayer --node-id agente_alfa --role proponent --node-dir datasets/agente_alfa < input_envelope.json
```

Execute with in-memory volatile AST-guided logits masking and deterministic pruning:
```bash
lllayer --node-id agente_alfa --role proponent --node-dir datasets/agente_alfa --constrained-decoding < input_envelope.json
```

Host note for `--constrained-decoding`: the masking layer runs on Linux and on Windows. The loader
picks the shared object name of the host, `masker.dll` on Windows, `libmasker.dylib` on macOS,
`libmasker.so` elsewhere, and compiles `src/lingologos_orchestrator/native/masker.cpp` on the fly with
any GCC compatible compiler it finds through `shutil.which` (`c++`, `g++`, `clang++`, or a MinGW-w64
triplet on Windows). When no such compiler is installed, which is the case on a plain Windows
workstation, `NativeASTLogitsProcessor` uses a persistent numpy buffer and applies exactly the same
mask (`0.0` on admitted tokens, `-inf` elsewhere, Hadamard addition in place), so the emitted envelope
is byte identical and only the per-step latency differs. `LINGOLOGOS_DISABLE_NATIVE_KERNEL=1` forces
that branch on any host, and the test suite compares the two branches for equality. The flag is
consumed exclusively by the isolated `lllayer` CLI, because `NodeProcessRunner` builds a fixed node
command line without flag pass-through, so topology, federated, and consolidated sessions never
activate it. The full matrix and the measured results are recorded in section 15.5 of
`doc/TECHNICAL_DOCUMENTATION.md`.

Query the same node with a sentence instead of an envelope. `--data TEXT` (or `--data-file PATH`)
routes the text to a grammar source and returns one decoded tree, and the source decides the alphabet:

```bash
# generic: the built-in alphabet of the node, no corpus and no topology file needed
lllayer --node-id agente_alfa --node-dir datasets/agente_alfa \
  --data "notifica l utente in caso di allarme urgente" --format tree
( Transition MODIFIER )

# topology: the declared node names and the routing matrix become the alphabet
lllayer --node-id agente_alfa --node-dir datasets/agente_alfa \
  --topology-config config/topology.json \
  --data "passa da agente alfa ad arbitro logico" --format tree
( Transition agente_alfa arbitro_logico )

# what the topology declares, before asking it anything
lllayer --node-id agente_alfa --node-dir datasets/agente_alfa \
  --topology-config config/topology.json --show-topology

# dataset: an example track publishes trained weights; any corpus language in, validated tree out
lllayer --node-id agente_alfa --node-dir datasets/agente_alfa \
  --dataset-root datasets/<track> \
  --data "Se la temperatura supera 24 gradi, spegni il condizionatore."
grammar: dataset
tree:    ( IF ( BETWEEN [ temperature ] [ celsius 24 ] [ celsius 24 ] ) ( OFF [ air_conditioner ] ) )
skeleton: (([][][])([]))
valid:   yes  steps: 47
model:   datasets/<track>/checkpoints/hub model.safetensors vocab=3921 operators=53
```

With no `--grammar` the order is: the package named by `--model-package`, a package inside
`--node-dir`, `--topology-config`, then the generic alphabet. `--grammar generic|topology|dataset`
forces one and refuses instead of falling back. `--node-dir` is mandatory in turn mode, which reads
`seed_payload.json` from it, and optional in query mode: when the grammar source is a package or a
topology file the node folder plays no part, and only `--node-id` is needed to sign the answer. The bracket algebra automaton of
`src/neuro_symbolic/masking.py` feeds its admissible token ids into the same in-place masking kernel
of `--constrained-decoding`, so an illegal skeleton or an undeclared route cannot be sampled at all:
a route the routing matrix does not declare never enters the mask. After decoding the tree is checked
again for well-formedness, for grounding (every non structural symbol must appear in the sentence,
matched in accent-insensitive form), and for declaration. A refusal exits with code 2, prints one
English line on stderr listing the declared alternatives, and prints nothing on stdout. `--format
tree` is the scripting form, `--format report|json` print the evidence (mentioned symbols with their
positions, candidate routes, the decoded route, package statistics), and when an envelope is piped on
stdin the answer is an envelope carrying the tree in `payload.abstractSyntaxTree` with the evidence
under `volatileState.grammarRoute`, so the turn protocol described above still holds. The dataset
route loads the torch weights only when it is selected, and it inherits what the model it reads can
actually do: the gates those weights passed travel with their release, and the README of the track that
published them states the accuracy, the families where it is weaker and what the model does not do, which
is why the formal check stays in the path.
Section 15.8 of `doc/TECHNICAL_DOCUMENTATION.md` states the invariants and the measured commands.

---

## 13. Edge-NSAC Dynamic Board Topology (Single-Chip vs Multi-Chip)

LingoLOGOS supports dynamic hardware topology switching on board via the `TOPOLOGY_CTRL` CSR register and the `lingologos_topology_router.sv` hardware routing module:

- **Single-Chip Mode (Standalone)**:
  `TOPOLOGY_CTRL.STANDALONE = 1`. High-speed SerDes links are clock-gated for low power, and local transmission is routed directly to local reception via internal loopback. Local Z3 SMT and GF validation operate autonomously on the primary dataset.
- **Multi-Chip Mode (Distributed Mesh)**:
  `TOPOLOGY_CTRL.STANDALONE = 0`. Discovers $N-1$ neighboring nodes, reads physical node strapping (`NODE_ID`), and routes traffic across board traces or locally based on destination node ID matching. Each chip processes assigned dataset shards (`DATASET_ID`).

### The Switch That Implements It

`asic_chip/rtl/core/lingologos_topology_router.sv` is one registered single hop switch per die. `ROUTING_MASK` names direct neighbours and nothing else: a frame whose destination is not one of them is refused by the die that injected it, and that die reports it with a one cycle pulse on `routing_error`. There is no relay and no queue, because 16 mask bits cannot hold a path. In front of a destination that will not take a frame the fabric buffers four frames and refuses the fifth, so the core of the injecting die sees back pressure instead of an always ready port. `TOPOLOGY_CTRL.DATASET_ID` is deliberately not wired to the switch: it selects the corpus shard the firmware of the die loads, and the fabric cannot act on a shard number.

Two make targets cover it, and both ran in the container image `verilator/verilator:latest`:

```sh
make -f asic_chip/Makefile lint-topology
make -f asic_chip/Makefile sim-topology
```

The lint of the switch and of its bench ends with `topology switch lint clean` at exit 0. The bench instantiates four dies, wires the six links that `config/topology.json` declares, and ends with `checked 42 topology assertions on four dies, 0 mismatches` and `FINAL VERDICT: PASS`; section 2.4.1 of `asic_chip/doc/verification_report.md` holds the commands and the quoted output.

What has not been done to that module: it is not instantiated in `lingologos_top.sv` and it is in the `VERILOG_FILES` list of neither OpenLane config, so it has never been synthesized, placed or routed, and no timing, area or power figure in this repository describes it. `tests/test_topology_router.py` locks that boundary in both directions, and section 14.2 of `doc/TECHNICAL_DOCUMENTATION.md` states the design decisions and why the mask is a neighbour set rather than a route.

### Windows PowerShell & WSL Bridge
For cross-platform development on Windows host with isolated Linux execution, `.venv-train/Scripts/Activate.ps1` (and `scripts/Activate.ps1`) provides idempotent WSL environment bootstrap and auto-sync for dependencies and CLI tools.
