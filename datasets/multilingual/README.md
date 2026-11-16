# LingoLOGOS dataset

**Project code and architecture: [https://github.com/tailot/LingoLOGOS](https://github.com/tailot/LingoLOGOS)**

This folder is the training data companion of LingoLOGOS. It contains
13,490 everyday instructions written in Italian, English, French, Spanish and
German, each paired with the exact meaning the assistant is expected to act on.

The folder is independent from the rest of the system: it holds the corpus plus
the small scripts that build and check it, and nothing in `src/` reads it while
the assistant runs. Everything here was produced from the sentence patterns in
`templates/`, so this is a hobby model built for fun on synthetic everyday
routines, not a component of the product.

**Everything in this folder is an example.** The corpus, the learned vocabulary,
the mask specification, the induced grammar, the trained network and every number
reported for them in this file and in [TRAINING.md](TRAINING.md) are a worked
demonstration of the pipeline on synthetic data, kept so that every gate and every
measurement of the engine has something real to run against. None of it is a
feature of the assistant, none of it is loaded by the product at run time, and
none of it is a claim about a deployed system. The repository presents this folder,
like the other material under `datasets/` and the topology documents under
`topologies/`, as sample material: the system itself is the engine, the ASIC track
and the workbench.

## What the assistant is supposed to do

You say a routine sentence you repeat at home or at work:

- "Se la temperatura del salotto supera 24 gradi, spegni il condizionatore."
- "If the balance drops below 300 euro, pause the monthly transfer and alert me."
- "Alle 18:30 ricordami la pastiglia se la glicemia e sotto la soglia."

The assistant should understand the sentence, keep the quantities, the schedule and the
condition, and produce an unambiguous instruction that a symbolic solver can check before
anything is executed. It must never guess the semantics of a routine: the meaning is
stated, so the action can be verified, refused or explained.

That mapping is what this corpus teaches in both directions: from the phrase to the
meaning, and from the meaning back to a natural phrase in the user's language. Five areas
of daily life are covered: home automation, agenda, personal finances, health,
productivity. The model that learns it is built from scratch, with no pretrained weights
and no downloaded vocabulary: the plan, the parts that are already trained and the parts
that are missing are in `doc/PURE_MODEL_ROADMAP.md` at the repository root.

## What is inside

| Path | Content |
| --- | --- |
| `data/<lang>/` | 13,490 items, one JSON object per line, in five language folders |
| `sft/` | 26,980 chat pairs (phrase to meaning, meaning to phrase), plus 50 seed pairs |
| `corpus/` | the structural words of every item, for grammar induction |
| `templates/` | the 50 sentence patterns and the multilingual lexicon behind the items |
| `configs/` | split definitions, one per language plus the combined one |
| `tools/` | generator, validator, grammar induction, tokenizer learning, mask verification, model training, held-out evaluation, hub publishing, plus the shared root probe `corpus_paths.py` |
| `checkpoints/` | what the tools learn and measure, generated locally and kept out of git |
| `../src/neuro_symbolic/` | not part of this folder: the engine its tools reuse unchanged (byte-pair vocabulary, symbol mask, transformer, training loop) |

Every item is a parallel group: the same meaning in the five languages, with the
identical logical form. 2,698 distinct meanings, 26,980 language views of them,
13,490 distinct phrases.

## Quick start

```bash
python datasets/multilingual/tools/validate_dataset.py      # re-checks every consistency rule
python datasets/multilingual/tools/train_tokenizer.py       # learns the vocabulary from the training split
python datasets/multilingual/tools/verify_masking.py        # checks the symbolic decoding mask, gates M1-M5
python datasets/multilingual/tools/train_grammar.py         # induces the grammar of the algebra
```

Outputs go to `datasets/multilingual/checkpoints/`, which stays out of git: the learned vocabulary in
`checkpoints/tokenizer/tokenizer.json`, its gate report in `checkpoints/tokenizer/metrics.json`,
the mask report in `checkpoints/masking/masking_metrics.json`, the grammar in
`checkpoints/grammar/`.

Those commands run from any working directory, and this folder does not have
to sit at the path above: each tool walks up from its own file to find the
corpus (marker `templates/operators.json`) and the repository root (marker
`pyproject.toml` with `src`), so moving or renaming the folder needs no source
change. `LINGOLOGOS_DATASET_ROOT` and `LINGOLOGOS_REPO_ROOT` redirect the two
probes when the corpus lives outside this repository.

This folder is also published as a release of the repository, free to download: the
first release is tagged `dataset-v1` and the tags increment from there (`dataset-v2`,
`dataset-v3`, ...), with `releases/latest` pointing at the newest.

```bash
gh release download dataset-v1 -r lingologos-dataset.zip
```

A release carries `lingologos-dataset.zip` (the items, the chat pairs, the structural
corpus, the templates, the configurations and the tools), `grammar.json`,
`grammar.yaml`, `DatasetGrammar.gf` and `grammar-metrics.json` (the grammar the
pipeline induced, with its gates `G1` to `G4`), `tokenizer.json` and
`tokenizer-metrics.json` (the vocabulary learned from this corpus, with its gates
`T1` to `T5`), `masking-metrics.json` (the symbolic decoding gates `M1` to `M5`)
and `dataset-stats.json`. So the two trained components can be reused without
retraining them. Nothing in this repository downloads a model or a vocabulary from
anywhere. Details, commands and the full gate tables: see [TRAINING.md](TRAINING.md);
the summary of what was measured, the corrections it produced and the publishing
route of the trained weights are in the three sections below.

The same steps also run in GitHub Actions, in `.github/workflows/dataset.yml`:
the corpus is checked, the vocabulary and the mask are learned and gated, the grammar is
induced on it, and the packaged files are published at the end in a GitHub Release of this
repository. The pipeline never trains the network: hosted runners have no GPU and a six
hour limit, the 20 epoch run of this corpus is an 809 second CUDA job on the development
machine, and no stage of the pipeline contacts a hub.

## What was measured here

Every stage of this corpus has been run on the development machine and gated by its
own tool. The full tables, the commands and the honest caveats are in
[TRAINING.md](TRAINING.md); this is the summary, with the numbers that belong to the
shipped corpus.

| stage | gate | measured |
| --- | --- | --- |
| grammar induction | `G1` to `G4` | all true: exported grammar compiles with total mass 1, generates 100 percent of the dataset bracket words, rejects all 29 ill-bracketed controls, and Z3 proves `L(G) subset of Dyck-2` with `UNSAT` |
| learned vocabulary | `T1` to `T5` | all pass: 0 round trip failures and 0 unknown tokens over 5,786,067 produced tokens, 267 of 267 formal symbols encoded as one token, 13,490 bracket words re-encoded exactly |
| decoding mask | `M1` to `M5` | all pass: 1,350 of 1,350 held-out streams accepted, 16,350 of 16,350 mutated streams refused, 300 of 300 constrained samples legal, 67,190 of 67,190 gold token decisions accepted |
| trained network | `E1` exact tree on the 675 test rows | 0.9393 at epoch 11 (best validation loss), 0.9511 at epoch 20, bracket skeleton 1.0000 with 0 truncations |
| trained network | `E2` and `E3` | 675 of 675 streams accepted by an independent oracle; all 675 rebuilt in Z3 as contingent, 0 satisfiable, 0 valid, 0 build errors; 5 rows (0.74 percent) carry a number the sentence never stated |
| trained network | `E5` mask ablation, identical weights | legality 1.0000 with the mask, 0.0000 without it |

`E5` is the number that says what the symbolic constraint contributes: with the mask
switched off at inference time the decoder never reaches the closing marker and
repeats symbols until the step budget expires, and its raw argmax sits outside the
admissible set at 83.6 percent of the teacher-forced positions even though its
masked accuracy is over 99 percent. The mistakes that remain at `E1` are labelling
mistakes, not structural ones: they concentrate in the `threshold` logic family and
in the `domotica` domain, while the language of the sentence is not what moves the
score. Gate `E4`, verbalization from tree back to sentence, is specified but not
trained, so it is not reported.

### Corrections the corpus forced on the specification

| the specification says | what this corpus and this machine give |
| --- | --- |
| vocabulary `V` of 8,000 to 32,000 | the byte-pair vocabulary **saturates at 3,921** (1,578 word types, 3,040,076 atomic units): any target above that is unreachable, so `V = 3,921` is the number to size the embedding on |
| warmup of 4,000 steps | one epoch over the 12,140 training pairs is 95 steps at batch 128 and 190 steps at batch 64, so 4,000 steps of warmup would be about 42 epochs: warmup has to be a fraction of an epoch |
| cosine decay of the learning rate | implemented as declared: linear warmup over one epoch, then cosine down to 10 percent of the peak rate; the schedule belongs to this loop rather than to the annealing convention of the grammar trainer |
| loss `L = L_CE + lambda * L_Symbolic` with `lambda = 0.1` | under the hard mask of phase 2 the invalid region already has zero probability mass, so the penalty is redundant: keep it as an ablation, not as the mechanism that enforces legality. Measured as `E5`, 1.0000 against 0.0000 at identical weights |
| minimum hardware: RTX 3090 or A100, 16 GB | an RTX 5070 Ti with 16,303 MiB matches the 16 GB budget: 20 epochs over 12,140 pairs in 809 seconds at batch 64 with bf16 autocast, peak process memory 2,467,995,648 bytes, 340.6 samples per second. Memory is not the constraint; data volume is |

## Publishing the trained weights

The published weights are **`model.safetensors`**, not a pickle, on the Hugging Face
Hub as `tailot/lingologos-pure-1`. `save_model_safetensors` in
`src/neuro_symbolic/model.py` hands the whole module to `safetensors.torch.save_model`,
so the tied output projection is stored once: 260 tensor entries for the 261 names
`state_dict` declares, 193,044,672 bytes for 48,253,952 distinct parameters. The header
carries the format tag `lingologos-transformer-1`, the configuration, the padding id and
the provenance block, and `load_model_safetensors` rebuilds the architecture from that
header and loads with `strict=True`, which re-establishes the tie.

`tools/publish_to_hub.py` assembles the release folder and, with `--upload` and the
access token read from `$env:HF_TOKEN`, pushes it; without the flag the tool touches no
network at all. The folder holds seven files: `model.safetensors`, `config.json`,
`tokenizer.json`, `mask_spec.json`, `metadata.json`, `evaluation_report.json` and the
generated model card `README.md`. `mask_spec.json` counts as much as the weights: a
downloaded model without the operator arities, the leaf forms and the observed child
kinds cannot be driven as a constrained generator, so `AlgebraMask.to_spec` and
`AlgebraMask.from_spec` make the symbolic constraint part of the release. The tool
refuses a checkpoint whose gate report is not `PASS`, refuses a checkpoint whose recorded
vocabulary disagrees with the tokenizer it names, reloads the file it has just written and
requires a bit-identical round trip (261 tensor names, 50,261,504 parameter entries,
largest logit difference 0.0) before anything leaves the machine, and records the sha256 of
every artifact in `metadata.json`.

The model card repeats the gate table as printed by `tools/evaluate_model.py`, the
breakdown by language, domain and logic family, the `E5` mask ablation, and what the model
does not do: no verbalization direction back to a sentence, no chat ability, and a license
line because the source repository declares none.

Extra artifacts go to a GitHub release instead:

```powershell
Compress-Archive -Path datasets\multilingual\checkpoints\tokenizer\* -DestinationPath dist\tokenizer.zip
gh release upload dataset-v1 dist\tokenizer.zip --clobber
```

and the raw `torch.save` checkpoint stays available the same way, for anyone who wants the
optimizer state: measured 201,112,479 bytes for `model.pt` and 201,130,518 for
`model_best.pt`, inside the 2 GiB limit of a release asset, while a workflow artifact of
that size would expire after 90 days.

```powershell
git tag model-v1
git push origin model-v1
gh release upload model-v1 datasets/multilingual/checkpoints/model/model_best.pt --clobber
```

## Running a sentence through the orchestrator

The published folder is also a grammar source: `lllayer` can decode a sentence of this
corpus with the trained weights and the same mask, and hand back a tree that the symbolic
path then checks.

```bash
lllayer --node-id agente_alfa --node-dir datasets/agente_alfa \
  --dataset-root datasets/multilingual \
  --data "Se la temperatura supera 24 gradi, spegni il condizionatore."
grammar: dataset
tree:    ( IF ( BETWEEN [ temperature ] [ celsius 24 ] [ celsius 24 ] ) ( OFF [ air_conditioner ] ) )
skeleton: (([][][])([]))
valid:   yes  steps: 47
model:   datasets/multilingual/checkpoints/hub model.safetensors vocab=3921 operators=53
```

`--grammar dataset` forces that route and refuses instead of falling back to another
grammar. The route reads the folder this tool writes in `checkpoints/hub`, loads the torch
weights only when it is selected, and inherits the measured limits of those weights: exact
tree accuracy below 1, weaker on the `threshold` family, bracket skeleton always correct,
legality 675 out of 675, which is why the formal check of the tree stays in the path. The
bracket automaton feeds its admissible token ids into the in-place masking kernel of
`--constrained-decoding`, so an illegal skeleton cannot be sampled at all; after decoding
the tree is checked again for well-formedness, for grounding of every non structural
symbol in the sentence, and for declaration.

## Notes for the reader

- Each item stores the meaning in four equivalent renderings: a bracket tree, a
  human-readable pattern, an abstract grammar expression and an SMT assertion.
- Only one of them is a model target: the bracket tree, written as
  `<RULE_START> ( IF [ flag 1 ] [ day mon ] ) <RULE_END> <EOS>` with one symbol per
  token. The other three are derived from that tree by `tools/logic_map.py`, so the
  model never has to generate JSON or SMT text and cannot make those views disagree
  with the tree it chose.
- The bracket notation is the piece the symbolic engine learns; it is balanced by
  construction, and the validator proves it.
- Split assignment happens at the level of whole parallel groups, so no meaning
  appears both in training and in evaluation.

Licensing follows the [LingoLOGOS repository](https://github.com/tailot/LingoLOGOS).
