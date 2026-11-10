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
| `tools/` | generator, validator, grammar induction, tokenizer learning, mask verification, plus the shared root probe `corpus_paths.py` |

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

The release also carries `tokenizer.json` and `tokenizer-metrics.json` (the vocabulary learned
from this corpus and its gates) and `masking-metrics.json` (the symbolic decoding gates), so the
two trained components can be reused without retraining them. Nothing in this repository
downloads a model or a vocabulary from anywhere. Details, measured numbers and the phase plan:
see [TRAINING.md](TRAINING.md).

The same steps also run in GitHub Actions, in `.github/workflows/dataset.yml`:
the corpus is checked, the vocabulary and the mask are learned and gated, the grammar is
induced on it, and the packaged files are published at the end in a GitHub Release of this
repository.

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
