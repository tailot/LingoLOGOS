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

## What the trained model does

You say a routine sentence you repeat at home or at work:

- "Se la temperatura del salotto supera 24 gradi, spegni il condizionatore."
- "If the balance drops below 300 euro, pause the monthly transfer and alert me."
- "Alle 18:30 ricordami la pastiglia se la glicemia e sotto la soglia."

The model understands the sentence, keeps the quantities, the schedule and the
condition, and produces an unambiguous instruction that a symbolic solver can
check before anything is executed. The assistant never guesses the semantics of
a routine: the meaning is stated, so the action can be verified, refused or
explained.

The dataset teaches that mapping in both directions: from the phrase to the
meaning, and from the meaning back to a natural phrase in the user's language.
Five areas of daily life are covered: home automation, agenda, personal
finances, health, productivity.

## What is inside

| Path | Content |
| --- | --- |
| `data/<lang>/` | 13,490 items, one JSON object per line, in five language folders |
| `sft/` | 26,980 chat pairs (phrase to meaning, meaning to phrase) for fine-tuning |
| `corpus/` | the structural words of every item, for grammar induction |
| `templates/` | the 50 sentence patterns and the multilingual lexicon behind the items |
| `configs/` | split definitions, one per language plus the combined one |
| `tools/` | generator, validator and the two training entry points |

Every item is a parallel group: the same meaning in the five languages, with the
identical logical form. 2,698 distinct meanings, 26,980 language views of them,
13,490 distinct phrases.

## Quick start

```bash
python dataset/tools/validate_dataset.py      # re-checks every consistency rule
python dataset/tools/train_grammar.py         # induces the grammar of the algebra
python dataset/tools/train_sft.py             # fine-tunes a chat model on the pairs
python dataset/tools/evaluate_sft.py          # scores a trained model on held-out rows
python dataset/tools/ask_model.py             # queries the model and checks its answer
python dataset/tools/publish_to_hub.py        # prepares the adapter files for the hub
```

This folder is also published as a release of the repository, free to download: the
first release is tagged `dataset-v1` and the tags increment from there (`dataset-v2`,
`dataset-v3`, ...), with `releases/latest` pointing at the newest.

```bash
gh release download dataset-v1 -r lingologos-dataset.zip
```

`publish_to_hub.py` uploads only the files that make a trained adapter usable (adapter,
tokenizer files, model card, measured scores, `ask_model.py`) to the model repository on
the hub; base model weights are refused. Details, hyperparameters and the environment to
use: see [TRAINING.md](TRAINING.md).

The same steps also run in GitHub Actions, in `.github/workflows/dataset.yml`:
the corpus is checked, the grammar is induced on it, and the packaged files are
published at the end in a GitHub Release of this repository. The workflow itself
touches no external hub; pushing a trained adapter to the model repository is a
separate local step with `publish_to_hub.py`.

## Notes for the reader

- Each item stores the meaning in four equivalent renderings: a bracket tree, a
  human-readable pattern, an abstract grammar expression and an SMT assertion.
- The bracket notation is the piece the symbolic engine learns; it is balanced by
  construction, and the validator proves it.
- Split assignment happens at the level of whole parallel groups, so no meaning
  appears both in training and in evaluation.

Licensing follows the [LingoLOGOS repository](https://github.com/tailot/LingoLOGOS).
