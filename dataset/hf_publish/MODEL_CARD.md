---
base_model:
  - {{BASE_MODEL}}
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - it
  - en
  - fr
  - es
  - de
  - synthetic
  - neuro-symbolic
---

<!-- Add the license field above once the project license is chosen. The corpus and this
adapter inherit whatever license the GitHub repository carries. -->

# LingoLOGOS adapter

A LoRA adapter that maps everyday recurrence instructions in Italian, English, French,
Spanish and German onto a formal logic form, and reads that form back as a sentence.
It is a hobby model built on synthetic data: every training sentence was generated from
50 sentence patterns over five everyday domains (home automation, agenda, finances,
health, productivity). It is not a general assistant and it is not connected to the
LingoLOGOS product pipeline.

## What the model does

Input a sentence such as an Italian instruction, output one JSON object with four
equivalent renderings of the meaning: a balanced bracket tree, a readable pattern, an
abstract grammar expression and an SMT assertion. Round brackets mark operator nodes,
square brackets mark value and entity leaves, so the bracket sequence is always a
balanced word of the Dyck-2 language.

## How to load it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "{{BASE_MODEL}}"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "tailot/LingoLOGOS")
```

The prompt has to match the training format: one system message that states the task,
then one user message of the form `Italian: <sentence>` (the language name is written in
English). `ask_model.py` in this repository builds that prompt for you, parses the
answer, checks the operator arities and asks the solver whether the conditions can hold
at the same time.

```bash
python ask_model.py --model {{BASE_MODEL}} --adapter . --lang it --ask "Accendi la lampada della cucina solo se il livello di CO2 supera 50 percento"
```

## Training data

The corpus is versioned in this GitHub repository and published as a release asset,
freely downloadable: <https://github.com/tailot/LingoLOGOS/releases/tag/{{DATASET_TAG}}>.
It holds 13,490 phrases, 2,698 distinct meanings, each written in the five languages,
with the formal form of every meaning and its SMT translation. New releases increment the
tag: {{DATASET_TAG}}, then {{DATASET_TAG}} incremented, and so on; the newest one is
always listed at <https://github.com/tailot/LingoLOGOS/releases/latest>.

## Adapter settings

| setting | value |
| --- | --- |
| base model | {{BASE_MODEL}} |
| adapter rank | {{LORA_R}} |
| adapter alpha | {{LORA_ALPHA}} |
| epochs | {{EPOCHS}} |
| learning rate | {{LR}} |

## Measured behaviour

Scored on rows that were never part of training{{EVAL_ROWS}}:

| measure | value |
| --- | --- |
| exact match with the reference answer | {{EXACT}} |
| identical bracket tree | {{STRUCTURE}} |
| all three formal renderings identical | {{PAYLOAD}} |
| balanced brackets in the answer | {{WELL_FORMED}} |

The bracket tree measure is the one to read: it says how often the model produced the
exact meaning for the sentence, not just plausible text.

## Limits

- The vocabulary of operators covers the 50 sentence patterns of the corpus; sentences
outside those five domains are out of scope and the model will still produce a form for
them, which is why `ask_model.py` verifies the output instead of trusting it.
- Numbers, units and entity names come from a fixed lexicon, so unusual values are often
rendered with the closest known leaf.
- Everything was produced by a generator, so the phrasing is regular by construction.
- This repository carries the adapter only. The base model weights belong to their own
license and are downloaded from their original source.
