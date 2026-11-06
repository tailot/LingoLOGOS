# Pure Model Roadmap: the Assistant Trained From Scratch

This document is the working specification of the **pure neuro-symbolic assistant** for the
`dataset/` corpus: a model with no pretrained checkpoint, no downloaded vocabulary and no
third-party weights anywhere. It records what is already built and measured, what is still
missing, and the corrections that measurement forced on the original specification.

Related: `doc/TECHNICAL_DOCUMENTATION.md` sections 1.6 and 1.7 (tokenizer and mask design),
`dataset/TRAINING.md` (commands, flags, gates), `README.md` section 11 (short version).

---

## 1. Non-negotiable constraints

1. **Zero pre-training.** Every tensor starts from a stochastic initialization. No
   `from_pretrained`, no imported vocabulary, no external corpus. The only training data is
   this repository's corpus. The former fine-tuning branch that wrapped an external chat model
   has been deleted rather than kept as an alternative.
2. **The symbolic constraint is hard, not soft.** Legality of the output is enforced by the
   decoding mask of `src/neuro_symbolic/masking.py` (admissible logits only, everything else at
   $-\infty$), not by a penalty that the network may or may not respect.
3. **Every answer is checkable.** A generated bracket tree is rebuilt into a logical form and
   handed to Z3, the same way `dataset/tools/validate_dataset.py` checks the corpus.
4. **One object, four views.** The model produces the bracket tree. The semantic pattern, the GF
   abstraction and the SMT assertion are derived from it by `dataset/tools/logic_map.py`, so the
   model never emits JSON and cannot make the four views disagree. This is a change from the
   `sft/*.jsonl` rows, whose assistant turn is a JSON object with four string keys: that format
   belongs to the deleted external-model branch.

## 2. Status

| phase | content | state |
| --- | --- | --- |
| 1 | data audit, grammar of the algebra, **vocabulary learned from the corpus** | done and gated (`T1`-`T5` all pass) |
| 2 | **prefix automaton of admissible symbols** plus token-level logits mask | done and gated (`M1`-`M5` all pass) |
| 3 | encoder-decoder network, stochastic initialization | specified below, not written |
| 4 | training loop with masked cross-entropy | specified below, not written |
| 5 | held-out benchmark and Z3 audit of generated answers | specified below, not written |

Reproduce phases 1 and 2:

```powershell
python dataset/tools/train_tokenizer.py --vocab-size 6000
python dataset/tools/verify_masking.py --samples 300 --refine
python -m pytest -v tests/
```

## 3. Output protocol

```
<RULE_START> ( IF [ flag 1 ] [ day mon ] ) <RULE_END> <EOS>
```

one symbol of the algebra per token, exactly one space token between two symbols, the closing
marker offered only when the automaton reports the tree complete and followed only by `<EOS>`.
Special tokens `<PAD> <UNK> <BOS> <EOS> <RULE_START> <RULE_END>` are ids 0 to 5 of the learned
vocabulary. The verbalization direction (tree to sentence) uses the same vocabulary without the
mask, and is the second task of the model.

## 4. Measured data ledger

Measured with the trained vocabulary and the trained mask on the shipped corpus:

| quantity | train | valid | test |
| --- | --- | --- | --- |
| tree-to-meaning pairs (the masked task) | 12,140 | 675 | 675 |
| algebra symbols per target (min / mean / p50 / p99 / max) | 9 / 24.0 / 22 / 45 / 45 | 9 / 22.8 / 22 / 45 / 45 | 12 / 24.0 / 22 / 45 / 45 |
| target token ids, `<RULE_START>` to `<EOS>` inclusive | $\le 92$ | $\le 92$ | $\le 92$ |
| encoder input tokens of the utterance (min / mean / p50 / p99 / max) | 11 / 32.7 / 31 / 60 / 69 | | |

Consequences for the design: the encoder never needs more than 96 positions, the decoder target
never needs more than 96 positions, so there is no truncation anywhere in the dataset, and the
whole sequence budget is far below what a 16 GB card handles at batch 64 or 128.

Alphabet and vocabulary: 3,921 learned entries, with ids 0 to 5 reserved for
`<PAD> <UNK> <BOS> <EOS> <RULE_START> <RULE_END>`. 268 of those entries carry the masked
alphabet (4 brackets, 53 operators, 136 leaf tags, 75 payload tokens); the single space between
two symbols is its own atomic token, and `<EOS>` plus `<PAD>` complete the protocol.

## 5. Phase 3 - network and initialization

Encoder-decoder transformer, no pretrained part:

| component | value |
| --- | --- |
| layers | 6 encoder + 6 decoder |
| attention heads | 8, `d_head` 64 |
| `d_model` | 512 |
| `d_ff` | 2048 |
| dropout | 0.1 |
| vocabulary | 3,921 (learned in phase 1), input and output embeddings shared |
| positions | learned, max 96 (measured maximum 69 encoder, 92 decoder) |
| initialization | linear and attention projections: Kaiming for ReLU-side matrices and Xavier for the rest; embedding and output projection: $N(0, 0.02)$ |
| parameter count | about 48 million (6 encoder layers at 3.15 M each, 6 decoder layers at 4.19 M each, 2.0 M for the shared embedding table, layernorms and biases) |

The two facts that make this small model practical here: the language it must learn is
**context-free and the mask removes every illegal continuation**, so the network does not have to
memorize bracket balance; and the sequences are short, so a full attention matrix at 96 positions
is negligible.

`<RULE_START>` is the only decoder input at the first step; the mask then supplies the rest of the
structure. This is also the honest reason to keep the model small: with 12,140 training pairs, a
network of this size is already at the point where regularization, not capacity, decides the result.

## 6. Phase 4 - training loop

| hyperparameter | planned value |
| --- | --- |
| optimizer | AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$ |
| learning rate | $5 \cdot 10^{-4}$ |
| warmup | 190 steps (one epoch at batch 64), linear ramp |
| decay | cosine to $5 \cdot 10^{-5}$ over the remaining steps |
| weight decay | 0.01, not applied to biases and layernorm gains |
| batch | 64 (128 if the memory probe leaves headroom), gradient clip 1.0 |
| loss | cross-entropy on the masked logits, `<PAD>` ignored; optional $\lambda L_{\mathrm{Symbolic}}$ ablation only |
| seed | fixed per run, recorded in the metrics file together with the vocabulary statistics |

Loss detail. `TokenStreamMask.mask_logits` fills inadmissible entries with $-\infty$ and leaves the
admissible scores untouched, so the cross-entropy is taken over the real alternatives only and the
reported perplexity is perplexity over legal continuations. The soft symbolic term of the original
specification is therefore **not the mechanism**: it is kept as an optional ablation to measure how
much a penalty would add on top of a constraint that is already exact.

Reported per validation epoch: masked cross-entropy, perplexity per symbol, exact-tree accuracy on
the held-out split (the whole target sequence, teacher forced), legality rate of free sampling with
the mask (must be 100%, it is a regression guard on the mask, not on the model) and legality rate of
free sampling with the mask disabled (this is the number that shows what the mask buys).

## 7. Phase 5 - held-out benchmark and Z3 audit

| gate | requirement |
| --- | --- |
| E1 | exact bracket tree on the 675 held-out tree-to-meaning pairs, reported per language, per domain and per task |
| E2 | free-sampled answers: 100% accepted by the mask automaton and by the independent oracle of `verify_masking.py` |
| E3 | every generated tree rebuilt by `logic_map` into a logical form; Z3 must build it, and the answer must not contradict the quantities and units present in the input sentence |
| E4 | verbalization direction: generated sentence must keep every number, unit and cycle that appears in the input tree (checked by string inclusion of the payload tokens) |
| E5 | ablation of the mask at generation time on the same weights, reported next to the masked numbers |

E1 is the only gate whose value is unknown in advance; E2 is expected to be exactly 100 because the
mask makes illegal output unreachable, and E3 is where an honestly wrong model gets caught: a tree
that is legal but semantically inconsistent with the sentence still fails there.

## 8. Corrections to the original specification

| the specification states | measured reality in this repository |
| --- | --- |
| $V \approx 8{,}000$ to $32{,}000$ | the byte-pair vocabulary **saturates at 3,921** (1,578 word types, 3,040,076 atomic units, 3,578 merges, stop reason `no mergeable pair left`). Sizing embeddings for 32,000 would add parameters the corpus cannot fill; $V = 3{,}921$ |
| warmup of 4,000 steps | one epoch at batch 64 is 190 steps, so 4,000 steps of warmup is about 21 epochs of warmup (42 at batch 128). Warmup is set to one epoch |
| cosine decay | kept, but stated explicitly: the existing grammar trainer in this repository anneals temperature linearly, so the pure model loop declares its own schedule rather than inheriting a convention |
| $L = L_{CE} + \lambda L_{\mathrm{Symbolic}}$, $\lambda = 0.1$ | with the hard mask the invalid region has zero probability mass, so the term is redundant. Kept as an ablation (gate E5), never as the enforcement mechanism |
| minimum hardware RTX 3090 or A100, 16 GB | development machine: RTX 5070 Ti, 16,303 MiB, driver 621.14, PyTorch 2.14.1 with CUDA available, 68 GB system RAM. The 48 M parameter model is well inside that budget |
| tokenizer BPE or WordPiece with special tokens | implemented as a **protected-atom** BPE: the formal alphabet can never be split or merged, which a stock BPE does not guarantee. This is what makes one-symbol-per-token decoding and therefore the symbol-level mask possible |
| roadmap phase 1 "data and grammar", phase 2 "masking layer and incremental parser" | both done, with the incremental parser existing as the prefix automaton of `AlgebraMask` and the model-free reader `parse_algebra` used by the independent oracle |

## 9. What was deleted and why

The previous dataset pipeline wrapped a third-party chat model. It is gone, not hidden behind a
flag, because a repository whose goal is a pure model must not carry a working path around an
imported one:

- `dataset/tools/train_sft.py` (LoRA fine-tuning of `Qwen/Qwen2.5-7B-Instruct`),
  `dataset/tools/evaluate_sft.py`, `dataset/tools/ask_model.py`, `dataset/tools/publish_to_hub.py`
  and `dataset/hf_publish/MODEL_CARD.md`.
- the `fine-tuning-smoke` job and the `sft_smoke` input of `.github/workflows/dataset.yml`.

What survived from those scripts is model-free logic, reimplemented without any external
dependency: the bracket reader and the structural checks now live in `src/neuro_symbolic/masking.py`
(`parse_algebra`, `tokenize_algebra`, `stream_of_tree`, `skeleton_of_stream`) and
`dataset/tools/logic_map.py`. The chat pairs in `dataset/sft/` and the seed file
`dataset/sft/fewshot_seed.jsonl` stay: they are data, and the JSON renderings inside them are
regenerated by `dataset/tools/generate_dataset.py` from the trees, which remain the single source of
truth.

## 10. Open risks to measure in phase 4

1. **Data volume.** 12,140 pairs, 53 operators, 136 leaf tags: exact-tree accuracy may plateau well
   below 1 for the rare operator combinations. The leaf tags carry most of the difficulty, since the
   mask already fixes the shape; the model has to learn which tag belongs where.
2. **Leaves outside the training alphabet.** `M4` found 52 of 53 operators used and every held-out
   leaf inside the training leaf alphabet; a new domain means relearning both the vocabulary and the
   mask, so the gate numbers above are properties of this corpus, not general claims.
3. **Ambiguous trees.** The corpus stores one canonical tree per meaning; if the grammar accepts
   several trees for one sentence, exact-tree accuracy under-reports correctness. E3 and E4 are the
   guards against that interpretation problem.
4. **Verbalization is unmasked.** Only the tree direction is constrained, so the free-text direction
   keeps the usual failure modes and is judged only on payload preservation.
