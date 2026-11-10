# Pure Model Roadmap: the Assistant Trained From Scratch

This document is the working specification of the **pure neuro-symbolic assistant** for the
`datasets/multilingual/` corpus: a model with no pretrained checkpoint, no downloaded vocabulary and no
third-party weights anywhere. It records what is already built and measured, what is still
missing, and the corrections that measurement forced on the original specification.

Related: `doc/TECHNICAL_DOCUMENTATION.md` sections 1.6 and 1.7 (tokenizer and mask design),
`datasets/multilingual/TRAINING.md` (commands, flags, gates), `README.md` section 11 (short version).

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
   handed to Z3, the same way `datasets/multilingual/tools/validate_dataset.py` checks the corpus.
4. **One object, four views.** The model produces the bracket tree. The semantic pattern, the GF
   abstraction and the SMT assertion are derived from it by `datasets/multilingual/tools/logic_map.py`, so the
   model never emits JSON and cannot make the four views disagree. This is a change from the
   `sft/*.jsonl` rows, whose assistant turn is a JSON object with four string keys: that format
   belongs to the deleted external-model branch.

## 2. Status

| phase | content | state |
| --- | --- | --- |
| 1 | data audit, grammar of the algebra, **vocabulary learned from the corpus** | done and gated (`T1`-`T5` all pass) |
| 2 | **prefix automaton of admissible symbols** plus token-level logits mask | done and gated (`M1`-`M5` all pass) |
| 3 | encoder-decoder network, stochastic initialization | done and gated (`test_model.py`, 20 checks) |
| 4 | training loop with masked cross-entropy | done and measured (20 epochs on the RTX 5070 Ti, ledger below) |
| 5 | held-out benchmark and Z3 audit of generated answers | done and measured on 675 test rows for both checkpoints (section 7); gates E1, E2, E3, E5 pass, E4 out of scope |
| release | packaging of what phases 3 to 5 produced: weights in safetensors, mask specification, gate report | published as `tailot/lingologos-pure-1` and verified: round trip of the written file reproduces the checkpoint with a maximum logit difference of 0.0, 48,253,952 parameters, 193,044,672 bytes |

Reproduce phases 1 and 2:

```powershell
python datasets/multilingual/tools/train_tokenizer.py --vocab-size 6000
python datasets/multilingual/tools/verify_masking.py --samples 300 --refine
python -m pytest -v tests/
```

Reproduce the model track (809 seconds of training on the RTX 5070 Ti, about 7 minutes per evaluation
pass over the 675 test rows including the Z3 audit and the ablation):

```powershell
python datasets/multilingual/tools/train_model.py --epochs 20 --seed 17
python datasets/multilingual/tools/evaluate_model.py --checkpoint datasets/multilingual/checkpoints/model/model_best.pt --split test --min-exact-tree-accuracy 0.90
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
| vocabulary | 3,921 (learned in phase 1); the decoder input embedding and the output projection are **tied**, the encoder keeps its own table |
| positions | learned, max 96 (measured maximum 69 encoder, 92 decoder) |
| initialization | linear and attention projections: Kaiming for ReLU-side matrices and Xavier for the rest; embedding and output projection: $N(0, 0.02)$ |
| parameter count | **48,253,952** trainable, from `ModelConfig.parameter_breakdown`: encoder blocks 18,914,304, decoder blocks 25,224,192, token tables 4,015,104, position tables 98,304, layernorms 2,048, tied output projection 0 |

Two corrections to the table above that only became visible once the code ran. The specification
assumed one shared embedding for both sides; the trained model uses **two** token tables (encoder
and decoder) and ties only decoder input to output projection, because the encoder reads free text
while the decoder emits the protocol alphabet: one table shared by two such different distributions
buys nothing. Tying saves exactly $V \cdot d_{model} = 3{,}921 \cdot 512 = 2{,}007{,}552$ parameters,
which `test_model.py` checks directly.

The two facts that make this small model practical here: the language it must learn is
**context-free and the mask removes every illegal continuation**, so the network does not have to
memorize bracket balance; and the sequences are short, so a full attention matrix at 96 positions
is negligible.

`<RULE_START>` is the only decoder input at the first step; the mask then supplies the rest of the
structure. This is also the honest reason to keep the model small: with 12,140 training pairs, a
network of this size is already at the point where regularization, not capacity, decides the result.

Measured at construction, seed 17, on the shipped configuration: `initialization_report` returns 100
weight matrices, standard deviation 0.044993 over them, no non-finite parameter, matrix rows from 96
(the position tables) to 3,921 (the token tables). One detail of PyTorch 2.14 is worth recording
because the tests encode it: `nn.init.kaiming_uniform_(weight, a=sqrt(5), nonlinearity="relu")` applies
the ReLU gain, so the ReLU-side matrices follow the pure He bound $\sqrt{6/fan_{in}}$, measured
$\max|w| = 0.3062$ at $fan_{in} = 64$, while the Xavier-side matrices stay at $\sqrt{6/(2 \cdot fan_{in})} = 0.2165$.
The specification said "Kaiming and Xavier" without saying which matrix gets which rule; the code
assigns He to the inner projection of every feed-forward block and Xavier to everything else, and
`tests/test_model.py` asserts the two distributions are distinguishable rather than assuming it.

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

Reported per epoch: masked cross-entropy, perplexity, free-position accuracy, exact target-sequence
accuracy under teacher forcing, the same four on the validation split, the legality and exactness of
greedy constrained decoding on a slice of the validation split, throughput and peak memory. The full
history lands in `datasets/multilingual/checkpoints/model/history.json` together with `train_metrics.json`, and the
best-validation-loss weights are kept separately as `model_best.pt`.

Ledger of the real run (seed 17, batch 64, bf16 autocast, 190 updates per epoch, 3,800 updates,
wall clock 809.2 s on the RTX 5070 Ti):

| quantity | epoch 1 | epoch 20 |
| --- | --- | --- |
| train masked cross-entropy | 0.3666 | 0.000246 |
| perplexity over admissible alternatives | 1.44 | 1.00025 |
| free-position accuracy | 0.555 | 0.99931 |
| exact target sequence, teacher forced (train) | 0.021 | 0.99333 |
| validation loss | 0.1112 | 0.00558, best 0.00404 at epoch 11 |
| exact target sequence, teacher forced (validation) | 0.090 | 0.96148 |
| exact stream, greedy constrained decoding of 16 validation sentences | 0.125 | 0.9375 |
| unmasked argmax outside the admissible set | -- | 0.834 |
| throughput | -- | 340.6 samples/s, 16,681 supervised positions/s |
| peak memory of the process | -- | 2,467,995,648 bytes on a 16 GB card |

Two numbers in that table deserve a plain reading. Only **19.66 percent** of the supervised positions
are real choices: the rest are single-token positions the automaton forces (the space between symbols,
the closing marker, `<EOS>`), so a free-position accuracy of 0.999 is 0.999 over the hard fifth of the
sequence, not over the whole sequence. And the network, read without the mask, puts its raw argmax
outside the admissible set at **83.4 percent** of positions even at convergence, because masked
cross-entropy never penalizes an inadmissible logit: it only refuses to reward it. That single number
is the quantitative statement of why the constraint is not a training trick but part of the model.

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

### Measured on the test split

Both checkpoints, 675 held-out rows, greedy constrained decoding, seed 17, batch 32, Z3 pass enabled
(`python datasets/multilingual/tools/evaluate_model.py --checkpoint ... --split test --min-exact-tree-accuracy 0.90`);
reports in `datasets/multilingual/checkpoints/evaluation/model_best-test.json` and `model-test.json`, which also
carry the per-gate `passed` flags and the overall `final_verdict`:

| gate | `model_best.pt` (epoch 11, best validation loss) | `model.pt` (epoch 20, final) |
| --- | --- | --- |
| E1 exact tree | 0.9393 (634 / 675) | **0.9511** (642 / 675) |
| E1 bracket skeleton | 1.0000 | 1.0000 |
| E1 truncated at the position budget | 0 | 0 |
| E1 mean probability of the selected token | 0.9995 | 0.9998 |
| E2 streams accepted by the independent oracle | 675 / 675 | 675 / 675 |
| E3 trees rebuilt in Z3, registry rejections, build errors | 675, 0, 0 | 675, 0, 0 |
| E3 contingent, unsatisfiable, valid | 675, 0, 0 | 675, 0, 0 |
| E3 payloads absent from the input sentence | 5 (0.0074) | 7 (0.0104) |
| E5 legality with mask versus without | 1.0000 versus 0.0000 | 1.0000 versus 0.0000 |

Three readings of that table matter more than the headline number.

**The errors are labelling errors, not structure errors.** The bracket skeleton is correct on all 675
generated streams, including the wrong ones, and every generated stream is legal. What the model gets
wrong is *which* operator and *which* leaf tag sits at a position, and the mistakes are not spread
evenly. On the epoch-20 checkpoint (the breakdown for `model_best.pt` is in the report and points the
same way): by logic family, `recurrence` (255 rows) and `frequency` (35) are at 1.000, `conditional`
(185) at 0.989 and `threshold` (200) at 0.845; by domain, `agenda` and `produttivita` are at 1.000,
`finanze` 0.957, `domotica` 0.916, `salute` 0.900; by language the spread is 0.941 to 0.956, so
multilinguality is not the difficulty. Threshold-style sentences are the weak spot of this corpus.

**E3 catches the semantic slip that E2 cannot.** Threshold rules compare a quantity against a bound,
and on 7 of 675 rows the generated tree states a number the input sentence never contains (for
instance `SAL-10-0046-de: 78.0`). That is exactly what the invented-number audit is for: the stream is
legal, the tree is well-typed, Z3 builds the form, and all 675 forms are contingent (none is a
contradiction, none is a validity) - the answer is still wrong about the world. The rate is 1.0 percent
against a threshold of 2 percent; the honest reading is that the model does not invent numbers often,
does invent them, and the audit is what notices.

**Checkpoint selection follows validation loss, not exact-tree accuracy**, and on this run that choice
costs 1.2 points: the epoch-11 weights have the better validation loss (0.00404 versus 0.00558) and the
worse test accuracy (0.9393 versus 0.9511). The loss averages the 80 percent of positions the automaton
forces, so it is a weak proxy for the metric the gate reports. Selecting the checkpoint on validation
exact-tree accuracy is the first change to make in the next run.

The ablation E5 is the cleanest number in the report. With the mask switched off at generation time, at
identical weights, 0 of 675 streams reach `<RULE_END>`, none is legal, and the decoder degenerates into
repetition (`WEEKDAY_ORD ord ord ord ...`) until the step budget runs out. Note the shape of the
failure: the unmasked decoder still emits only tokens of the algebra alphabet (0 rows outside it), so
the *vocabulary* choice is learned and the *structure* is not. Read at the token level, the raw argmax
of the trained model sits outside the admissible set at 83.6 percent of the teacher-forced positions
while the masked argmax is right at 99.3 percent of the free ones. The constraint carries
well-formedness; the weights carry the labelling.

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

- `datasets/multilingual/tools/train_sft.py` (LoRA fine-tuning of `Qwen/Qwen2.5-7B-Instruct`),
  `datasets/multilingual/tools/evaluate_sft.py`, `datasets/multilingual/tools/ask_model.py`, `datasets/multilingual/tools/publish_to_hub.py`
  and `datasets/multilingual/hf_publish/MODEL_CARD.md`.
- the `fine-tuning-smoke` job and the `sft_smoke` input of `.github/workflows/dataset.yml`.

What survived from those scripts is model-free logic, reimplemented without any external
dependency: the bracket reader and the structural checks now live in `src/neuro_symbolic/masking.py`
(`parse_algebra`, `tokenize_algebra`, `stream_of_tree`, `skeleton_of_stream`) and
`datasets/multilingual/tools/logic_map.py`. The chat pairs in `datasets/multilingual/sft/` and the seed file
`datasets/multilingual/sft/fewshot_seed.jsonl` stay: they are data, and the JSON renderings inside them are
regenerated by `datasets/multilingual/tools/generate_dataset.py` from the trees, which remain the single source of
truth.

## 10. Open risks, measured where measurement exists

1. **Data volume.** 12,140 pairs, 53 operators, 136 leaf tags. Measured: exact-tree accuracy on held-out
   data reaches 0.951, and the remaining error is concentrated, not diffuse - the `threshold` family
   sits at 0.845 over 200 rows while `recurrence` and `frequency` are perfect. The prediction that the
   leaf labels, not the shape, carry the difficulty is confirmed: skeleton accuracy is 1.000.
2. **Leaves outside the training alphabet.** `M4` found 52 of 53 operators used and every held-out leaf
   inside the training leaf alphabet; the evaluation confirms the boundary from the other side, since
   0 rows of the test split fall outside the mask alphabet. A new domain means relearning both the
   vocabulary and the mask, so the gate numbers above are properties of this corpus, not general
   claims about the architecture.
3. **Ambiguous trees.** The corpus stores one canonical tree per meaning, so exact-tree accuracy can
   under-report correctness. Measured, the under-report is small but real: the skeleton is always right,
   so a wrong tree differs from the gold one by an operator or leaf label. `E3` rebuilds each generated
   tree into a logical form and all 675 are contingent, so none of the wrong trees is a contradiction;
   a genuine semantic equivalence check between generated and gold forms is the natural next gate and
   is not implemented.
4. **Verbalization (phase 4 direction, gate E4).** Still unmeasured: only the tree direction is trained
   and constrained, so gate E4 remains a specification. It is not covered by the numbers in section 7,
   which all describe sentence-to-tree.
5. **Invented quantities.** New in phase 5 and not anticipated by the original risk list: the model
   emits a numeric payload the input never states on 1.0 percent of held-out rows. The mask cannot
   prevent it (the payload is admissible), and only the audit catches it. Tightening this means either a
   payload-copying constraint at the mask level or more threshold data; both are open.
6. **Checkpoint selection.** Validation loss picked epoch 11 and gave up 1.2 accuracy points relative to
   the final weights. The selection criterion should be the gate metric, not the loss.
