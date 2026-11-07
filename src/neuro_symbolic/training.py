"""Training machinery of the pure model: data, masked loss, schedules, decoding.

The network of :mod:`neuro_symbolic.model` stays independent of the algebra. Everything
symbolic lives here: the admissible id sets that :class:`AllowedSetTable` interns, the
cross-entropy taken only over admissible continuations, and the constrained decoding used
by the evaluation gates.

Layout
------
``label_allowed_sets``     admissible ids of each predicted position, one automaton walk
``AllowedSetTable``        interns those sets, so a batch mask is a single gather
``score_batch``            teacher-forced loss and metrics of one padded batch
``build_optimizer``        AdamW, decay on matrices only
``cosine_schedule``        linear warmup then cosine decay to a fixed fraction
``generate_batch``         greedy or sampled decoding under the token-level mask

A target is the protocol ``<RULE_START> sym " " sym ... sym <RULE_END> <EOS>``: the
decoder is teacher forced with the whole protocol and predicts one position to the right,
so the symbol at position ``i`` is scored against the admissible set of the prefix
``protocol[:i]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .masking import TokenStreamMask
from .model import EncoderDecoderModel


class TrainingError(Exception):
    """Raised when a target cannot be trained on or decoded."""


# ---------------------------------------------------------------------------
# Examples and their protocol encoding
# ---------------------------------------------------------------------------
@dataclass
class AlgebraExample:
    """One supervised pair: a sentence and the algebra that means it."""

    identifier: str
    source_text: str
    symbols: List[str]
    language: str = ""
    domain: str = ""
    logic_family: str = ""


def protocol_of(token_mask: TokenStreamMask, symbols: Sequence[str]) -> List[int]:
    """Ids of the full target protocol, start marker through end-of-sequence."""
    return token_mask.encode_target(list(symbols), add_eos=True)


def label_allowed_sets(token_mask: TokenStreamMask, symbols: Sequence[str]) -> List[Tuple[int, ...]]:
    """Admissible token ids of every predicted position, in a single automaton walk.

    The predicted positions are ``protocol[1:]``. Before a symbol the automaton offers the
    symbols that fit the current expectation; right after a symbol the choice collapses to
    the separator, or to the closing marker once the stack is empty and something has been
    emitted. The result lines up one-to-one with ``protocol[1:]``.
    """
    state = token_mask.mask.initial_state()
    sets: List[Tuple[int, ...]] = []

    def ids_for(candidate: Iterable[str]) -> Tuple[int, ...]:
        return tuple(sorted(token_mask.symbol_ids[symbol] for symbol in candidate))

    for symbol in symbols:
        sets.append(ids_for(token_mask.mask.allowed_symbols(state)))
        token_mask.mask.step(state, symbol)
        if token_mask.mask.is_complete(state):
            sets.append((token_mask.rule_end_id,))
        else:
            sets.append((token_mask.space_id,))
    sets.append((token_mask.eos_id,))
    return sets


def reference_label_allowed_sets(token_mask: TokenStreamMask, symbols: Sequence[str]) -> List[Tuple[int, ...]]:
    """The same sets recomputed by replaying each prefix, the reference implementation."""
    protocol = protocol_of(token_mask, symbols)
    return [
        tuple(sorted(token_mask.allowed_next_ids(protocol[:position])))
        for position in range(1, len(protocol))
    ]


# ---------------------------------------------------------------------------
# Interning of admissible sets
# ---------------------------------------------------------------------------
class AllowedSetTable:
    """Every distinct admissible id set of the corpus, held once as a row of a matrix.

    A training position is described by the row index of its admissible set, so the mask of
    a batch is one gather of shape ``[batch, length, vocabulary]`` and the loss is taken
    over exactly the entries the symbolic constraint allows.
    """

    def __init__(self, token_mask: TokenStreamMask) -> None:
        self.token_mask = token_mask
        self.vocabulary_size = token_mask.tokenizer.vocabulary_size
        self.rows: List[Tuple[int, ...]] = []
        self._index: Dict[Tuple[int, ...], int] = {}
        self.rule_start_row = self.intern((token_mask.rule_start_id,))

    def intern(self, ids: Iterable[int]) -> int:
        key = tuple(sorted(set(int(value) for value in ids)))
        if not key:
            raise TrainingError("an admissible set can not be empty")
        row = self._index.get(key)
        if row is None:
            row = len(self.rows)
            self.rows.append(key)
            self._index[key] = row
        return row

    def cardinality(self, row: int) -> int:
        return len(self.rows[row])

    def matrix(self, device: Optional[str] = None) -> torch.Tensor:
        """Boolean matrix ``[sets, vocabulary]`` marking the admissible entries."""
        table = torch.zeros(len(self.rows), self.vocabulary_size, dtype=torch.bool)
        for row, ids in enumerate(self.rows):
            index = torch.tensor(list(ids), dtype=torch.long)
            table[row].index_fill_(0, index, True)
        return table.to(device) if device else table

    def describe(self) -> Dict[str, Any]:
        sizes = [len(row) for row in self.rows]
        return {
            "distinct_admissible_sets": len(self.rows),
            "widest_set": max(sizes),
            "narrowest_set": min(sizes),
            "single_choice_sets": sum(1 for size in sizes if size == 1),
            "vocabulary_size": self.vocabulary_size,
        }


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    """Padded tensors of one batch, already on the device of the run."""

    identifiers: List[str]
    language: List[str]
    domain: List[str]
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    decoder_input: torch.Tensor
    labels: torch.Tensor
    states: torch.Tensor
    position_mask: torch.Tensor
    free_mask: torch.Tensor


def encode_examples(
    examples: Sequence[AlgebraExample],
    token_mask: TokenStreamMask,
    table: AllowedSetTable,
) -> List[Dict[str, Any]]:
    """Tokenize every example once, with the admissible set of each predicted position.

    The encoder input is the sentence in the learned vocabulary and the decoder target is
    the protocol. Each predicted position stores the row index of its admissible set; the
    rows that carry no prediction reuse the singleton set of the start marker, so padded
    positions never contribute an infinite term to the loss.
    """
    tokenizer = token_mask.tokenizer
    encoded: List[Dict[str, Any]] = []
    for example in examples:
        if not example.symbols:
            raise TrainingError("example without algebra symbols: " + example.identifier)
        input_ids = tokenizer.encode(example.source_text, add_bos=True, add_eos=True)
        target = protocol_of(token_mask, example.symbols)
        allowed = label_allowed_sets(token_mask, example.symbols)
        labels = target[1:]
        if len(allowed) != len(labels):
            raise TrainingError("the admissible sets do not line up with the protocol")
        states = [table.intern(ids) for ids in allowed]
        for position, label in enumerate(labels):
            if label not in table.rows[states[position]]:
                raise TrainingError(
                    "gold label is inadmissible at position " + str(position) + " of " + example.identifier
                )
        encoded.append(
            {
                "identifier": example.identifier,
                "language": example.language,
                "domain": example.domain,
                "logic_family": example.logic_family,
                "source_text": example.source_text,
                "symbols": list(example.symbols),
                "input_ids": input_ids,
                "target": target,
                "labels": labels,
                "states": states,
                "free": [table.cardinality(state) > 1 for state in states],
            }
        )
    return encoded


def make_batch(
    encoded: Sequence[Dict[str, Any]],
    token_mask: TokenStreamMask,
    table: AllowedSetTable,
    device: str,
) -> Batch:
    """Pad encoded examples into the tensors the model and the masked loss need."""
    if not encoded:
        raise TrainingError("cannot build a batch out of nothing")
    rows = len(encoded)
    width = max(len(item["input_ids"]) for item in encoded)
    height = max(len(item["labels"]) for item in encoded)
    input_ids = torch.full((rows, width), token_mask.pad_id, dtype=torch.long, device=device)
    input_mask = torch.zeros((rows, width), dtype=torch.bool, device=device)
    decoder_input = torch.full((rows, height), token_mask.pad_id, dtype=torch.long, device=device)
    labels = torch.full((rows, height), token_mask.rule_start_id, dtype=torch.long, device=device)
    states = torch.full((rows, height), table.rule_start_row, dtype=torch.long, device=device)
    position_mask = torch.zeros((rows, height), dtype=torch.bool, device=device)
    free_mask = torch.zeros((rows, height), dtype=torch.bool, device=device)
    for row, item in enumerate(encoded):
        source = item["input_ids"]
        target = item["target"]
        input_ids[row, : len(source)] = torch.tensor(source, dtype=torch.long)
        input_mask[row, : len(source)] = True
        decoder_input[row, : len(target) - 1] = torch.tensor(target[:-1], dtype=torch.long)
        labels[row, : len(item["labels"])] = torch.tensor(item["labels"], dtype=torch.long)
        states[row, : len(item["states"])] = torch.tensor(item["states"], dtype=torch.long)
        position_mask[row, : len(item["labels"])] = True
        free_mask[row, : len(item["free"])] = torch.tensor(item["free"], dtype=torch.bool)
    return Batch(
        identifiers=[item["identifier"] for item in encoded],
        language=[item["language"] for item in encoded],
        domain=[item["domain"] for item in encoded],
        input_ids=input_ids,
        input_mask=input_mask,
        decoder_input=decoder_input,
        labels=labels,
        states=states,
        position_mask=position_mask,
        free_mask=free_mask,
    )


def batch_of(encoded: Sequence[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    """Contiguous batches; shuffling is left to the caller with an explicit seed."""
    width = max(1, int(size))
    return [list(encoded[start : start + width]) for start in range(0, len(encoded), width)]


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------
def masked_logits(logits: torch.Tensor, allowed_matrix: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Turn every inadmissible entry of ``logits`` into minus infinity, position by position."""
    allow = allowed_matrix[states]
    scored = logits.clone()
    scored.masked_fill_(~allow, float("-inf"))
    return scored


def score_batch(
    model: EncoderDecoderModel,
    batch: Batch,
    allowed_matrix: torch.Tensor,
    table: AllowedSetTable,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Teacher-forced loss and metrics of one batch.

    The loss averages the negative log probability of the gold token over real positions,
    each term taken under the softmax of the admissible entries only. Metrics also report
    the share of positions where the constraint leaves a real choice, because the others
    are deterministic and contribute a zero term by construction.
    """
    if autocast_dtype is None:
        logits = model(batch.input_ids, batch.input_mask, batch.decoder_input)
    else:
        with torch.autocast(device_type=str(batch.input_ids.device.type), dtype=autocast_dtype):
            logits = model(batch.input_ids, batch.input_mask, batch.decoder_input)
    scored = masked_logits(logits.float(), allowed_matrix, batch.states)
    log_prob = torch.log_softmax(scored, dim=-1)
    gathered = log_prob.gather(-1, batch.labels.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(gathered[batch.position_mask]).all()):
        raise TrainingError("the masked cross-entropy produced a non finite value")
    weight = batch.position_mask.to(gathered.dtype)
    loss_sum = (-gathered * weight).sum()
    positions = int(weight.sum())
    loss = loss_sum / max(1, positions)

    predictions = logits.argmax(dim=-1)
    masked_predictions = scored.argmax(dim=-1)
    hits = (masked_predictions == batch.labels)
    free = batch.free_mask & batch.position_mask
    illegal_argmax = (~allowed_matrix[batch.states].gather(-1, predictions.unsqueeze(-1)).squeeze(-1)) & batch.position_mask
    value = float(loss.detach())
    return loss, {
        "loss": value,
        "loss_sum": value * max(1, positions),
        "positions": positions,
        "perplexity": math.exp(min(50.0, value)),
        "exact_sequences": int((hits | ~batch.position_mask).all(dim=1).sum()),
        "sequences": len(batch.identifiers),
        "free_positions": int(free.sum()),
        "free_correct": int((hits & free).sum()),
        "free_ratio": int(free.sum()) / max(1, positions),
        "illegal_unmasked_argmax": int(illegal_argmax.sum()),
    }


def training_step(
    model: EncoderDecoderModel,
    batch: Batch,
    allowed_matrix: torch.Tensor,
    table: AllowedSetTable,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    gradient_clip: float,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """One optimization step under the masked objective, returning its batch metrics."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = score_batch(model, batch, allowed_matrix, table, autocast_dtype)
    loss.backward()
    if gradient_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return metrics


# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------
def build_optimizer(
    model: EncoderDecoderModel,
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.98),
    epsilon: float = 1e-9,
) -> torch.optim.Optimizer:
    """AdamW with the coefficients of the specification; biases and gains take no decay."""
    decay: List[torch.Tensor] = []
    exempt: List[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name:
            exempt.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": exempt, "weight_decay": 0.0},
        ],
        lr=float(learning_rate),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(epsilon),
    )


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    final_fraction: float = 0.1,
) -> Any:
    """Linear warmup over ``warmup_steps``, then cosine decay to ``final_fraction`` of the peak."""
    warmup = max(1, int(warmup_steps))
    total = max(warmup + 1, int(total_steps))
    floor = float(final_fraction)

    def factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(1.0, float(step - warmup) / float(total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def learning_rates(scheduler: Any, steps: int) -> List[float]:
    """The rate the schedule proposes for the first ``steps`` updates, for a dry-run report."""
    return [float(scheduler.lr_lambdas[0](step)) for step in range(int(steps))]


# ---------------------------------------------------------------------------
# Constrained decoding
# ---------------------------------------------------------------------------
@dataclass
class GeneratedSequence:
    """Protocol decoded for one sentence, with the symbol stream it spelled out."""

    identifier: str
    protocol_ids: List[int]
    symbols: List[str]
    language: str = ""
    domain: str = ""
    logic_family: str = ""
    steps: int = 0
    stopped_at_end: bool = False
    probabilities: List[float] = field(default_factory=list)


@torch.no_grad()
def generate_batch(
    model: EncoderDecoderModel,
    token_mask: TokenStreamMask,
    encoded: Sequence[Dict[str, Any]],
    max_steps: int,
    mode: str = "greedy",
    temperature: float = 1.0,
    seed: int = 0,
) -> List[GeneratedSequence]:
    """Decode several sentences in parallel under the token-level symbolic mask.

    Admissibility at every step comes from :meth:`TokenStreamMask.allowed_next_ids`, the
    reference used by the mask gates, so an illegal stream can only appear through a bug in
    the model or in this routine, never by relaxing the constraint.
    """
    if mode not in ("greedy", "sample"):
        raise TrainingError("unknown decoding mode: " + mode)
    if not encoded:
        raise TrainingError("nothing to decode")
    if mode == "sample" and float(temperature) <= 0.0:
        raise TrainingError("a sampling temperature has to be positive")
    budget = model.config.max_decoder_positions
    if int(max_steps) + 1 > budget:
        raise TrainingError("max_steps plus the start marker exceeds the decoder position budget")
    device = next(model.parameters()).device
    model.eval()
    torch.manual_seed(int(seed))

    width = max(len(item["input_ids"]) for item in encoded)
    input_ids = torch.full((len(encoded), width), token_mask.pad_id, dtype=torch.long, device=device)
    input_mask = torch.zeros((len(encoded), width), dtype=torch.bool, device=device)
    for row, item in enumerate(encoded):
        source = item["input_ids"]
        input_ids[row, : len(source)] = torch.tensor(source, dtype=torch.long, device=device)
        input_mask[row, : len(source)] = True
    memory = model.encode(input_ids, input_mask)

    prefixes: List[List[int]] = [[token_mask.rule_start_id] for _ in encoded]
    probabilities: List[List[float]] = [[] for _ in encoded]
    alive = [True] * len(encoded)
    for _ in range(int(max_steps)):
        active = [row for row, flag in enumerate(alive) if flag]
        if not active:
            break
        longest = max(len(prefixes[row]) for row in active)
        decoder_input = torch.full(
            (len(active), longest), token_mask.pad_id, dtype=torch.long, device=device
        )
        for slot, row in enumerate(active):
            prefix = prefixes[row]
            decoder_input[slot, : len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=device)
        rows = torch.tensor(active, dtype=torch.long, device=device)
        logits = model.decode(decoder_input, memory[rows], input_mask[rows])
        for slot, row in enumerate(active):
            position = len(prefixes[row]) - 1
            scored = token_mask.mask_logits(logits[slot, position], prefixes[row])
            distribution = torch.softmax(scored, dim=-1)
            if mode == "sample":
                chosen = int(torch.multinomial(distribution, 1).item())
            else:
                chosen = int(torch.argmax(scored).item())
            probabilities[row].append(float(distribution[chosen]))
            prefixes[row].append(chosen)
            if chosen == token_mask.eos_id:
                alive[row] = False
            elif chosen == token_mask.pad_id:
                raise TrainingError("the mask offered the padding token during generation")

    results: List[GeneratedSequence] = []
    for row, item in enumerate(encoded):
        protocol = prefixes[row]
        results.append(
            GeneratedSequence(
                identifier=item["identifier"],
                protocol_ids=protocol,
                symbols=token_mask.tokenizer.decode_symbol_stream(protocol),
                language=item.get("language", ""),
                domain=item.get("domain", ""),
                logic_family=item.get("logic_family", ""),
                steps=len(protocol) - 1,
                stopped_at_end=protocol[-1] == token_mask.eos_id,
                probabilities=probabilities[row],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------
def aggregate(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-batch metrics into a single position-weighted summary."""
    if not metrics:
        return {}
    positions = sum(item["positions"] for item in metrics)
    sequences = sum(item["sequences"] for item in metrics)
    loss_sum = sum(item["loss_sum"] for item in metrics)
    free_positions = sum(item["free_positions"] for item in metrics)
    free_correct = sum(item["free_correct"] for item in metrics)
    mean = loss_sum / max(1, positions)
    illegal = sum(item.get("illegal_unmasked_argmax", 0) for item in metrics)
    return {
        "loss": mean,
        "perplexity": math.exp(min(50.0, mean)),
        "exact_sequence_accuracy": sum(item["exact_sequences"] for item in metrics) / max(1, sequences),
        "free_position_accuracy": free_correct / max(1, free_positions),
        "free_position_share": free_positions / max(1, positions),
        "illegal_unmasked_argmax_rate": illegal / max(1, positions),
        "positions": positions,
        "sequences": sequences,
    }
