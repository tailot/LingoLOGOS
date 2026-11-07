"""Tests for the training machinery: label sets, masked loss, schedules, constrained decoding."""

import math
import random

import pytest
import torch
import torch.nn.functional as F

from neuro_symbolic.masking import AlgebraMask, AlgebraMaskError, TokenStreamMask
from neuro_symbolic.model import EncoderDecoderModel, ModelConfig, apply_initialization
from neuro_symbolic.tokenizer import ALGEBRA_BRACKETS, Tokenizer
from neuro_symbolic.training import (
    AllowedSetTable,
    AlgebraExample,
    TrainingError,
    aggregate,
    batch_of,
    build_optimizer,
    cosine_schedule,
    encode_examples,
    generate_batch,
    label_allowed_sets,
    learning_rates,
    make_batch,
    masked_logits,
    protocol_of,
    reference_label_allowed_sets,
    score_batch,
    training_step,
)

ARITIES = {"IF": 2, "AT": 1, "SCHEDULE": 1, "WEEKDAY_ORD": 2, "EVERY": 1}
LEAF_PAYLOADS = [["flag", "1"], ["day", "mon"], ["clock", "07", "00"], ["ord", "3"], ["month"], ["x"]]
LEGAL = [
    "( AT [ clock 07 00 ] )",
    "( IF [ flag 1 ] [ day mon ] )",
    "( SCHEDULE [ month ] )",
    "( WEEKDAY_ORD [ ord 3 ] ( EVERY ( IF [ flag 1 ] [ day mon ] ) ) )",
    "( IF [ x ] ( AT [ day mon ] ) )",
]
PAIRS = [
    ("manda un promemoria ogni lunedi alle 07:00", "( AT [ clock 07 00 ] )"),
    ("se attivo allora programma il promemoria", "( IF [ flag 1 ] [ day mon ] )"),
    ("pianifica il mensile", "( SCHEDULE [ month ] )"),
    ("terzo lunedi alle sette con richiamo ogni giorno attivo", "( WEEKDAY_ORD [ ord 3 ] ( EVERY ( IF [ flag 1 ] [ day mon ] ) ) )"),
    ("attiva il promemoria giornaliero", "( IF [ x ] ( AT [ day mon ] ) )"),
    ("promemoria attivo alle sette", "( IF [ flag 1 ] ( AT [ clock 07 00 ] ) )"),
    ("pianifica il mensile se attivo", "( IF [ flag 1 ] ( SCHEDULE [ month ] ) )"),
    ("richiama il promemoria ogni giorno", "( EVERY ( SCHEDULE [ month ] ) )"),
    ("terzo lunedi con promemoria attivo", "( WEEKDAY_ORD [ ord 3 ] ( IF [ flag 1 ] [ day mon ] ) )"),
    ("pianifica un richiamo al giorno attivo", "( EVERY ( IF [ flag 1 ] ( SCHEDULE [ month ] ) ) )"),
    ("alle sette pianifica il mensile", "( AT ( SCHEDULE [ month ] ) )"),
    ("se attivo al terzo lunedi programma", "( IF [ flag 1 ] ( WEEKDAY_ORD [ ord 3 ] ( SCHEDULE [ month ] ) ) )"),
]
TEXTS = [sentence for sentence, _ in PAIRS] + [word for _, word in PAIRS]
ALGEBRA_TOKENS = sorted(
    {str(token) for payload in LEAF_PAYLOADS for token in payload} | set(ARITIES) | set(ALGEBRA_BRACKETS)
)


@pytest.fixture(scope="module")
def token_mask():
    mask = AlgebraMask.from_leaf_payloads(ARITIES, LEAF_PAYLOADS)
    tokenizer = Tokenizer.train(TEXTS, vocab_size=4000, min_frequency=2, atomic_symbols=ALGEBRA_TOKENS)
    return TokenStreamMask(mask, tokenizer)


def examples_of(pairs=PAIRS):
    return [
        AlgebraExample(
            identifier="pair-" + str(index),
            source_text=sentence,
            symbols=words.split(),
            language="it",
            domain="test",
        )
        for index, (sentence, words) in enumerate(pairs)
    ]


def tiny_model(token_mask, seed=3):
    config = ModelConfig(
        vocabulary_size=token_mask.tokenizer.vocabulary_size,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        attention_heads=4,
        feed_forward_size=64,
        dropout=0.0,
        max_encoder_positions=64,
        max_decoder_positions=96,
    )
    return apply_initialization(EncoderDecoderModel(config, padding_id=token_mask.pad_id), seed=seed)


# ---------------------------------------------------------------------------
# Admissible sets of the predicted positions
# ---------------------------------------------------------------------------
def test_the_reference_streams_are_legal_trees(token_mask):
    for text in LEGAL:
        token_mask.mask.validate_stream(text.split())


def test_the_incremental_pass_matches_the_prefix_reference(token_mask):
    streams = [words.split() for _, words in PAIRS]
    rng = random.Random(11)
    for _ in range(12):
        streams.append(token_mask.mask.sample_stream(rng, max_depth=4, max_symbols=60))
    for symbols in streams:
        fast = label_allowed_sets(token_mask, symbols)
        reference = reference_label_allowed_sets(token_mask, symbols)
        assert fast == reference
        assert len(fast) == len(protocol_of(token_mask, symbols)) - 1


def test_the_gold_token_always_sits_in_the_admissible_set(token_mask):
    for symbols in [words.split() for _, words in PAIRS]:
        protocol = protocol_of(token_mask, symbols)
        for position in range(1, len(protocol)):
            assert protocol[position] in token_mask.allowed_next_ids(protocol[:position])


def test_deterministic_positions_offer_exactly_one_choice(token_mask):
    symbols = PAIRS[3][1].split()
    sets = label_allowed_sets(token_mask, symbols)
    forced = [entry for entry in sets if entry in ((token_mask.space_id,), (token_mask.rule_end_id,), (token_mask.eos_id,))]
    assert forced and all(len(entry) == 1 for entry in forced)
    assert sets[0] == (token_mask.symbol_ids["("],)
    assert sets[-1] == (token_mask.eos_id,)
    assert len(sets) == 2 * len(symbols) + 1
    assert sum(1 for entry in sets if len(entry) == 1) >= len(symbols) + 1
    assert all(token_mask.rule_start_id not in entry for entry in sets)


def test_the_table_stores_every_set_once_and_marks_exactly_its_ids(token_mask):
    table = AllowedSetTable(token_mask)
    first = table.intern([7, 3, 3, 9])
    again = table.intern([9, 7, 3])
    other = table.intern([7, 3])
    assert first == again != other
    assert table.cardinality(first) == 3
    matrix = table.matrix()
    assert matrix.shape == (len(table.rows), token_mask.tokenizer.vocabulary_size)
    assert sorted(torch.nonzero(matrix[first]).flatten().tolist()) == [3, 7, 9]
    assert int(matrix.sum()) == sum(len(row) for row in table.rows)
    assert table.rows[table.rule_start_row] == (token_mask.rule_start_id,)
    with pytest.raises(TrainingError):
        table.intern([])


def test_describe_reports_the_spread_of_the_sets(token_mask):
    table = AllowedSetTable(token_mask)
    for symbols in [words.split() for _, words in PAIRS]:
        for entry in label_allowed_sets(token_mask, symbols):
            table.intern(entry)
    summary = table.describe()
    assert summary["distinct_admissible_sets"] == len(table.rows)
    assert summary["widest_set"] >= summary["narrowest_set"] == 1
    assert summary["vocabulary_size"] == token_mask.tokenizer.vocabulary_size


# ---------------------------------------------------------------------------
# Encoding and batching
# ---------------------------------------------------------------------------
def test_encoding_lines_up_labels_states_and_free_flags(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(), token_mask, table)
    assert len(encoded) == len(PAIRS)
    for item in encoded:
        assert len(item["labels"]) + 1 == len(item["target"])
        assert len(item["states"]) == len(item["labels"]) == len(item["free"])
        for position, label in enumerate(item["labels"]):
            assert label in table.rows[item["states"][position]]
        assert sum(item["free"]) == sum(1 for entry in label_allowed_sets(token_mask, item["symbols"]) if len(entry) > 1)
    assert 0.0 < table.describe()["single_choice_sets"]


def test_an_example_without_symbols_is_refused(token_mask):
    table = AllowedSetTable(token_mask)
    with pytest.raises(TrainingError):
        encode_examples([AlgebraExample("empty", "testo", [])], token_mask, table)


def test_batching_pads_with_a_loss_free_configuration(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(PAIRS[:3]), token_mask, table)
    batch = make_batch(encoded, token_mask, table, "cpu")
    height = batch.labels.shape[1]
    for row, item in enumerate(encoded):
        real = len(item["labels"])
        assert bool(batch.position_mask[row, :real].all())
        if real < height:
            assert not bool(batch.position_mask[row, real:].any())
            assert bool((batch.labels[row, real:] == token_mask.rule_start_id).all())
            assert bool((batch.states[row, real:] == table.rule_start_row).all())
            assert not bool(batch.free_mask[row, real:].any())
        assert bool((batch.input_ids[row] != token_mask.pad_id).any())
    assert batch.input_ids.shape[0] == 3


def test_a_batch_built_from_nothing_is_refused(token_mask):
    table = AllowedSetTable(token_mask)
    with pytest.raises(TrainingError):
        make_batch([], token_mask, table, "cpu")


def test_batches_cover_the_corpus_once_in_order(token_mask):
    encoded = list(range(10))
    chunks = batch_of(encoded, 3)
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]
    assert [value for chunk in chunks for value in chunk] == encoded


# ---------------------------------------------------------------------------
# Masked loss
# ---------------------------------------------------------------------------
def test_masked_logits_keep_only_the_admissible_entries():
    logits = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    states = torch.tensor([[0, 1, 0], [1, 1, 0]])
    matrix = torch.zeros(2, 8, dtype=torch.bool)
    matrix[0, 4] = True
    matrix[1, [0, 3, 7]] = True
    scored = masked_logits(logits, matrix, states)
    assert scored.shape == logits.shape
    for row in range(2):
        for column in range(3):
            keep = matrix[states[row, column]]
            assert bool((scored[row, column][keep] == logits[row, column][keep]).all())
            assert bool(torch.isneginf(scored[row, column][~keep]).all())
    assert float(logits[0, 1, 1]) == 9.0


def test_the_masked_cross_entropy_matches_a_manual_computation(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(PAIRS[:3]), token_mask, table)
    batch = make_batch(encoded, token_mask, table, "cpu")
    matrix = table.matrix()
    model = tiny_model(token_mask, seed=4).eval()
    with torch.no_grad():
        loss, metrics = score_batch(model, batch, matrix, table)
        logits = model(batch.input_ids, batch.input_mask, batch.decoder_input)
        scored = masked_logits(logits, matrix, batch.states)
        log_prob = F.log_softmax(scored, dim=-1)
        picked = log_prob.gather(-1, batch.labels.unsqueeze(-1)).squeeze(-1)
        weight = batch.position_mask.to(torch.float32)
        manual = float((-picked * weight).sum() / weight.sum())
    assert abs(float(loss) - manual) < 1e-5
    assert math.isfinite(metrics["loss"]) and metrics["positions"] == int(weight.sum())
    assert metrics["loss_sum"] == pytest.approx(metrics["loss"] * metrics["positions"], rel=1e-6)
    assert 0.0 <= metrics["free_ratio"] <= 1.0
    assert metrics["exact_sequences"] < metrics["sequences"]


def test_the_unmasked_argmax_is_counted_when_it_is_illegal(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(PAIRS[:2]), token_mask, table)
    batch = make_batch(encoded, token_mask, table, "cpu")
    matrix = table.matrix()
    counts = []
    for seed in (0, 1, 2, 3, 4):
        model = tiny_model(token_mask, seed=seed).eval()
        with torch.no_grad():
            _, metrics = score_batch(model, batch, matrix, table)
        counts.append(metrics["illegal_unmasked_argmax"])
    assert sum(counts) > 0


def test_aggregate_weights_batches_by_positions():
    merged = aggregate(
        [
            {"loss": 1.0, "loss_sum": 10.0, "positions": 10, "sequences": 2, "exact_sequences": 1,
             "free_positions": 4, "free_correct": 2, "illegal_unmasked_argmax": 3},
            {"loss": 0.2, "loss_sum": 4.0, "positions": 20, "sequences": 4, "exact_sequences": 2,
             "free_positions": 6, "free_correct": 3, "illegal_unmasked_argmax": 1},
        ]
    )
    assert merged["loss"] == pytest.approx(14.0 / 30.0)
    assert merged["perplexity"] == pytest.approx(math.exp(14.0 / 30.0))
    assert merged["exact_sequence_accuracy"] == pytest.approx(0.5)
    assert merged["free_position_accuracy"] == pytest.approx(0.5)
    assert merged["illegal_unmasked_argmax_rate"] == pytest.approx(4.0 / 30.0)
    assert aggregate([]) == {}


# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------
def test_weight_decay_skips_biases_and_normalization_gains(token_mask):
    model = tiny_model(token_mask)
    optimizer = build_optimizer(model, 5e-4, 0.01)
    assert len(optimizer.param_groups) == 2
    decayed, exempt = optimizer.param_groups
    assert decayed["weight_decay"] == 0.01 and exempt["weight_decay"] == 0.0
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    assert all(not names[id(p)].endswith(".bias") and "norm" not in names[id(p)] for p in decayed["params"])
    assert all(names[id(p)].endswith(".bias") or "norm" in names[id(p)] for p in exempt["params"])
    assert decayed["betas"] == (0.9, 0.98) and decayed["eps"] == pytest.approx(1e-9)
    assert len(decayed["params"]) + len(exempt["params"]) == len(list(model.parameters()))


def test_the_schedule_warms_up_then_decays_to_the_floor(token_mask):
    optimizer = build_optimizer(tiny_model(token_mask), 5e-4, 0.01)
    scheduler = cosine_schedule(optimizer, 10, 100, 0.1)
    multipliers = learning_rates(scheduler, 101)
    assert multipliers[0] == pytest.approx(0.1)
    assert multipliers[9] == pytest.approx(1.0)
    assert multipliers[10] == pytest.approx(1.0)
    assert all(multipliers[index] >= multipliers[index + 1] for index in range(10, 100))
    assert multipliers[99] < 0.15
    assert multipliers[100] == pytest.approx(0.1, abs=1e-3)


def test_a_schedule_shorter_than_its_warmup_is_repaired(token_mask):
    optimizer = build_optimizer(tiny_model(token_mask), 5e-4, 0.01)
    scheduler = cosine_schedule(optimizer, 50, 10)
    multipliers = learning_rates(scheduler, 3)
    assert multipliers[0] < multipliers[1] < multipliers[2] <= 1.0


# ---------------------------------------------------------------------------
# Training loop and constrained decoding
# ---------------------------------------------------------------------------
def test_the_loop_overfits_a_handful_of_pairs(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(), token_mask, table)
    matrix = table.matrix()
    model = tiny_model(token_mask, seed=8)
    optimizer = build_optimizer(model, 1e-3, 0.01)
    scheduler = cosine_schedule(optimizer, 20, 390, 0.1)
    metrics = []
    for epoch in range(130):
        order = list(range(len(encoded)))
        random.Random(epoch).shuffle(order)
        shuffled = [encoded[index] for index in order]
        for chunk in batch_of(shuffled, 4):
            batch = make_batch(chunk, token_mask, table, "cpu")
            metrics.append(training_step(model, batch, matrix, table, optimizer, scheduler, 1.0, None))
    summary = aggregate(metrics[-3:])
    assert summary["loss"] < 0.05
    assert summary["exact_sequence_accuracy"] == 1.0

    decoded = generate_batch(model, token_mask, encoded, max_steps=max(len(item["target"]) for item in encoded), mode="greedy", seed=5)
    assert len(decoded) == len(encoded)
    assert [entry.symbols for entry in decoded] == [item["symbols"] for item in encoded]
    assert all(entry.stopped_at_end for entry in decoded)


def test_decoding_stays_inside_the_legal_language_even_before_training(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(PAIRS[:6]), token_mask, table)
    model = tiny_model(token_mask, seed=2)
    for mode, temperature in (("greedy", 1.0), ("sample", 0.7)):
        decoded = generate_batch(model, token_mask, encoded, max_steps=70, mode=mode, temperature=temperature, seed=6)
        assert len(decoded) == 6
        for entry in decoded:
            assert entry.symbols, "an untrained model may still emit an empty stream"
            token_mask.mask.replay(entry.symbols)
            if entry.stopped_at_end:
                token_mask.mask.validate_stream(entry.symbols)
        assert any(entry.stopped_at_end for entry in decoded)


def test_generation_rejects_bad_requests(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(PAIRS[:2]), token_mask, table)
    model = tiny_model(token_mask, seed=1)
    with pytest.raises(TrainingError):
        generate_batch(model, token_mask, [], max_steps=10)
    with pytest.raises(TrainingError):
        generate_batch(model, token_mask, encoded, max_steps=10, mode="beam")
    with pytest.raises(TrainingError):
        generate_batch(model, token_mask, encoded, max_steps=10, mode="sample", temperature=0.0)
    with pytest.raises(TrainingError):
        generate_batch(model, token_mask, encoded, max_steps=96)


def test_an_illegal_stream_is_reported_by_the_mask_and_not_by_the_model(token_mask):
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples_of(PAIRS[:2]), token_mask, table)
    model = tiny_model(token_mask, seed=2)
    decoded = generate_batch(model, token_mask, encoded, max_steps=40, mode="greedy", seed=9)
    broken = list(decoded[0].symbols)
    for index, symbol in enumerate(broken):
        if symbol in ("(", ")"):
            broken[index] = "[" if symbol == "(" else "]"
            break
    with pytest.raises((AlgebraMaskError, ValueError)):
        token_mask.mask.validate_stream(broken)
