"""Tests for the symbolic decoding mask: the legal language is exactly the algebra, nothing more."""

import random

import pytest
import torch

from neuro_symbolic.masking import (
    AlgebraMask,
    AlgebraMaskError,
    TokenStreamMask,
    algebra_word,
    parse_algebra,
    skeleton_of_stream,
    stream_of_tree,
    tokenize_algebra,
)
from neuro_symbolic.tokenizer import ALGEBRA_BRACKETS, Tokenizer, TokenizerError

ARITIES = {"IF": 2, "AT": 1, "SCHEDULE": 1, "WEEKDAY_ORD": 2, "EVERY": 1}
LEAF_PAYLOADS = [
    ["flag", "1"],
    ["day", "mon"],
    ["clock", "07", "00"],
    ["ord", "3"],
    ["month"],
    ["x"],
]

VALID = [
    "( AT [ clock 07 00 ] )",
    "( IF [ flag 1 ] [ day mon ] )",
    "( WEEKDAY_ORD [ ord 3 ] ( EVERY ( IF [ flag 1 ] [ day mon ] ) ) )",
]

INVALID = {
    "missing argument": "( IF [ flag 1 ] )",
    "extra argument": "( IF [ flag 1 ] [ day mon ] [ x ] )",
    "round closed by a square bracket": "( IF [ flag 1 ] ]",
    "square closed by a round bracket": "( AT ]",
    "leaf opened where an operator name belongs": "( [ x ] )",
    "leaf too short for its tag": "( AT [ clock 07 ] )",
    "leaf too long for its tag": "( AT [ month 07 ] )",
    "leaf longer than the algebra allows": "( AT [ clock 07 00 00 ] )",
    "unknown operator": "( NOPE [ x ] )",
    "unknown leaf tag": "( AT [ zzz 07 00 ] )",
    "payload without an open leaf": "( AT x )",
    "truncated tree": "( AT [ clock 07 00",
    "second root node": "( AT [ x ] ) ( SCHEDULE [ x ] )",
    "leaf holding a node": "[ [ x ] ]",
}


@pytest.fixture(scope="module")
def mask():
    return AlgebraMask.from_leaf_payloads(ARITIES, LEAF_PAYLOADS)


def is_dyk2(word: str) -> bool:
    """Independent Dyck-2 check, written without the mask under test."""
    partner = {")": "(", "]": "["}
    stack = []
    for char in word:
        if char in ("(", "["):
            stack.append(char)
        elif char in partner:
            if not stack or stack[-1] != partner[char]:
                return False
            stack.pop()
        else:
            return False
    return not stack


def accepted(mask: AlgebraMask, stream) -> bool:
    try:
        mask.validate_stream(stream)
    except (AlgebraMaskError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# The tree reader and writer
# ---------------------------------------------------------------------------
def test_reader_and_writer_are_inverse():
    for text in VALID:
        tree = parse_algebra(text)
        assert algebra_word(tree) == text
        assert stream_of_tree(tree) == text.split()


def test_reader_tolerates_missing_spaces():
    tight = parse_algebra("(IF[flag 1][day mon])")
    assert algebra_word(tight) == "( IF [ flag 1 ] [ day mon ] )"
    assert tokenize_algebra("( WEEKDAY_ORD [ ord 3 ] )") == ["(", "WEEKDAY_ORD", "[", "ord", "3", "]", ")"]


@pytest.mark.parametrize("text", ["", "just words", "[ ]", "[ [ x ] ]", "( A ) )", "( )"])
def test_reader_rejects_malformed_text(text):
    with pytest.raises(AlgebraMaskError):
        parse_algebra(text)


# ---------------------------------------------------------------------------
# The automaton
# ---------------------------------------------------------------------------
def test_every_valid_tree_is_accepted(mask):
    for text in VALID:
        tree = mask.validate_stream(text.split())
        assert algebra_word(tree) == text


@pytest.mark.parametrize("reason", sorted(INVALID))
def test_every_invalid_tree_is_refused(mask, reason):
    assert not accepted(mask, INVALID[reason].split())


def test_the_operator_name_decides_the_brackets(mask):
    # After the name the number of children is known, so so is every closing bracket.
    state = mask.replay(["(", "IF"])
    assert mask.allowed_symbols(state) == {"[", "("}
    state = mask.replay(["(", "IF", "[", "flag", "1", "]"])
    assert mask.allowed_symbols(state) == {"[", "("}
    state = mask.replay(["(", "IF", "[", "flag", "1", "]", "[", "day", "mon", "]"])
    assert mask.allowed_symbols(state) == {")"}


def test_root_is_always_an_operator_node(mask):
    state = mask.initial_state()
    assert mask.allowed_symbols(state) == {"("}
    assert not accepted(mask, ["[", "x", "]"])
    assert not mask.is_complete(state)
    complete = mask.replay(["(", "SCHEDULE", "[", "x", "]", ")"])
    assert mask.is_complete(complete)
    assert mask.allowed_symbols(complete) == set()


def test_a_single_leaf_shape_is_enforced(mask):
    # clock is only ever seen with three payload tokens, month with one.
    assert accepted(mask, "( AT [ clock 07 00 ] )".split())
    assert not accepted(mask, "( AT [ clock 07 ] )".split())
    assert not accepted(mask, "( AT [ clock 07 00 00 ] )".split())
    assert accepted(mask, "( AT [ month ] )".split())
    assert not accepted(mask, "( AT [ month 07 ] )".split())


def test_validate_reports_where_the_stream_stops(mask):
    with pytest.raises(AlgebraMaskError) as info:
        mask.validate_stream(["(", "AT", "[", "clock", "07", "00", "]"])
    assert "still open" in str(info.value)


def test_step_error_names_the_symbol(mask):
    with pytest.raises(AlgebraMaskError) as info:
        mask.step(mask.replay(["(", "AT"]), "]")
    assert "']'" in str(info.value)


def test_constrained_sampling_only_produces_legal_trees(mask):
    rng = random.Random(11)
    for _ in range(200):
        stream = mask.sample_stream(rng, 5)
        tree = mask.validate_stream(stream)
        assert is_dyk2(skeleton_of_stream(stream))
        assert algebra_word(tree) == " ".join(stream)
        # Every proper prefix of a legal stream still has a legal continuation.
        for index in range(len(stream)):
            state = mask.replay(stream[:index])
            assert not mask.is_complete(state)
            assert mask.allowed_symbols(state)


def test_sampling_is_reproducible_for_a_seed(mask):
    first = AlgebraMask.from_leaf_payloads(ARITIES, LEAF_PAYLOADS)
    stream_a = first.sample_stream(random.Random(4), 4)
    stream_b = mask.sample_stream(random.Random(4), 4)
    assert stream_a == stream_b


def test_the_refinement_narrows_the_allowed_children(mask):
    refined = AlgebraMask(
        ARITIES,
        mask.leaf_forms,
        mask.leaf_value_tokens,
        {("IF", 0): {"["}},
    )
    text = "( IF ( EVERY ( SCHEDULE [ x ] ) ) [ day mon ] )"
    assert accepted(mask, text.split())
    assert not accepted(refined, text.split())
    assert accepted(refined, "( IF [ flag 1 ] [ day mon ] )".split())


def test_the_alphabet_of_the_mask_is_reported(mask):
    summary = mask.describe()
    assert summary["operators"] == len(ARITIES)
    assert summary["leaf_first_tokens"] == len(LEAF_PAYLOADS)
    assert summary["max_leaf_length"] == 3


def test_payload_longer_than_the_alphabet_is_refused():
    with pytest.raises(AlgebraMaskError):
        AlgebraMask.from_leaf_payloads(ARITIES, [["clock", "07", "00", "00"]])
    with pytest.raises(AlgebraMaskError):
        AlgebraMask.from_leaf_payloads({}, LEAF_PAYLOADS)
    with pytest.raises(AlgebraMaskError):
        AlgebraMask.from_leaf_payloads(ARITIES, [])


# ---------------------------------------------------------------------------
# Token level view
# ---------------------------------------------------------------------------
ALGEBRA_TOKENS = sorted(
    {str(token) for payload in LEAF_PAYLOADS for token in payload} | set(ARITIES) | set(ALGEBRA_BRACKETS)
)
TEXTS = [
    "manda un promemoria ogni lunedi alle 07:00",
    "( IF [ flag 1 ] [ day mon ] ) ( SCHEDULE [ month ] )",
    "( AT [ clock 07 00 ] )",
    "( WEEKDAY_ORD [ ord 3 ] ( EVERY ( IF [ flag 1 ] [ day mon ] ) ) )",
]


@pytest.fixture(scope="module")
def token_mask(mask):
    tokenizer = Tokenizer.train(TEXTS, vocab_size=4000, min_frequency=2, atomic_symbols=ALGEBRA_TOKENS)
    return TokenStreamMask(mask, tokenizer)


def test_gold_target_is_accepted_token_by_token(token_mask):
    for text in VALID:
        ids = token_mask.encode_target(text.split())
        for index in range(len(ids) + 1):
            allowed = token_mask.allowed_next_ids(ids[:index])
            if index < len(ids):
                assert ids[index] in allowed
            assert allowed


def test_start_and_end_markers_are_forced(token_mask):
    assert token_mask.allowed_next_ids([]) == {token_mask.rule_start_id}
    for text in VALID:
        ids = token_mask.encode_target(text.split())
        end_position = len(ids) - 2
        for index in range(1, end_position):
            assert token_mask.rule_end_id not in token_mask.allowed_next_ids(ids[:index])
        assert token_mask.allowed_next_ids(ids[:end_position]) == {token_mask.rule_end_id}
        assert token_mask.allowed_next_ids(ids[: end_position + 1]) == {token_mask.eos_id}
        assert token_mask.allowed_next_ids(ids) == {token_mask.pad_id}


def test_the_canonical_single_space_is_the_only_separator(token_mask):
    ids = token_mask.encode_target("( IF [ flag 1 ] [ day mon ] )".split())
    middle = ids.index(token_mask.tokenizer.token_id("flag"))
    assert token_mask.allowed_next_ids(ids[: middle + 1]) == {token_mask.space_id}
    after_space = ids[: middle + 2]
    assert token_mask.space_id not in token_mask.allowed_next_ids(after_space)
    assert token_mask.tokenizer.token_id("1") in token_mask.allowed_next_ids(after_space)


def test_mask_logits_removes_every_illegal_choice(token_mask):
    vocabulary_size = token_mask.tokenizer.vocabulary_size
    logits = torch.arange(vocabulary_size, dtype=torch.float32)
    ids = token_mask.encode_target("( AT [ clock 07 00 ]".split())
    masked = token_mask.mask_logits(logits, ids)
    allowed = token_mask.allowed_next_ids(ids)
    legal = torch.isfinite(masked)
    assert int(legal.sum()) == len(allowed)
    for token_id in range(vocabulary_size):
        if token_id in allowed:
            assert masked[token_id] == logits[token_id]
        else:
            assert masked[token_id] == float("-inf")
    probabilities = torch.softmax(masked, dim=0)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert float(probabilities[[i for i in range(vocabulary_size) if i not in allowed]].sum()) == 0.0


def test_mask_logits_keeps_the_device_and_refuses_a_narrower_vector(token_mask):
    vocabulary_size = token_mask.tokenizer.vocabulary_size
    with pytest.raises(ValueError):
        token_mask.mask_logits(torch.zeros(vocabulary_size - 1), [])
    with pytest.raises(ValueError):
        token_mask.mask_logits(torch.zeros(1, vocabulary_size), [])


def test_token_mask_needs_the_whole_alphabet(mask):
    tokenizer = Tokenizer.train(TEXTS, vocab_size=4000, min_frequency=2, atomic_symbols=ALGEBRA_TOKENS)
    widened = AlgebraMask(dict(ARITIES, EXTRA_OP=1), mask.leaf_forms, mask.leaf_value_tokens)
    with pytest.raises(TokenizerError):
        TokenStreamMask(widened, tokenizer)


def test_the_space_token_must_be_atomic(token_mask):
    with pytest.raises(TokenizerError):
        TokenStreamMask(token_mask.mask, token_mask.tokenizer, space_token="_")
