"""Tests for the tokenizer learned from scratch: reversibility and the atomic algebra alphabet."""

import pytest

from neuro_symbolic.tokenizer import (
    ALGEBRA_BRACKETS,
    BOS,
    EOS,
    PAD,
    RULE_END,
    RULE_START,
    UNK,
    Tokenizer,
    TokenizerError,
    pre_tokenize,
)

OPERATORS = ("IF", "AT", "SCHEDULE", "WEEKDAY_ORD")
LEAF_TOKENS = ("flag", "clock", "day", "ord", "x", "07", "00", "1", "3", "mon", "month")

CORPUS_TEXTS = [
    "You are LingoLOGOS. You map an everyday recurrence instruction to its formal logic.",
    "Italian: manda un promemoria ogni lunedi mattina alle 07:00 perche il fuso e critico.",
    "German: Plane das Projektreview jeden 3. Montag des Monats um 07:00.",
    "Portuguese: envie um lembrete as 13:00 de cada dia util.",
    "( IF [ flag 1 ] [ day mon ] ( AT [ clock 07 00 ] ( SCHEDULE [ project_review ] ) ) )",
    "( WEEKDAY_ORD [ ord 3 ] [ month 11 ] ( AT [ clock 09 30 ] ( SCHEDULE [ review ] ) ) )",
    '{"bracket_ast": "( AT [ clock 07 00 ] ( SCHEDULE [ note ] ) )", "gf_abstract": "atClock 07 00 note"}',
    "multi\nline\ttext with 42 numbers and 3.5 decimals",
]


def atomic_symbols():
    return set(ALGEBRA_BRACKETS) | set(OPERATORS) | {token for token in LEAF_TOKENS if not token.isdigit()}


def build_tokenizer(vocab_size: int = 4000):
    return Tokenizer.train(CORPUS_TEXTS, vocab_size=vocab_size, min_frequency=2, atomic_symbols=atomic_symbols())


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


def test_train_is_deterministic():
    first = build_tokenizer()
    second = build_tokenizer()
    assert first.merges == second.merges
    assert first.vocabulary == second.vocabulary
    assert first.atomic_symbols == second.atomic_symbols


def test_pre_tokenize_is_lossless():
    for text in CORPUS_TEXTS:
        units = pre_tokenize(text)
        assert "".join(units) == text
        assert all(units)


def test_no_merge_touches_an_atomic_symbol(tokenizer):
    for left, right in tokenizer.merges:
        assert left not in tokenizer.atomic_symbols
        assert right not in tokenizer.atomic_symbols
        assert not left.isdigit()
        assert not left.isspace()


def test_round_trip_is_byte_exact_on_the_whole_corpus(tokenizer):
    for text in CORPUS_TEXTS:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    failures, samples = tokenizer.round_trip_failures(CORPUS_TEXTS)
    assert failures == 0
    assert samples == []


def test_no_unknown_token_on_the_whole_corpus(tokenizer):
    unknown, total = tokenizer.unknown_tokens(CORPUS_TEXTS)
    assert total > 200
    assert unknown == 0


def test_unseen_whitespace_and_number_do_not_become_unknown(tokenizer):
    # Whitespace runs and digit runs never entered the merge process, yet the
    # single characters are always in the vocabulary.
    unseen = "trailing tabs\t\tand a number 998877\n\nnext line"
    unknown, total = tokenizer.unknown_tokens([unseen])
    assert unknown == 0
    assert tokenizer.decode(tokenizer.encode(unseen)) == unseen


def test_structural_brackets_are_single_tokens(tokenizer):
    for bracket in ALGEBRA_BRACKETS:
        assert tokenizer.encode_algebra([bracket]) == [tokenizer.token_id(bracket)]


def test_algebra_stream_is_one_token_per_symbol(tokenizer):
    stream = "( WEEKDAY_ORD [ ord 3 ] [ month 11 ] ( AT [ clock 09 30 ] ( SCHEDULE [ x ] ) ) )".split()
    ids = tokenizer.encode_algebra(stream)
    assert len(ids) == len(stream)
    assert tokenizer.decode_symbol_stream(ids) == stream


def test_encode_algebra_refuses_a_split_symbol(tokenizer):
    with pytest.raises(TokenizerError):
        tokenizer.encode_algebra(["SCHEDULEX"])
    with pytest.raises(TokenizerError):
        tokenizer.encode_algebra(["review"])


def test_special_protocol_tokens_exist(tokenizer):
    for token in (PAD, UNK, BOS, EOS, RULE_START, RULE_END):
        token_id = tokenizer.token_id(token)
        assert tokenizer.token_of(token_id) == token
        assert tokenizer.is_special(token_id)
    for token in ("(", "IF", "flag"):
        assert not tokenizer.is_special(tokenizer.token_id(token))


def test_out_of_range_ids_are_refused(tokenizer):
    with pytest.raises(TokenizerError):
        tokenizer.token_of(tokenizer.vocabulary_size)
    with pytest.raises(TokenizerError):
        tokenizer.token_of(-1)
    with pytest.raises(TokenizerError):
        tokenizer.decode([99999])


def test_vocab_target_is_respected(tokenizer):
    target = tokenizer.vocabulary_size - 5
    assert target > 80
    truncated = Tokenizer.train(CORPUS_TEXTS, vocab_size=target, min_frequency=2, atomic_symbols=atomic_symbols())
    assert truncated.vocabulary_size == target
    assert truncated.statistics["stop_reason"] == "vocabulary target reached"
    # The base inventory is unconditional: with a target below it no merge is
    # learned, and the merge table stays empty instead of splitting the alphabet.
    tiny = Tokenizer.train(CORPUS_TEXTS, vocab_size=1, min_frequency=2, atomic_symbols=atomic_symbols())
    assert tiny.merges == []
    assert tiny.vocabulary_size < tokenizer.vocabulary_size


def test_encodable_symbols_split_the_inventory(tokenizer):
    ok, bad = tokenizer.encodable_symbols(["(", "[", "IF", "SCHEDULE", "flag", "review", "nope"])
    assert ok == ["(", "[", "IF", "SCHEDULE", "flag"]
    assert bad == ["review", "nope"]


def test_save_and_load_keep_the_tokenizer_identical(tokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.vocabulary == tokenizer.vocabulary
    assert loaded.merges == tokenizer.merges
    assert loaded.atomic_symbols == tokenizer.atomic_symbols
    for text in CORPUS_TEXTS:
        assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.statistics["merges_learned"] == tokenizer.statistics["merges_learned"]


def test_load_rejects_a_foreign_format(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "other-tokenizer-0", "vocabulary": []}', encoding="utf-8")
    with pytest.raises(TokenizerError):
        Tokenizer.load(path)


def test_vocabulary_requires_the_protocol_tokens():
    with pytest.raises(TokenizerError):
        Tokenizer(["a", "(", ")", "[", "]"], [], ())
    with pytest.raises(TokenizerError):
        Tokenizer([PAD, UNK, BOS, EOS, RULE_START, RULE_END, "(", ")", "["], [], ())
