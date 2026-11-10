"""Byte-Pair Encoding tokenizer learned from the project corpus, from scratch.

The system in this repository is trained with no pretrained weights and no
third-party vocabulary, so the tokenizer has to be learned from the corpus too.
Two properties are what make the symbolic layer work on top of it:

* the structural symbols of the bracket algebra stay atomic. ``( ) [ ]``, every
  operator name of ``datasets/multilingual/templates/operators.json`` and every canonical leaf
  identifier are one single token, and they never take part in a merge, so the
  constrained decoder in :mod:`neuro_symbolic.masking` can reason about them one
  token at a time;
* the pre-tokenization is reversible. A string is split into word units,
  whitespace runs and single punctuation marks, and merges only ever happen
  inside one unit, so ``decode(encode(text)) == text`` holds for every string the
  corpus contains (the test suite checks it on the whole dataset).

Size note. The corpus is closed and small: about 1.9k distinct word forms, 17
accented letters and 1.96M tokens, and every word form occurs at least twice.
The merge process therefore saturates far below the 8k-32k vocabulary a web
scale model needs. :meth:`Tokenizer.train` records where it stopped, and
``datasets/multilingual/tools/train_tokenizer.py`` reports the saturation point instead of
padding the vocabulary with tokens that never occur.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Special tokens of the protocol. <RULE_START> and <RULE_END> delimit the region
# that holds one rule of the bracket algebra: the decoder is allowed to emit the
# closing marker exactly when the symbolic mask says the tree is complete.
PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
RULE_START = "<RULE_START>"
RULE_END = "<RULE_END>"

SPECIAL_TOKENS: Tuple[str, ...] = (PAD, UNK, BOS, EOS, RULE_START, RULE_END)

# The four terminals of the Dyck-2 algebra. They are always atomic.
ALGEBRA_BRACKETS: Tuple[str, ...] = ("(", ")", "[", "]")

FORMAT_NAME = "lingologos-bpe-1"

# Word units (unicode letters, digits and underscore), whitespace runs and every
# other single character. This split is loss free: joining the units back gives
# the original string.
_UNIT_PATTERN = re.compile(r"\w+|\s+|[^\w\s]")

# Whitespace carries the canonical spacing of the algebra and digits carry leaf
# values, so these characters are always in the vocabulary: an unseen whitespace
# run or number can still be spelled one character at a time instead of turning
# into <UNK> and breaking reversibility.
_BASE_CHARACTERS: Tuple[str, ...] = (
    " ",
    "\n",
    "\r",
    "\t",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)


def pre_tokenize(text: str) -> List[str]:
    """Split a string into word units, whitespace runs and single symbols."""
    return _UNIT_PATTERN.findall(text)


def is_whitespace_unit(unit: str) -> bool:
    """Whitespace is atomic: it carries the canonical spacing of the algebra."""
    return unit.isspace()


def is_numeric_unit(unit: str) -> bool:
    """Digit runs are atomic: leaf values and clock times must stay readable."""
    return unit.isdigit()


def is_atomic_unit(unit: str, atomic_symbols: Iterable[str]) -> bool:
    """True when a unit must stay one single token and never be merged."""
    return unit in atomic_symbols or is_whitespace_unit(unit) or is_numeric_unit(unit)


class TokenizerError(Exception):
    """Raised when a token operation cannot be performed."""


class Tokenizer:
    """Reversible BPE tokenizer with an atomic structural alphabet."""

    def __init__(
        self,
        vocabulary: Sequence[str],
        merges: Sequence[Tuple[str, str]],
        atomic_symbols: Iterable[str],
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vocabulary: List[str] = list(vocabulary)
        self.id_of: Dict[str, int] = {token: index for index, token in enumerate(self.vocabulary)}
        self.merges: List[Tuple[str, str]] = [tuple(pair) for pair in merges]  # type: ignore[assignment]
        self.merge_rank: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self.atomic_symbols: Set[str] = set(atomic_symbols) | set(ALGEBRA_BRACKETS)
        self.special_tokens: Tuple[str, ...] = tuple(special_tokens)
        self.statistics: Dict[str, Any] = dict(statistics or {})
        missing = [token for token in self.special_tokens if token not in self.id_of]
        if missing:
            raise TokenizerError("special tokens missing from the vocabulary: " + ", ".join(missing))
        for bracket in ALGEBRA_BRACKETS:
            if bracket not in self.id_of:
                raise TokenizerError("structural bracket missing from the vocabulary: " + bracket)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 6000,
        min_frequency: int = 2,
        atomic_symbols: Iterable[str] = (),
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "Tokenizer":
        """Learn BPE merges from the corpus; no external vocabulary is read.

        Merges are chosen by descending frequency (ties broken by the pair
        itself) so the result is deterministic for a fixed input order. The loop
        stops at ``vocab_size``, or earlier when no candidate pair reaches
        ``min_frequency``, which is what happens on a closed corpus.
        """
        if vocab_size < 1:
            raise TokenizerError("vocab_size must be positive")
        declared_atomic = set(atomic_symbols) | set(ALGEBRA_BRACKETS)
        type_counts: Counter = Counter()
        atomic_counts: Counter = Counter()
        unit_total = 0
        text_total = 0
        for text in texts:
            text_total += 1
            for unit in pre_tokenize(text):
                unit_total += 1
                if is_atomic_unit(unit, declared_atomic):
                    atomic_counts[unit] += 1
                else:
                    type_counts[unit] += 1

        symbols: Dict[str, List[str]] = {word: list(word) for word in type_counts}
        base_chars = sorted({char for word in symbols for char in word})

        # Every atomic unit the corpus actually uses becomes one token: whitespace
        # runs carry the canonical spacing of the algebra, digit runs carry leaf
        # values, and the declared algebra symbols are the alphabet that the
        # decoding mask reasons about one token at a time.
        atomic_tokens = sorted(set(declared_atomic) | set(atomic_counts))
        vocabulary: List[str] = []
        for token in list(special_tokens) + atomic_tokens + list(_BASE_CHARACTERS) + base_chars:
            if token not in vocabulary:
                vocabulary.append(token)
        vocab_set: Set[str] = set(vocabulary)

        merges: List[Tuple[str, str]] = []
        vocab_after_merge: List[int] = []
        stop_reason = "vocabulary target reached"
        while len(vocabulary) < vocab_size:
            pairs: Counter = Counter()
            for word, weight in type_counts.items():
                sequence = symbols[word]
                for left, right in zip(sequence, sequence[1:]):
                    pairs[(left, right)] += weight
            if not pairs:
                stop_reason = "no mergeable pair left"
                break
            best_pair, best_count = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[0]
            if best_count < min_frequency:
                stop_reason = "best pair below min_frequency"
                break
            left, right = best_pair
            merged = left + right
            if merged not in vocab_set:
                vocab_set.add(merged)
                vocabulary.append(merged)
                vocab_after_merge.append(len(vocabulary))
            merges.append((left, right))
            for word in symbols:
                sequence = symbols[word]
                if not any((a, b) == best_pair for a, b in zip(sequence, sequence[1:])):
                    continue
                rebuilt: List[str] = []
                index = 0
                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and sequence[index] == left
                        and sequence[index + 1] == right
                    ):
                        rebuilt.append(merged)
                        index += 2
                    else:
                        rebuilt.append(sequence[index])
                        index += 1
                symbols[word] = rebuilt

        statistics = {
            "format": FORMAT_NAME,
            "texts": text_total,
            "units": unit_total,
            "word_types": len(type_counts),
            "atomic_units": sum(atomic_counts.values()),
            "atomic_types": len(atomic_counts),
            "merges_learned": len(merges),
            "vocabulary_size": len(vocabulary),
            "vocab_target": vocab_size,
            "min_frequency": min_frequency,
            "stop_reason": stop_reason,
            "saturated_at": vocab_after_merge[-1] if vocab_after_merge else len(vocabulary),
        }
        return cls(vocabulary, merges, declared_atomic, special_tokens, statistics)

    # ------------------------------------------------------------------
    # Encoding and decoding
    # ------------------------------------------------------------------
    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def pieces_for_unit(self, unit: str) -> List[str]:
        """Apply the learned merges to one unit, always in merge-rank order."""
        sequence = list(unit)
        while len(sequence) > 1:
            best_index: Optional[int] = None
            best_rank: Optional[int] = None
            for index in range(len(sequence) - 1):
                rank = self.merge_rank.get((sequence[index], sequence[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_index, best_rank = index, rank
            if best_index is None:
                break
            sequence[best_index : best_index + 2] = [sequence[best_index] + sequence[best_index + 1]]
        return sequence

    def encode_unit(self, unit: str) -> List[int]:
        """Encode one pre-tokenization unit; unknown pieces become <UNK>.

        An atomic unit that is not in the vocabulary (an unseen whitespace run or
        number) is spelled with its characters, all of which are always in the
        vocabulary, so reversibility never depends on having seen that exact run.
        """
        unk = self.id_of[UNK]
        if is_atomic_unit(unit, self.atomic_symbols):
            if unit in self.id_of:
                return [self.id_of[unit]]
            return [self.id_of.get(char, unk) for char in unit]
        return [self.id_of.get(piece, unk) for piece in self.pieces_for_unit(unit)]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode a string into token ids."""
        ids: List[int] = []
        for unit in pre_tokenize(text):
            ids.extend(self.encode_unit(unit))
        if add_bos:
            ids.insert(0, self.id_of[BOS])
        if add_eos:
            ids.append(self.id_of[EOS])
        return ids

    def encode_algebra(self, symbols: Sequence[str]) -> List[int]:
        """Encode a bare stream of algebra symbols, one id per symbol.

        Raises :class:`TokenizerError` when a symbol is not one atomic token, so
        a decoding mask can never be handed a split structural symbol.
        """
        ids: List[int] = []
        for symbol in symbols:
            if not is_atomic_unit(symbol, self.atomic_symbols):
                raise TokenizerError("symbol is not atomic in this vocabulary: " + repr(symbol))
            if symbol not in self.id_of:
                raise TokenizerError("symbol not in the vocabulary: " + repr(symbol))
            ids.append(self.id_of[symbol])
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Concatenate token strings; the special markers are dropped by default."""
        parts: List[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.vocabulary):
                raise TokenizerError("token id out of range: " + str(token_id))
            token = self.vocabulary[token_id]
            if skip_special_tokens and token in self.special_tokens:
                continue
            parts.append(token)
        return "".join(parts)

    def decode_symbol_stream(self, ids: Sequence[int]) -> List[str]:
        """Decode ids back into the algebra symbol stream (spaces and markers out)."""
        symbols: List[str] = []
        for token_id in ids:
            token = self.token_of(token_id)
            if token in self.special_tokens or token == " ":
                continue
            symbols.append(token)
        return symbols

    # ------------------------------------------------------------------
    # Structural helpers used by the constrained decoder
    # ------------------------------------------------------------------
    def token_id(self, token: str) -> int:
        if token not in self.id_of:
            raise TokenizerError("token not in the vocabulary: " + repr(token))
        return self.id_of[token]

    def token_of(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.vocabulary):
            raise TokenizerError("token id out of range: " + str(token_id))
        return self.vocabulary[token_id]

    def is_special(self, token_id: int) -> bool:
        return self.token_of(token_id) in self.special_tokens

    # ------------------------------------------------------------------
    # Quality reports
    # ------------------------------------------------------------------
    def unknown_tokens(self, texts: Iterable[str]) -> Tuple[int, int]:
        """Count produced <UNK> tokens and all produced tokens."""
        unk = self.id_of[UNK]
        unknown = 0
        total = 0
        for text in texts:
            for token_id in self.encode(text):
                total += 1
                if token_id == unk:
                    unknown += 1
        return unknown, total

    def round_trip_failures(self, texts: Iterable[str], limit: int = 3) -> Tuple[int, List[str]]:
        """Count strings whose encoding does not decode back byte for byte."""
        failures = 0
        samples: List[str] = []
        for text in texts:
            if self.decode(self.encode(text)) != text:
                failures += 1
                if len(samples) < limit:
                    samples.append(text[:120])
        return failures, samples

    def encodable_symbols(self, symbols: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Split a symbol inventory into single-token and not-single-token lists."""
        ok: List[str] = []
        bad: List[str] = []
        for symbol in symbols:
            if is_atomic_unit(symbol, self.atomic_symbols) and symbol in self.id_of:
                ok.append(symbol)
            else:
                bad.append(symbol)
        return ok, bad

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Any) -> None:
        """Write the vocabulary and the merge table as readable UTF-8 JSON."""
        payload = {
            "format": FORMAT_NAME,
            "special_tokens": list(self.special_tokens),
            "atomic_symbols": sorted(self.atomic_symbols),
            "vocabulary": self.vocabulary,
            "merges": [list(pair) for pair in self.merges],
            "statistics": self.statistics,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> "Tokenizer":
        source = Path(path)
        if not source.exists():
            raise TokenizerError("tokenizer file not found: " + str(source))
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT_NAME:
            raise TokenizerError("unsupported tokenizer format: " + str(payload.get("format")))
        return cls(
            payload["vocabulary"],
            [tuple(pair) for pair in payload["merges"]],
            payload["atomic_symbols"],
            payload["special_tokens"],
            payload.get("statistics"),
        )
