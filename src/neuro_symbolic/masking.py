"""Constrained decoding over the bracket algebra: what may come next, token by token.

The neural side proposes a distribution over the whole vocabulary; the symbolic
side says which of those tokens are legal at the current position of the rule
being written. Everything outside that set gets a score of minus infinity, so a
structurally impossible continuation cannot be sampled at all. This is the piece
that removes structural hallucination, and it replaces the post-hoc filtering the
engine used to do (parse the finished word, then reject it).

Two levels:

* :class:`AlgebraMask` works on the **symbol stream** of the algebra, the closed
  alphabet of ``( ) [ ]``, operator names and leaf payload tokens. It is a
  pushdown automaton over viable prefixes: the operator registry fixes how many
  children a node still needs, and the observed leaf shapes fix how many payload
  tokens a square node may hold. It needs no model and no tensor.
* :class:`TokenStreamMask` lifts the same automaton to **token ids** of the
  learned tokenizer, including the canonical single-space rendering of the
  algebra and the ``<RULE_START>`` / ``<RULE_END>`` protocol markers, and applies
  the mask to a logits vector.

The serialization convention comes from ``dataset/tools/logic_map.py``: the stored
bracket word of a tree is exactly ``" ".join(stream_of_tree(node))``.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from .tokenizer import (
    EOS,
    PAD,
    RULE_END,
    RULE_START,
    Tokenizer,
    TokenizerError,
    is_atomic_unit,
)

ROUND_OPEN = "("
ROUND_CLOSE = ")"
SQUARE_OPEN = "["
SQUARE_CLOSE = "]"

BRACKETS: Tuple[str, ...] = (ROUND_OPEN, ROUND_CLOSE, SQUARE_OPEN, SQUARE_CLOSE)
KIND_OPENERS: Tuple[str, ...] = (ROUND_OPEN, SQUARE_OPEN)

MAX_LEAF_LENGTH = 3


class AlgebraMaskError(Exception):
    """Raised when a symbol sequence is not a viable prefix of the algebra."""


# ---------------------------------------------------------------------------
# Model-free reading and writing of the algebra
# ---------------------------------------------------------------------------
def tokenize_algebra(text: str) -> List[str]:
    """Split algebra text into symbols, tolerant about spacing."""
    symbols: List[str] = []
    buffer: List[str] = []
    for char in text:
        if char in BRACKETS or char.isspace():
            if buffer:
                symbols.append("".join(buffer))
                buffer = []
            if char in BRACKETS:
                symbols.append(char)
        else:
            buffer.append(char)
    if buffer:
        symbols.append("".join(buffer))
    return symbols


def parse_algebra(text: str) -> Dict[str, Any]:
    """Read one algebra expression into a logic tree (``op``/``args`` or ``leaf``).

    Structural only: brackets must balance and a round node must start with a
    name, but the operator arity is not checked here. That stays with the caller
    (``dataset/tools/logic_map.check_tree``) and with :class:`AlgebraMask`, which
    refuses a wrong arity while the token is still being chosen.
    """
    symbols = tokenize_algebra(text)
    if not symbols:
        raise AlgebraMaskError("empty algebra expression")

    def parse_node(index: int) -> Tuple[Dict[str, Any], int]:
        symbol = symbols[index]
        if symbol == SQUARE_OPEN:
            payload: List[str] = []
            cursor = index + 1
            while cursor < len(symbols) and symbols[cursor] != SQUARE_CLOSE:
                if symbols[cursor] in BRACKETS:
                    raise AlgebraMaskError("a leaf cannot contain a bracket at position " + str(cursor))
                payload.append(symbols[cursor])
                cursor += 1
            if cursor >= len(symbols):
                raise AlgebraMaskError("leaf opened at " + str(index) + " is never closed")
            if not payload:
                raise AlgebraMaskError("leaf at " + str(index) + " holds no token")
            return {"leaf": payload}, cursor + 1
        if symbol != ROUND_OPEN:
            raise AlgebraMaskError("a node must open with ( or [ but found " + repr(symbol))
        if index + 1 >= len(symbols) or symbols[index + 1] in BRACKETS:
            raise AlgebraMaskError("round node at " + str(index) + " has no operator name")
        name = symbols[index + 1]
        args: List[Dict[str, Any]] = []
        cursor = index + 2
        while cursor < len(symbols) and symbols[cursor] != ROUND_CLOSE:
            if symbols[cursor] == SQUARE_CLOSE:
                raise AlgebraMaskError("unexpected ] at position " + str(cursor))
            child, cursor = parse_node(cursor)
            args.append(child)
        if cursor >= len(symbols):
            raise AlgebraMaskError("round node for " + name + " is never closed")
        return {"op": name, "args": args}, cursor + 1

    tree, position = parse_node(0)
    if position != len(symbols):
        raise AlgebraMaskError("trailing symbols after the root node: " + " ".join(symbols[position:]))
    return tree


def stream_of_tree(node: Dict[str, Any]) -> List[str]:
    """Flatten a logic tree into its algebra symbol stream."""
    if "leaf" in node:
        return [SQUARE_OPEN] + [str(token) for token in node["leaf"]] + [SQUARE_CLOSE]
    symbols = [ROUND_OPEN, node["op"]]
    for child in node.get("args", []):
        symbols.extend(stream_of_tree(child))
    symbols.append(ROUND_CLOSE)
    return symbols


def algebra_word(node: Dict[str, Any]) -> str:
    """Canonical bracket word of a tree: the symbols joined by single spaces."""
    return " ".join(stream_of_tree(node))


def skeleton_of_stream(symbols: Sequence[str]) -> str:
    """Project a symbol stream onto the four Dyck-2 terminals."""
    return "".join(symbol for symbol in symbols if symbol in BRACKETS)


# ---------------------------------------------------------------------------
# The automaton
# ---------------------------------------------------------------------------
class AlgebraState:
    """Stack of open frames describing how much of the tree has been written."""

    __slots__ = ("stack", "emitted")

    def __init__(self) -> None:
        self.stack: List[Tuple[Any, ...]] = []
        self.emitted = 0

    def copy(self) -> "AlgebraState":
        clone = AlgebraState()
        clone.stack = list(self.stack)
        clone.emitted = self.emitted
        return clone

    def depth(self) -> int:
        return len(self.stack)


class AlgebraMask:
    """Viable-prefix automaton of the bracket algebra, learned alphabet.

    Parameters
    ----------
    arities:
        Operator name to number of children, from the operator registry.
    leaf_forms:
        First payload token of a square node to the set of payload lengths that
        token is observed with (length counted with the first token included).
    leaf_value_tokens:
        Tokens allowed after the first payload token (values, units, ordinals).
    child_kinds:
        Optional refinement learned from the corpus: which node kinds an operator
        argument position is ever filled with. Without it, every argument may be
        either a round node or a square node.
    """

    def __init__(
        self,
        arities: Dict[str, int],
        leaf_forms: Dict[str, Set[int]],
        leaf_value_tokens: Iterable[str],
        child_kinds: Optional[Dict[Tuple[str, int], Set[str]]] = None,
        max_leaf_length: int = MAX_LEAF_LENGTH,
    ) -> None:
        if not arities:
            raise AlgebraMaskError("the operator registry is empty")
        self.arities = {name: int(arity) for name, arity in arities.items()}
        self.leaf_forms = {token: set(lengths) for token, lengths in leaf_forms.items() if lengths}
        self.leaf_value_tokens = set(leaf_value_tokens)
        self.child_kinds = dict(child_kinds or {})
        self.max_leaf_length = int(max_leaf_length)
        for token, lengths in self.leaf_forms.items():
            if max(lengths) > self.max_leaf_length:
                raise AlgebraMaskError(
                    "leaf form " + repr(token) + " needs " + str(max(lengths)) + " tokens"
                )

    # ------------------------------------------------------------------
    # Construction from data
    # ------------------------------------------------------------------
    @classmethod
    def from_leaf_payloads(
        cls,
        arities: Dict[str, int],
        leaf_payloads: Iterable[Sequence[str]],
        child_kinds: Optional[Dict[Tuple[str, int], Set[str]]] = None,
        max_leaf_length: int = MAX_LEAF_LENGTH,
    ) -> "AlgebraMask":
        """Build the leaf grammar from observed square-node payloads."""
        forms: Dict[str, Set[int]] = {}
        values: Set[str] = set()
        for payload in leaf_payloads:
            payload = [str(token) for token in payload]
            if not payload:
                continue
            if len(payload) > max_leaf_length:
                raise AlgebraMaskError(
                    "leaf payload longer than the allowed maximum: " + " ".join(payload)
                )
            forms.setdefault(payload[0], set()).add(len(payload))
            values.update(payload[1:])
        if not forms:
            raise AlgebraMaskError("no leaf payload observed")
        return cls(arities, forms, values, child_kinds, max_leaf_length)

    @staticmethod
    def observe_child_kinds(node: Dict[str, Any], counts: Counter) -> None:
        """Count which kind fills each operator argument position."""
        if "leaf" in node:
            return
        for position, child in enumerate(node.get("args", [])):
            kind = SQUARE_OPEN if "leaf" in child else ROUND_OPEN
            counts[(node["op"], position, kind)] += 1
            AlgebraMask.observe_child_kinds(child, counts)

    @staticmethod
    def child_kinds_from_counts(counts: Counter) -> Dict[Tuple[str, int], Set[str]]:
        table: Dict[Tuple[str, int], Set[str]] = {}
        for (name, position, kind), _ in counts.items():
            table.setdefault((name, position), set()).add(kind)
        return table

    # ------------------------------------------------------------------
    # Automaton
    # ------------------------------------------------------------------
    def initial_state(self) -> AlgebraState:
        return AlgebraState()

    def is_complete(self, state: AlgebraState) -> bool:
        """True when a whole tree has been written (the stack is back to empty)."""
        return not state.stack and state.emitted > 0

    def allowed_symbols(self, state: AlgebraState) -> Set[str]:
        """The set of legal next algebra symbols for this prefix."""
        if not state.stack:
            if state.emitted == 0:
                # The root is always an operator node in this language.
                return {ROUND_OPEN}
            return set()
        frame = state.stack[-1]
        kind = frame[0]
        if kind == "op_open":
            return set(self.arities)
        if kind == "op":
            name, produced = frame[1], frame[2]
            if produced >= self.arities[name]:
                return {ROUND_CLOSE}
            openers = set(KIND_OPENERS)
            narrowing = self.child_kinds.get((name, produced))
            if narrowing:
                openers &= set(narrowing)
            return openers or set(KIND_OPENERS)
        if kind == "leaf":
            first, count = frame[1], frame[2]
            if count == 0:
                return set(self.leaf_forms)
            lengths = self.leaf_forms.get(first)
            if lengths is None:
                return {SQUARE_CLOSE} | self.leaf_value_tokens
            allowed: Set[str] = set()
            if count < max(lengths) and self.leaf_value_tokens:
                allowed |= self.leaf_value_tokens
            if count in lengths:
                allowed.add(SQUARE_CLOSE)
            return allowed or {SQUARE_CLOSE}
        raise AlgebraMaskError("unknown automaton frame " + repr(frame))

    def step(self, state: AlgebraState, symbol: str) -> AlgebraState:
        """Advance the automaton by one symbol, or raise on an illegal one.

        The only place where the shape of the tree is decided is the operator
        name: it fixes the number of children that follow, and with it every
        later closing bracket. The root restriction needs no extra rule, the
        empty stack with nothing emitted already offers only ``(``.
        """
        allowed = self.allowed_symbols(state)
        if symbol not in allowed:
            raise AlgebraMaskError(
                "symbol " + repr(symbol) + " is not legal after this prefix "
                + "(open nodes: " + str(state.depth()) + ")"
            )
        frame = state.stack[-1] if state.stack else None
        if symbol == ROUND_OPEN:
            state.stack.append(("op_open",))
        elif symbol == SQUARE_OPEN:
            state.stack.append(("leaf", None, 0))
        elif frame is not None and frame[0] == "op_open":
            state.stack[-1] = ("op", symbol, 0)
        elif symbol == SQUARE_CLOSE or symbol == ROUND_CLOSE:
            state.stack.pop()
            self._count_child(state)
        else:
            if frame is None or frame[0] != "leaf":
                raise AlgebraMaskError("a payload token needs an open square node")
            first, count = frame[1], frame[2]
            state.stack[-1] = ("leaf", symbol if count == 0 else first, count + 1)
        state.emitted += 1
        return state

    @staticmethod
    def _count_child(state: AlgebraState) -> None:
        if state.stack and state.stack[-1][0] == "op":
            name, produced = state.stack[-1][1], state.stack[-1][2]
            state.stack[-1] = ("op", name, produced + 1)

    def replay(self, symbols: Sequence[str]) -> AlgebraState:
        """Run the automaton over a whole prefix; raise at the first illegal symbol."""
        state = self.initial_state()
        for symbol in symbols:
            self.step(state, symbol)
        return state

    def validate_stream(self, symbols: Sequence[str]) -> Dict[str, Any]:
        """Check a full stream: every step legal, exactly one complete tree at the end."""
        state = self.replay(symbols)
        if not self.is_complete(state):
            raise AlgebraMaskError(
                "the stream stops with " + str(state.depth()) + " node(s) still open"
            )
        tree = parse_algebra(" ".join(symbols))
        self._check_arities(tree)
        return tree

    def _check_arities(self, node: Dict[str, Any]) -> None:
        if "leaf" in node:
            return
        name = node["op"]
        if name not in self.arities:
            raise AlgebraMaskError("unknown operator " + repr(name))
        args = node.get("args", [])
        if len(args) != self.arities[name]:
            raise AlgebraMaskError(
                "operator " + name + " expects " + str(self.arities[name]) + " arguments, got "
                + str(len(args))
            )
        for child in args:
            self._check_arities(child)

    def sample_leaf(self, rng: random.Random) -> Dict[str, Any]:
        """Draw a square node whose payload shape is one the corpus actually uses."""
        first = rng.choice(sorted(self.leaf_forms))
        lengths = sorted(self.leaf_forms[first])
        length = rng.choice(lengths)
        payload = [first]
        if length > 1:
            if not self.leaf_value_tokens:
                raise AlgebraMaskError("no payload value token is available to fill a longer leaf")
            values = sorted(self.leaf_value_tokens)
            payload.extend(rng.choice(values) for _ in range(length - 1))
        return {"leaf": payload}

    def sample_node(self, rng: random.Random, depth: int, allow_leaf: bool = True) -> Dict[str, Any]:
        """Draw one random tree inside a depth budget."""
        if allow_leaf and (depth <= 1 or rng.random() < 0.4):
            return self.sample_leaf(rng)
        name = rng.choice(sorted(self.arities))
        children = [self.sample_node(rng, depth - 1) for _ in range(self.arities[name])]
        return {"op": name, "args": children}

    def sample_stream(self, rng: random.Random, max_depth: int = 5, max_symbols: int = 400) -> List[str]:
        """Draw one random legal stream: a tree inside the depth budget, rendered.

        The budget is on tree depth, not on stream length: a uniform step by step
        walk of a pushdown automaton keeps opening nodes and has no finite expected
        length, while a depth budget keeps the sample finite and still exercises
        every transition of the automaton.
        """
        if max_depth < 1:
            raise AlgebraMaskError("the depth budget must be at least 1")
        stream = stream_of_tree(self.sample_node(rng, max_depth, allow_leaf=False))
        if len(stream) > max_symbols:
            raise AlgebraMaskError("sampled stream exceeded " + str(max_symbols) + " symbols")
        return stream

    def describe(self) -> Dict[str, Any]:
        """Summary of the alphabet the mask allows, for the gate reports."""
        return {
            "operators": len(self.arities),
            "leaf_first_tokens": len(self.leaf_forms),
            "leaf_value_tokens": len(self.leaf_value_tokens),
            "leaf_length_profile": dict(
                Counter(len(self.leaf_forms[token]) for token in self.leaf_forms)
            ),
            "child_kind_positions": len(self.child_kinds),
            "max_leaf_length": self.max_leaf_length,
        }

    def to_spec(self) -> Dict[str, Any]:
        """JSON-safe description of the exact alphabet this mask admits."""
        return {
            "arities": {name: int(self.arities[name]) for name in sorted(self.arities)},
            "leaf_forms": {
                token: sorted(self.leaf_forms[token]) for token in sorted(self.leaf_forms)
            },
            "leaf_value_tokens": sorted(self.leaf_value_tokens),
            "child_kinds": {
                name + " " + str(position): sorted(kinds)
                for (name, position), kinds in sorted(self.child_kinds.items())
            },
            "max_leaf_length": self.max_leaf_length,
        }

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "AlgebraMask":
        """Rebuild the mask described by :meth:`to_spec`, refusing partial data."""
        if not isinstance(spec, dict):
            raise AlgebraMaskError("the mask specification must be a mapping")
        missing = [key for key in ("arities", "leaf_forms") if key not in spec]
        if missing:
            raise AlgebraMaskError("the mask specification is missing: " + ", ".join(missing))
        child_kinds: Dict[Tuple[str, int], Set[str]] = {}
        for key, kinds in (spec.get("child_kinds") or {}).items():
            name, separator, position = str(key).rpartition(" ")
            if not separator or not name or not position.isdigit():
                raise AlgebraMaskError("malformed child kind key: " + str(key))
            child_kinds[(name, int(position))] = set(kinds)
        return cls(
            arities={str(name): int(arity) for name, arity in spec["arities"].items()},
            leaf_forms={str(token): set(lengths) for token, lengths in spec["leaf_forms"].items()},
            leaf_value_tokens=set(spec.get("leaf_value_tokens") or ()),
            child_kinds=child_kinds,
            max_leaf_length=int(spec.get("max_leaf_length", MAX_LEAF_LENGTH)),
        )


# ---------------------------------------------------------------------------
# Token level view: the mask the training and decoding loops use
# ---------------------------------------------------------------------------
class TokenStreamMask:
    """Same automaton seen through the ids of the learned tokenizer.

    The target protocol is
    ``<RULE_START> sym " " sym " " ... sym <RULE_END> <EOS>``: the algebra is
    written with the canonical single-space rendering of the dataset, so a space
    token is forced between two symbols and nothing else, and ``<RULE_END>`` is
    offered exactly when the tree is complete.
    """

    def __init__(self, mask: AlgebraMask, tokenizer: Tokenizer, space_token: str = " ") -> None:
        self.mask = mask
        self.tokenizer = tokenizer
        if not is_atomic_unit(space_token, tokenizer.atomic_symbols):
            raise TokenizerError("the space token is not atomic in this vocabulary")
        self.space_id = tokenizer.token_id(space_token)
        self.rule_start_id = tokenizer.token_id(RULE_START)
        self.rule_end_id = tokenizer.token_id(RULE_END)
        self.eos_id = tokenizer.token_id(EOS)
        self.pad_id = tokenizer.token_id(PAD)
        self.symbol_ids: Dict[str, int] = {}
        missing: List[str] = []
        for symbol in list(mask.arities) + list(mask.leaf_forms) + list(mask.leaf_value_tokens):
            if symbol not in tokenizer.id_of:
                missing.append(symbol)
            else:
                self.symbol_ids[symbol] = tokenizer.id_of[symbol]
        for bracket in BRACKETS:
            self.symbol_ids[bracket] = tokenizer.token_id(bracket)
        if missing:
            raise TokenizerError(
                "algebra symbols missing from the vocabulary: " + ", ".join(sorted(missing)[:12])
            )

    def encode_target(self, symbols: Sequence[str], add_eos: bool = True) -> List[int]:
        """Ids of one rule region: start marker, spaced symbols, end marker."""
        ids: List[int] = [self.rule_start_id]
        for index, symbol in enumerate(symbols):
            if index:
                ids.append(self.space_id)
            ids.append(self.symbol_ids[symbol])
        ids.append(self.rule_end_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def region_start(self, prefix_ids: Sequence[int]) -> int:
        """Index of the last ``<RULE_START>``, or -1 when the region never opened."""
        for index in range(len(prefix_ids) - 1, -1, -1):
            if prefix_ids[index] == self.rule_start_id:
                return index
        return -1

    def region_ids(self, prefix_ids: Sequence[int]) -> List[int]:
        """Ids after the last ``<RULE_START>``, that is the part being written."""
        start = self.region_start(prefix_ids)
        if start < 0:
            return []
        return list(prefix_ids[start + 1 :])

    def allowed_next_ids(self, prefix_ids: Sequence[int]) -> Set[int]:
        """Legal next token ids for the given prefix of the target sequence."""
        if not prefix_ids:
            return {self.rule_start_id}
        last = self.tokenizer.token_of(prefix_ids[-1])
        if last in (EOS, PAD):
            return {self.pad_id}
        if last == RULE_END:
            return {self.eos_id}
        if self.region_start(prefix_ids) < 0:
            # A prefix written outside the rule protocol: restart the protocol.
            return {self.rule_start_id}
        region = self.region_ids(prefix_ids)
        if not region:
            # Nothing of the rule region has been written yet.
            state = self.mask.initial_state()
            return self._ids_for_state(state)
        tokens = [self.tokenizer.token_of(token_id) for token_id in region]
        if tokens[-1] == " ":
            state = self.mask.replay([token for token in tokens if token != " "])
            return self._ids_for_state(state)
        stream = [token for token in tokens if token != " "]
        state = self.mask.replay(stream)
        if self.mask.is_complete(state):
            return {self.rule_end_id}
        return {self.space_id}

    def _ids_for_state(self, state: AlgebraState) -> Set[int]:
        allowed = self.mask.allowed_symbols(state)
        return {self.symbol_ids[symbol] for symbol in allowed if symbol in self.symbol_ids}

    def mask_logits(self, logits: torch.Tensor, prefix_ids: Sequence[int]) -> torch.Tensor:
        """Return logits with every illegal entry set to minus infinity."""
        if logits.dim() != 1:
            raise ValueError("mask_logits expects a one dimensional logits vector")
        if logits.numel() < self.tokenizer.vocabulary_size:
            raise ValueError(
                "logits width " + str(logits.numel()) + " is smaller than the vocabulary "
                + str(self.tokenizer.vocabulary_size)
            )
        allowed = self.allowed_next_ids(prefix_ids)
        if not allowed:
            raise AlgebraMaskError("no legal continuation for this prefix")
        keep = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
        index = torch.tensor(sorted(allowed), dtype=torch.long, device=logits.device)
        keep.index_fill_(0, index, True)
        scored = logits.clone()
        scored[~keep] = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        return scored

    def report(self) -> Dict[str, Any]:
        summary = self.mask.describe()
        summary["vocabulary_size"] = self.tokenizer.vocabulary_size
        summary["masked_alphabet_tokens"] = len(self.symbol_ids)
        return summary
