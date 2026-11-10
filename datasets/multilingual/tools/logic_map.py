"""Canonical logic tree of the LingoLOGOS dataset and its deterministic renderings.

The tree is the single source of truth of a dataset item. Every other field of an
item is a rendering of the same tree:

  bracket_ast        parenthesised algebra, round brackets for operator nodes,
                     square brackets for value and entity leaves
  bracket_skeleton   bracket sequence only, a word of Dyck-2 over the four
                     terminals induced by the project engine
  semantic_pattern   functional notation used by the SFT supervision target
  gf_abstract        Grammatical Framework style abstract syntax
  z3_assertion       Python Z3 flavour assertion built with uninterpreted symbols

The module is pure Python: it does not import torch and it does not touch the
project sources, so it can be reused by the generator, the validator and the
training entry points.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROUND_OPEN = "("
ROUND_CLOSE = ")"
SQUARE_OPEN = "["
SQUARE_CLOSE = "]"

TERMINALS = (ROUND_OPEN, ROUND_CLOSE, SQUARE_OPEN, SQUARE_CLOSE)

# Nonterminal labels of the gold Dyck-2 grammar used by the project engine.
NONTERMINALS = ("S", "OPN_R", "CLS_R", "OPN_S", "CLS_S")
START_SYMBOL = "S"


class LogicTreeError(Exception):
    """Raised when a logic tree is malformed or uses an unknown operator."""


def is_operator_node(node: Any) -> bool:
    return isinstance(node, dict) and "op" in node


def is_leaf_node(node: Any) -> bool:
    return isinstance(node, dict) and "leaf" in node


def check_tree(node: Any, registry: Dict[str, Dict[str, Any]]) -> None:
    """Validate the structural invariants of a logic tree."""
    if is_leaf_node(node):
        tokens = node["leaf"]
        if not isinstance(tokens, list) or not tokens:
            raise LogicTreeError("leaf nodes must hold a non empty token list")
        for token in tokens:
            if not isinstance(token, (str, int)):
                raise LogicTreeError("leaf tokens must be strings or integers")
        return
    if not is_operator_node(node):
        raise LogicTreeError("a tree node must carry either an op or a leaf key")
    name = node["op"]
    if name not in registry:
        raise LogicTreeError("unknown operator " + str(name))
    args = node.get("args", [])
    expected = registry[name]["arity"]
    if len(args) != expected:
        raise LogicTreeError(
            "operator " + name + " expects " + str(expected) + " arguments, got " + str(len(args))
        )
    for child in args:
        check_tree(child, registry)


def node_key(node: Any) -> str:
    """Stable string identity of a resolved tree, used for deduplication."""
    if is_leaf_node(node):
        return "[" + " ".join(str(token) for token in node["leaf"]) + "]"
    inner = " ".join(node_key(child) for child in node["args"])
    return "(" + node["op"] + (" " + inner if inner else "") + ")"


def tree_fingerprint(node: Any) -> str:
    """Shape only identity: operators and leaf tags, leaf payloads dropped."""
    if is_leaf_node(node):
        return "[ ]"
    inner = "".join(tree_fingerprint(child) for child in node["args"])
    return "(" + node["op"] + inner + ")"


def bracket_ast(node: Any) -> str:
    """Round brackets for operators, square brackets for leaves."""
    if is_leaf_node(node):
        inner = " ".join(str(token) for token in node["leaf"])
        return SQUARE_OPEN + " " + inner + " " + SQUARE_CLOSE
    parts = [node["op"]]
    parts.extend(bracket_ast(child) for child in node["args"])
    return ROUND_OPEN + " " + " ".join(parts) + " " + ROUND_CLOSE


def bracket_skeleton(node: Any) -> str:
    """Bracket sequence of the tree: a word over the four Dyck-2 terminals."""
    if is_leaf_node(node):
        return SQUARE_OPEN + SQUARE_CLOSE
    inner = "".join(bracket_skeleton(child) for child in node["args"])
    return ROUND_OPEN + inner + ROUND_CLOSE


def skeleton_length(node: Any) -> int:
    return len(bracket_skeleton(node))


def is_well_bracketed(sequence: str) -> bool:
    """True when the sequence is a word of Dyck-2 with balanced round and square pairs."""
    stack: List[str] = []
    partner = {ROUND_CLOSE: ROUND_OPEN, SQUARE_CLOSE: SQUARE_OPEN}
    for char in sequence:
        if char == ROUND_OPEN or char == SQUARE_OPEN:
            stack.append(char)
        elif char in partner:
            if not stack or stack[-1] != partner[char]:
                return False
            stack.pop()
        else:
            return False
    return not stack


def nonterminal_sequence(sequence: str) -> List[str]:
    """Label every terminal with the gold nonterminal that derives it."""
    stack: List[str] = []
    labels: List[str] = []
    for char in sequence:
        if char == ROUND_OPEN:
            stack.append("OPN_R")
        elif char == SQUARE_OPEN:
            stack.append("OPN_S")
        elif char == ROUND_CLOSE:
            stack.append("CLS_R")
        elif char == SQUARE_CLOSE:
            stack.append("CLS_S")
        else:
            raise LogicTreeError("unexpected terminal " + repr(char))
        labels.append(stack[-1])
    return labels


# ---------------------------------------------------------------------------
# Leaf helpers
# ---------------------------------------------------------------------------

NUMERIC_TAGS = {"pct", "celsius", "eur", "minute", "hour", "day", "week", "month", "count", "mmhg", "mgdl", "bpm", "kg", "kwh"}


def leaf_tokens(node: Any) -> List[str]:
    return [str(token) for token in node["leaf"]]


def is_numeric_leaf(node: Any) -> bool:
    if not is_leaf_node(node):
        return False
    tokens = leaf_tokens(node)
    return len(tokens) == 2 and tokens[0] in NUMERIC_TAGS and tokens[1].lstrip("-").isdigit()


def numeric_value(node: Any) -> float:
    return float(leaf_tokens(node)[1])


UNIT_GAMES = {
    "pct": "Pct",
    "celsius": "Celsius",
    "eur": "Eur",
    "minute": "Minute",
    "hour": "Hour",
    "day": "Day",
    "week": "Week",
    "month": "Month",
    "count": "Count",
    "mmhg": "MmHg",
    "mgdl": "MgDl",
    "bpm": "Bpm",
    "kg": "Kg",
    "kwh": "Kwh",
}


LOGICAL_ATOMS = frozenset(
    (
        "IF",
        "UNLESS",
        "EXCEPT",
        "AND",
        "OR",
        "NOT",
        "STATE",
        "GT",
        "GTE",
        "LT",
        "LTE",
        "BETWEEN",
        "ON",
        "OFF",
        "AT",
        "DO",
        "SET",
    )
)


def camel(token: str) -> str:
    """CamelCase identifier for GF and SMT renderings of an operator name."""
    cleaned = "".join(part for part in token.replace(":", " ").split() if part)
    if not cleaned:
        return "X"
    if cleaned[0].isdigit():
        return cleaned
    if "_" not in cleaned and cleaned in LOGICAL_ATOMS:
        return cleaned
    parts = [part for part in cleaned.split("_") if part]
    out = []
    for part in parts:
        if len(part) == 1:
            out.append(part.upper())
        else:
            out.append(part[:1].upper() + part[1:].lower())
    return "".join(out)


def symbol_name(token: str) -> str:
    """SMT friendly identifier: lower case, alphanumeric and underscore only."""
    out = []
    for char in token.lower():
        out.append(char if char.isalnum() else "_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    if not name:
        return "x"
    if name[0].isdigit():
        name = "n" + name
    return name


# ---------------------------------------------------------------------------
# Semantic pattern rendering
# ---------------------------------------------------------------------------

def semantic_pattern(node: Any) -> str:
    """Functional notation, for example IF(AND(GT(humidity, Pct(65))), ON(dehumidifier))."""
    if is_leaf_node(node):
        tokens = leaf_tokens(node)
        if is_numeric_leaf(node):
            unit, value = tokens
            return UNIT_GAMES.get(unit, camel(unit)) + "(" + value + ")"
        if len(tokens) >= 2 and tokens[0] in ("clock", "band", "ord"):
            args = ", ".join('"' + token + '"' if not token.isdigit() else token for token in tokens[1:])
            return camel(tokens[0]) + "(" + args + ")"
        return symbol_name(tokens[0]) if len(tokens) == 1 else camel(tokens[0]) + "(" + ", ".join(tokens[1:]) + ")"
    args = ", ".join(semantic_pattern(child) for child in node["args"])
    return node["op"] + "(" + args + ")"


def gf_leaf(node: Any) -> str:
    """Leaf rendering for the GF view: single token bare, tagged leaves parenthesised."""
    tokens = leaf_tokens(node)
    if is_numeric_leaf(node):
        return "(" + UNIT_GAMES.get(tokens[0], camel(tokens[0])) + " " + tokens[1] + ")"
    if len(tokens) == 1:
        return camel(tokens[0])
    parts = []
    for index, token in enumerate(tokens):
        parts.append(camel(token) if index == 0 or not token.isdigit() else token)
    return "(" + " ".join(parts) + ")"


def gf_abstract(node: Any) -> str:
    """Grammatical Framework flavour abstract syntax, postfix and parenthesised."""
    if is_leaf_node(node):
        return gf_leaf(node)
    parts = [node.get("gf", camel(node["op"]))]
    for child in node["args"]:
        rendered = gf_abstract(child)
        if is_leaf_node(child):
            parts.append(rendered if rendered.startswith("(") else rendered)
        else:
            parts.append(rendered if rendered.startswith("(") else "(" + rendered + ")")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Z3 rendering
# ---------------------------------------------------------------------------

def z3_term(node: Any) -> str:
    """Python Z3 flavour term with uninterpreted symbols and functions."""
    if is_leaf_node(node):
        tokens = leaf_tokens(node)
        if is_numeric_leaf(node):
            value = tokens[1]
            return str(int(float(value))) if float(value).is_integer() else value
        if tokens[0] == "clock":
            return "clock_" + "_".join(tokens[1:])
        if tokens[0] == "state":
            return "state_" + symbol_name(tokens[1])
        if len(tokens) == 1 and tokens[0] == tokens[0].upper():
            return symbol_name(tokens[0])
        return symbol_name(tokens[0])
    op = node["op"]
    args = [z3_term(child) for child in node["args"]]
    if op == "IF":
        return "Implies(" + args[0] + ", " + args[1] + ")"
    if op in ("UNLESS", "EXCEPT"):
        return "Implies(Not(" + args[0] + "), " + args[1] + ")"
    if op == "AND":
        return "And(" + args[0] + ", " + args[1] + ")"
    if op == "OR":
        return "Or(" + args[0] + ", " + args[1] + ")"
    if op == "NOT":
        return "Not(" + args[0] + ")"
    if op == "STATE":
        return "state_" + args[0]
    if op == "GT":
        return "(" + args[0] + " > " + args[1] + ")"
    if op == "GTE":
        return "(" + args[0] + " >= " + args[1] + ")"
    if op == "LT":
        return "(" + args[0] + " < " + args[1] + ")"
    if op == "LTE":
        return "(" + args[0] + " <= " + args[1] + ")"
    if op == "BETWEEN":
        return "And((" + args[0] + " >= " + args[1] + "), (" + args[0] + " <= " + args[2] + "))"
    return node.get("gf", camel(op)) + "(" + ", ".join(args) + ")"


def z3_assertion(node: Any) -> str:
    return "s.add(" + z3_term(node) + ")"


def smtlib_text(node: Any) -> str:
    """SMT-LIB flavour rendering, a portable secondary view of the same tree."""
    if is_leaf_node(node):
        tokens = leaf_tokens(node)
        if is_numeric_leaf(node):
            value = tokens[1]
            return str(int(float(value))) if float(value).is_integer() else value
        if tokens[0] == "clock":
            return symbol_name("clock_" + "_".join(tokens[1:]))
        return symbol_name(tokens[-1])
    op = node["op"]
    args = [smtlib_text(child) for child in node["args"]]
    if op in ("UNLESS", "EXCEPT"):
        return "(implies (not " + args[0] + ") " + args[1] + ")"
    shape = {"IF": "implies", "AND": "and", "OR": "or", "NOT": "not"}.get(op)
    if shape:
        return "(" + shape + " " + " ".join(args) + ")"
    if op == "STATE":
        return "state_" + args[0]
    if op in ("GT", "GTE", "LT", "LTE"):
        rel = {"GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}[op]
        return "(" + rel + " " + args[0] + " " + args[1] + ")"
    if op == "BETWEEN":
        return "(and (>= " + args[0] + " " + args[1] + ") (<= " + args[0] + " " + args[2] + "))"
    return "(" + symbol_name(op) + " " + " ".join(args) + ")"


BOOL_CATEGORIES = frozenset(("logic", "conditional", "threshold", "predicate", "action", "recurrence", "frequency"))
INT_CATEGORIES = frozenset(("aggregate",))


def leaf_sort(node: Any, registry: Dict[str, Dict[str, Any]], position: str = "entity") -> str:
    """Sort of a leaf: int for numbers and clocks, int for comparison subjects, domain otherwise."""
    if is_numeric_leaf(node):
        return "int"
    tokens = leaf_tokens(node)
    if tokens and tokens[0] == "clock":
        return "int"
    if position == "subject":
        return "int"
    return "dom"


def node_sort(node: Any, registry: Dict[str, Dict[str, Any]], position: str = "entity") -> str:
    if is_leaf_node(node):
        return leaf_sort(node, registry, position)
    category = registry[node["op"]]["category"]
    if category in INT_CATEGORIES:
        return "int"
    if category in BOOL_CATEGORIES:
        return "bool"
    return "dom"


def build_z3(node: Any, z3_module: Any, registry: Dict[str, Dict[str, Any]], cache: Dict[str, Any], position: str = "entity") -> Any:
    """Build a real Z3 expression from a logic tree.

    Clause leaves become Bool constants named state_<id>, comparison subjects become
    Int constants, numbers become integer literals, clocks become Int constants,
    aggregates become Int valued uninterpreted functions, and every action, window,
    recurrence and frequency node becomes an uninterpreted function with derived sorts.
    """
    if is_leaf_node(node):
        tokens = leaf_tokens(node)
        if is_numeric_leaf(node):
            value = float(tokens[1])
            return z3_module.IntVal(int(value)) if value.is_integer() else z3_module.RealVal(value)
        name = symbol_name("clock_" + "_".join(tokens[1:])) if tokens[0] == "clock" else symbol_name(tokens[-1])
        if tokens[0] == "clock" or position == "subject":
            key = "int:" + name
            if key not in cache:
                cache[key] = z3_module.Const(name, z3_module.IntSort())
            return cache[key]
        key = "dom:" + name
        if key not in cache:
            dom = cache.setdefault("$sort", z3_module.DeclareSort("Domain"))
            cache[key] = z3_module.Const(name, dom)
        return cache[key]
    op = node["op"]
    args = node["args"]
    if op == "IF":
        return z3_module.Implies(
            build_z3(args[0], z3_module, registry, cache),
            build_z3(args[1], z3_module, registry, cache),
        )
    if op in ("UNLESS", "EXCEPT"):
        return z3_module.Implies(
            z3_module.Not(build_z3(args[0], z3_module, registry, cache)),
            build_z3(args[1], z3_module, registry, cache),
        )
    if op == "AND":
        return z3_module.And(
            build_z3(args[0], z3_module, registry, cache),
            build_z3(args[1], z3_module, registry, cache),
        )
    if op == "OR":
        return z3_module.Or(
            build_z3(args[0], z3_module, registry, cache),
            build_z3(args[1], z3_module, registry, cache),
        )
    if op == "NOT":
        return z3_module.Not(build_z3(args[0], z3_module, registry, cache))
    if op == "STATE":
        name = "state_" + symbol_name(leaf_tokens(args[0])[-1])
        if name not in cache:
            cache[name] = z3_module.Bool(name)
        return cache[name]
    if op in ("GT", "GTE", "LT", "LTE"):
        left = build_z3(args[0], z3_module, registry, cache, position="subject")
        right = build_z3(args[1], z3_module, registry, cache)
        if op == "GT":
            return left > right
        if op == "GTE":
            return left >= right
        if op == "LT":
            return left < right
        return left <= right
    if op == "BETWEEN":
        subject = build_z3(args[0], z3_module, registry, cache, position="subject")
        low = build_z3(args[1], z3_module, registry, cache)
        high = build_z3(args[2], z3_module, registry, cache)
        return z3_module.And(subject >= low, subject <= high)
    sorts = [node_sort(child, registry, "subject" if op in ("GT", "GTE", "LT", "LTE", "BETWEEN") and index == 0 else "entity") for index, child in enumerate(args)]
    result = node_sort(node, registry)
    terms = [
        build_z3(child, z3_module, registry, cache, position="subject" if op in ("GT", "GTE", "LT", "LTE", "BETWEEN") and index == 0 else "entity")
        for index, child in enumerate(args)
    ]
    abbrev = {"int": "i", "bool": "b", "dom": "d"}
    signature = "".join(abbrev[sort] for sort in sorts) + "_" + abbrev[result]
    fname = op.lower() + "_" + signature
    if fname not in cache:
        dom = cache.setdefault("$sort", z3_module.DeclareSort("Domain"))
        argument_sorts = []
        for sort in sorts:
            if sort == "int":
                argument_sorts.append(z3_module.IntSort())
            elif sort == "bool":
                argument_sorts.append(z3_module.BoolSort())
            else:
                argument_sorts.append(dom)
        if result == "int":
            codomain = z3_module.IntSort()
        elif result == "bool":
            codomain = z3_module.BoolSort()
        else:
            codomain = dom
        cache[fname] = z3_module.Function(fname, *(argument_sorts + [codomain]))
    return cache[fname](*terms)


def walk(node: Any) -> Iterable[Any]:
    yield node
    if is_operator_node(node):
        for child in node["args"]:
            yield from walk(child)


def negative_control(sequence: str, rng_random: Any) -> str:
    """Corrupt a valid skeleton: the project verifier uses these as rejection probes."""
    chars = list(sequence)
    if len(chars) < 4:
        return sequence
    mode = rng_random.randrange(3)
    openers = [i for i, c in enumerate(chars) if c in (ROUND_OPEN, SQUARE_OPEN)]
    closers = [i for i, c in enumerate(chars) if c in (ROUND_CLOSE, SQUARE_CLOSE)]
    if mode == 0 and closers:
        idx = rng_random.choice(closers)
        chars[idx] = SQUARE_CLOSE if chars[idx] == ROUND_CLOSE else ROUND_CLOSE
    elif mode == 1 and openers:
        idx = rng_random.choice(openers)
        del chars[idx]
    elif closers:
        chars.insert(rng_random.randrange(len(chars) + 1), rng_random.choice([ROUND_OPEN, SQUARE_OPEN]))
    return "".join(chars)
