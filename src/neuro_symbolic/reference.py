"""
Formal references for the Dyck-2 language: the gold CNF grammar, a structured prior for
weight initialization, negative controls and grammar-induction quality metrics.

This data is NOT learned: it is the comparison baseline of the metrics and the optional
prior used by `synthesize.py --init prior`. The `--init random` variant instead starts
from random weights and measures how much of the gold grammar induction really recovers.
"""

import random
from itertools import permutations, product
from typing import Dict, List, Optional, Set, Tuple

from .grammar import PCFGrammar, ProductionRule, Symbol

# Canonical order of the gold non-terminals (index 0 = start symbol S)
GOLD_NON_TERMINALS = ["S", "OPN_R", "CLS_R", "OPN_S", "CLS_S"]
GOLD_TERMINALS = ["(", ")", "[", "]"]
GOLD_START = "S"

# Gold rules in Chomsky Normal Form for Dyck-2:
#   S      -> S S  |  OPN_R CLS_R  |  OPN_S CLS_S
#   OPN_R  -> '('                OPN_S -> '['
#   CLS_R  -> ')'  |  S CLS_R   CLS_S  -> ']' | S CLS_S
GOLD_RULE_TRIPLES: List[Tuple[str, Tuple[str, ...]]] = [
    ("S", ("S", "S")),
    ("S", ("OPN_R", "CLS_R")),
    ("S", ("OPN_S", "CLS_S")),
    ("CLS_R", ("S", "CLS_R")),
    ("CLS_S", ("S", "CLS_S")),
    ("OPN_R", ("(",)),
    ("OPN_S", ("[",)),
    ("CLS_R", (")",)),
    ("CLS_S", ("]",)),
]

# Non-terminals that emit terminals in CNF (binary gate switched off)
GOLD_TERMINAL_NT = ["OPN_R", "OPN_S", "CLS_R", "CLS_S"]


def gold_dyck2_grammar() -> PCFGrammar:
    """Builds the reference CNF grammar for Dyck-2 (uniform rule distribution)."""
    symbols = {nt: Symbol(name=nt, is_terminal=False) for nt in GOLD_NON_TERMINALS}
    terminal_symbols = {t: Symbol(name=t, is_terminal=True) for t in GOLD_TERMINALS}

    by_lhs: Dict[str, List[Tuple[str, ...]]] = {}
    for lhs, rhs in GOLD_RULE_TRIPLES:
        by_lhs.setdefault(lhs, []).append(rhs)

    rules: List[ProductionRule] = []
    for lhs, rhs_options in by_lhs.items():
        weight = 1.0 / len(rhs_options)
        for rhs in rhs_options:
            rhs_symbols = [
                terminal_symbols[name] if name in terminal_symbols else symbols[name]
                for name in rhs
            ]
            rules.append(ProductionRule(lhs=symbols[lhs], rhs=rhs_symbols, probability=weight))

    return PCFGrammar(
        non_terminals=list(GOLD_NON_TERMINALS),
        terminals=list(GOLD_TERMINALS),
        start_symbol=GOLD_START,
        rules=rules,
    )


def apply_gold_prior(parser, nt_labels: List[str]) -> None:
    """
    Initializes the parser logits to the gold rules (structured prior, NOT learned).
    Used only by `--init prior`; the scientific validity of induction has to be measured
    with `--init random` and the gold rule F1.
    Requires nt_labels to be a permutation of GOLD_NON_TERMINALS with the same number
    of terminals as the parameter arrays.
    """
    if len(nt_labels) != len(GOLD_NON_TERMINALS):
        raise ValueError(
            f"the gold prior requires {len(GOLD_NON_TERMINALS)} non-terminals, "
            f"got {len(nt_labels)}"
        )

    index = {name: i for i, name in enumerate(nt_labels)}
    parser.rule_logits.data.zero_()
    parser.term_logits.data.zero_()
    parser.family_logits.data.zero_()

    for lhs, rhs in GOLD_RULE_TRIPLES:
        a = index[lhs]
        if len(rhs) == 2:
            b, c = index[rhs[0]], index[rhs[1]]
            parser.rule_logits.data[a, b, c] = 3.0
        else:
            terminal_index = GOLD_TERMINALS.index(rhs[0])
            parser.term_logits.data[a, terminal_index] = 3.0

    # Family gate: fully on for the structural non-terminals, off for the terminal ones
    for i, name in enumerate(nt_labels):
        parser.family_logits.data[i] = 4.0 if name not in GOLD_TERMINAL_NT else -4.0


def generate_negative_controls(
    positives: List[str],
    max_controls: int = 200,
    seed: int = 17,
) -> List[str]:
    """
    Generates NEGATIVE controls: strings obtained by perturbing valid Dyck-2 sentences and
    then filtered because they do NOT belong to Dyck-k (non-zero final counter or negative
    prefix). They measure the selectivity of the induced language.
    """
    rng = random.Random(seed)
    controls: Set[str] = set()
    brackets = list(GOLD_TERMINALS)

    for word in positives:
        chars = list(word)
        if len(chars) < 2:
            continue
        candidates: List[str] = []

        # 1. flip the bracket type at one position (type mismatch)
        for i in range(len(chars)):
            flipped = chars.copy()
            flipped[i] = {"(": ")", ")": "(", "[": "]", "]": "["}[flipped[i]]
            candidates.append("".join(flipped))
        # 2. delete one character
        for i in range(len(chars)):
            candidates.append("".join(chars[:i] + chars[i + 1 :]))
        # 3. duplicate one character
        for i in range(len(chars)):
            candidates.append("".join(chars[: i + 1] + chars[i:]))
        # 4. reversed string
        candidates.append("".join(reversed(chars)))
        # 5. random string of the same length
        candidates.append("".join(rng.choice(brackets) for _ in chars))

        for candidate in candidates:
            if is_not_in_dyck(candidate):
                controls.add(candidate)

    ordered = sorted(controls, key=lambda w: (len(w), w))
    return ordered[:max_controls]


def is_not_in_dyck(word: str) -> bool:
    """True when the string does NOT belong to Dyck-2."""
    round_balance = square_balance = 0
    for char in word:
        if char == "(":
            round_balance += 1
        elif char == ")":
            round_balance -= 1
        elif char == "[":
            square_balance += 1
        elif char == "]":
            square_balance -= 1
        if round_balance < 0 or square_balance < 0:
            return True
    return round_balance != 0 or square_balance != 0


def rule_set(grammar: PCFGrammar, probability_floor: float = 0.0) -> Set[Tuple[str, Tuple[str, ...]]]:
    """Set of rules (lhs, rhs) whose probability is above the given floor."""
    result: Set[Tuple[str, Tuple[str, ...]]] = set()
    for rule in grammar.rules:
        if float(rule.probability) < probability_floor:
            continue
        result.add((rule.lhs.name, tuple(str(symbol) for symbol in rule.rhs)))
    return result


def gold_rule_f1(
    induced: PCFGrammar,
    probability_floor: float = 1e-6,
) -> Tuple[float, float, float, Dict[str, object]]:
    """
    Precision / Recall / F1 between the induced rules and the gold rules, evaluated under
    the best non-terminal alignment (exhaustive search, feasible for small K:
    K = 5 -> 120 permutations). The alignment is required because the induced NT names
    carry no semantic meaning: without it the metric would be arbitrary.
    Returns (precision, recall, f1, details).
    """
    gold_rules = rule_set(gold_dyck2_grammar())
    induced_rules = rule_set(induced, probability_floor)
    induced_nts = list(induced.non_terminals)
    gold_nts = list(GOLD_NON_TERMINALS)

    # The alignment must map the INDUCED non-terminals onto the GOLD ones. Permuting only
    # the induced names (the historical bug) yielded F1 = 0 for every grammar whose names
    # differed from the gold ones, which made the metric useless under `--init random`.
    if len(induced_nts) <= len(gold_nts):
        # Injective induced -> gold maps: P(|gold|, |induced|) possibilities.
        candidate_maps = (
            dict(zip(induced_nts, perm))
            for perm in permutations(gold_nts, len(induced_nts))
        )
    else:
        assignments = len(gold_nts) ** len(induced_nts)
        if assignments > 400_000:
            raise ValueError(
                "gold_rule_f1: exhaustive search too large "
                f"(|induced| = {len(induced_nts)}, |gold| = {len(gold_nts)})"
            )
        candidate_maps = (
            dict(zip(induced_nts, combo))
            for combo in product(gold_nts, repeat=len(induced_nts))
        )

    best = (0.0, 0.0, 0.0, {})
    for mapping in candidate_maps:

        def translate(rule: Tuple[str, Tuple[str, ...]]) -> Tuple[str, Tuple[str, ...]]:
            lhs, rhs = rule
            rhs_mapped = tuple(
                mapping[name] if name in mapping else name for name in rhs
            )
            return mapping[lhs], rhs_mapped

        remapped = {translate(rule) for rule in induced_rules}
        true_positive = len(remapped & gold_rules)
        precision = true_positive / max(len(remapped), 1)
        recall = true_positive / max(len(gold_rules), 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        if f1 > best[2]:
            best = (
                precision,
                recall,
                f1,
                {
                    "alignment": mapping,
                    "gold_rules": len(gold_rules),
                    "induced_rules": len(remapped),
                    "matched": true_positive,
                    "missing": sorted(gold_rules - remapped),
                    "spurious": sorted(remapped - gold_rules),
                },
            )
    return best
