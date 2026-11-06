"""
Testing and formal verification module.

Available checks:
  1. `verify_lark_compilation()`   : the discrete grammar compiles as a Lark grammar (Earley).
  2. `verify_pcfg_distribution()`  : distribution invariant sum_alpha P(A -> alpha) = 1.
  3. `verify_corpus_coverage()`    : corpus membership (discrete CYK or Lark).
  4. `verify_negative_controls()`  : malformed strings must be REJECTED.
  5. `verify_language_soundness()` : SMT proof that L(G) is included in Dyck-k.
  6. `check_word_balance_z3()`     : check on a single string (with a consistency guard).

Why the flagship SMT check (point 5) works at GRAMMAR level and not on strings:
the previous formulation encoded the prefix non-negativity invariant AMONG THE
PREMISES P. That way, for every string whose prefixes violate the invariant, P turns out
inconsistent and P AND NOT G is trivially UNSAT: the solver claimed "100% soundness" even
on obviously malformed inputs such as `)(`, `][` or `())`. Moreover the check never
consulted the grammar, so it proved no property of the induced language.

The current check proves that L(G) is included in Dyck-k by induction on the structure of
the derivations, through an abstract certificate per non-terminal, and produces a concrete
counterexample (string + derivation tree) when the inclusion does not hold.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import z3

try:  # Lark is needed only by checks 1 and 3 (Earley variant)
    import lark  # type: ignore
except Exception:  # pragma: no cover
    lark = None  # type: ignore

from .grammar import PCFGrammar

# Effect of every terminal on the counters: (round delta, square delta)
BRACKET_EFFECT: Dict[str, Tuple[int, int]] = {
    "(": (1, 0),
    ")": (-1, 0),
    "[": (0, 1),
    "]": (0, -1),
}

STATUS_PROVED = "UNSAT"
STATUS_REFUTED = "SAT"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"


def counter_signature(tokens: Sequence[str]) -> Tuple[int, int, int, int]:
    """
    Counter signature of a concrete string:
    (delta_round, delta_square, min_round, min_square), where min_* is the minimum of
    the counter over the non-empty prefixes.
    """
    delta_round = delta_square = 0
    min_round = min_square = 0
    for token in tokens:
        d_round, d_square = BRACKET_EFFECT.get(token, (0, 0))
        delta_round += d_round
        delta_square += d_square
        min_round = min(min_round, delta_round)
        min_square = min(min_square, delta_square)
    return delta_round, delta_square, min_round, min_square


def is_well_bracketed(tokens: Sequence[str]) -> bool:
    """Operational definition of Dyck-k: final counters at zero and no negative prefix."""
    d_round, d_square, min_round, min_square = counter_signature(tokens)
    return d_round == 0 and d_square == 0 and min_round >= 0 and min_square >= 0


@dataclass
class SoundnessReport:
    """
    Outcome of the check that L(G) is included in Dyck-k.
    status: "UNSAT" (proved), "SAT" (refuted), "INCONCLUSIVE".
    """

    status: str
    sound: bool
    certificate: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    counterexample: Optional[str] = None
    derivation: Optional[Dict[str, Any]] = None
    search_bound: int = 0
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        text = f"{self.status} (sound={self.sound})"
        if self.counterexample:
            text += f" | counterexample: {self.counterexample!r}"
        if self.notes:
            text += " | " + "; ".join(self.notes)
        return text


class GrammarVerifier:
    """Testing and formal verification of the discrete grammar extracted from the neural parser."""

    def __init__(self, grammar: PCFGrammar):
        self.grammar = grammar

    # ------------------------------------------------------------------ #
    # 1. Syntactic compilation of the grammar (Lark)
    # ------------------------------------------------------------------ #
    def verify_lark_compilation(self) -> Tuple[bool, Optional[str]]:
        """
        Checks that the extracted EBNF specification compiles under Lark (Earley) without errors.
        It certifies the compilability of the grammar, not the correctness of the language:
        the semantic property is covered by verify_language_soundness().
        """
        if lark is None:
            return False, "the 'lark' package is not installed"
        ebnf = self.grammar.to_lark()
        try:
            lark.Lark(ebnf, parser="earley", start=self.grammar.start_symbol.lower())
            return True, None
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------ #
    # 2. Distribution invariant (valid PCFG)
    # ------------------------------------------------------------------ #
    def verify_pcfg_distribution(self, tolerance: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
        """
        Sum of the production probabilities of every non-terminal: it must equal 1.
        Returns (valid, masses_per_LHS).
        """
        return self.grammar.is_valid_pcfg(tolerance=tolerance)

    # ------------------------------------------------------------------ #
    # 3. Corpus coverage (membership)
    # ------------------------------------------------------------------ #
    def verify_corpus_coverage(
        self,
        sequences: List[str],
        engine: str = "cyk",
    ) -> Tuple[float, List[str]]:
        """
        Corpus coverage: for every sentence s a derivation must exist in the
        discrete grammar. Returns (percentage, failing sentences).

        engine="cyk" (default): membership through discrete CYK on the CNF grammar,
            cost O(n^3 * |R|) and independent of ambiguity.
        engine="lark": uses the Earley parser of Lark (can explode on ambiguous grammars).
        """
        if engine == "lark":
            return self._coverage_with_lark(sequences)

        failed: List[str] = []
        for s in sequences:
            if self.grammar.derive(list(s)) is None:
                failed.append(s)
        coverage = (len(sequences) - len(failed)) / max(len(sequences), 1)
        return coverage, failed

    def _coverage_with_lark(self, sequences: List[str]) -> Tuple[float, List[str]]:
        if lark is None:
            return 0.0, list(sequences)
        ebnf = self.grammar.to_lark()
        try:
            parser = lark.Lark(ebnf, parser="earley", start=self.grammar.start_symbol.lower())
        except Exception:
            return 0.0, list(sequences)

        failed: List[str] = []
        for s in sequences:
            spaced = " ".join(list(s))
            try:
                if parser.parse(spaced) is None:
                    failed.append(s)
            except Exception:
                failed.append(s)
        return (len(sequences) - len(failed)) / max(len(sequences), 1), failed

    # ------------------------------------------------------------------ #
    # 4. Negative controls: malformed strings must be rejected
    # ------------------------------------------------------------------ #
    def verify_negative_controls(self, sequences: List[str]) -> Tuple[float, List[str]]:
        """
        Measures the selectivity of the grammar on NEGATIVE controls (strings outside Dyck-k).
        Returns (rejection_rate, strings wrongly accepted).
        A value < 1.0 means the grammar generates out-of-language strings.
        """
        accepted_wrong: List[str] = []
        for s in sequences:
            if self.grammar.derive(list(s)) is not None:
                accepted_wrong.append(s)
        rejection_rate = (len(sequences) - len(accepted_wrong)) / max(len(sequences), 1)
        return rejection_rate, accepted_wrong

    # ------------------------------------------------------------------ #
    # 5. SMT proof that L(G) is included in Dyck-k
    # ------------------------------------------------------------------ #
    def find_counterexample(self, max_length: int = 8) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Exhaustive search, up to length `max_length`, of a string generated by the
        grammar but NOT in Dyck-k. The string comes with its derivation tree,
        the constructive proof of its membership in L(G).
        Returns (string, tree), or None if no witness is found within the bound.
        """
        terminals = list(self.grammar.terminals)
        if not terminals:
            return None
        for length in range(1, max_length + 1):
            for combo in product(terminals, repeat=length):
                word = list(combo)
                if is_well_bracketed(word):
                    continue
                tree = self.grammar.derive(word)
                if tree is not None:
                    return "".join(word), tree
        return None

    def verify_language_soundness(self, search_bound: int = 8) -> SoundnessReport:
        """
        Proves or refutes that L(G) is included in Dyck-k.

        Certificate (UNSAT branch): for every non-terminal A we ask for four integers
        (delta_round, delta_square, min_round, min_square) that must satisfy the
        transfer equations for ALL productions of A:

            A -> a     :  delta = effect(a)                     min = effect(a)
            A -> B C   :  delta = delta_B + delta_C             min = min(min_B, delta_B + min_C)

        If the system is satisfiable, then by induction on the derivation depth every
        string generated by A has exactly that signature: one non-terminal cannot generate
        strings with different signatures. If the start symbol then has delta = 0 and
        min >= 0, no derivation can violate the invariant: once NOT G is added the solver
        answers UNSAT and the certificate is a proof of inclusion (absence of syntactic
        hallucinations).

        The method is correct by default: an UNSAT system (non-functional signature) or a
        search bound that is too low yields INCONCLUSIVE, never a spurious PROVED.
        """
        notes: List[str] = []
        nt_order = list(self.grammar.non_terminals)
        if not nt_order:
            return SoundnessReport(STATUS_INCONCLUSIVE, False, notes=["no non-terminal declared"])

        # --- 5a. first look for a concrete counterexample (witness = string + tree) --- #
        witness = self.find_counterexample(max_length=search_bound)
        if witness is not None:
            word, tree = witness
            _, word_status, _ = self.check_word_balance_z3(list(word))
            return SoundnessReport(
                STATUS_REFUTED,
                False,
                counterexample=word,
                derivation=tree,
                search_bound=search_bound,
                notes=[
                    "witness derivable in L(G); check on the string: " + word_status,
                    f"exhaustive search up to length {search_bound}",
                ],
            )
        notes.append(f"no witness found within length {search_bound}")

        # --- 5b. per-non-terminal signature system (inductive certificate) --- #
        solver = z3.Solver()
        sig = {
            nt: (
                z3.Int(f"dr_{nt}"),
                z3.Int(f"ds_{nt}"),
                z3.Int(f"mr_{nt}"),
                z3.Int(f"ms_{nt}"),
            )
            for nt in nt_order
        }

        for rule in self.grammar.rules:
            lhs = rule.lhs.name
            if lhs not in sig:
                return SoundnessReport(
                    STATUS_INCONCLUSIVE, False, notes=[f"unknown LHS: {lhs}"]
                )
            d_r, d_s, m_r, m_s = sig[lhs]

            if len(rule.rhs) == 1 and rule.rhs[0].is_terminal:
                eff_r, eff_s = BRACKET_EFFECT.get(rule.rhs[0].name, (0, 0))
                solver.add(d_r == eff_r, d_s == eff_s, m_r == eff_r, m_s == eff_s)
            elif len(rule.rhs) == 2 and all(not s.is_terminal for s in rule.rhs):
                b_r, b_s, b_mr, b_ms = sig[rule.rhs[0].name]
                c_r, c_s, c_mr, c_ms = sig[rule.rhs[1].name]
                solver.add(d_r == b_r + c_r, d_s == b_s + c_s)
                solver.add(m_r == z3.If(b_mr <= b_r + c_mr, b_mr, b_r + c_mr))
                solver.add(m_s == z3.If(b_ms <= b_s + c_s, b_ms, b_s + c_s))
            else:
                return SoundnessReport(
                    STATUS_INCONCLUSIVE,
                    False,
                    notes=[f"production not in CNF: {rule.to_string()}"],
                )

        if solver.check() != z3.sat:
            return SoundnessReport(
                STATUS_INCONCLUSIVE,
                False,
                search_bound=search_bound,
                notes=[
                    "non-functional signature: at least one non-terminal generates strings with "
                    "incompatible counter signatures (grammar too permissive)",
                ],
            )

        def read_model(mapping: Dict[str, Any]) -> Dict[str, Tuple[int, int, int, int]]:
            out: Dict[str, Tuple[int, int, int, int]] = {}
            for nt in nt_order:
                out[nt] = (
                    mapping[sig[nt][0]].as_long(),
                    mapping[sig[nt][1]].as_long(),
                    mapping[sig[nt][2]].as_long(),
                    mapping[sig[nt][3]].as_long(),
                )
            return out

        certificate = read_model(solver.model())

        start_sig = sig.get(self.grammar.start_symbol)
        if start_sig is None:
            return SoundnessReport(
                STATUS_INCONCLUSIVE,
                False,
                notes=[f"start symbol {self.grammar.start_symbol} is not declared"],
            )
        s_r, s_s, s_mr, s_ms = start_sig
        thesis = z3.And(s_r == 0, s_s == 0, s_mr >= 0, s_ms >= 0)

        solver.add(z3.Not(thesis))
        verdict = solver.check()
        if verdict == z3.unsat:
            return SoundnessReport(
                STATUS_PROVED,
                True,
                certificate=certificate,
                search_bound=search_bound,
                notes=[
                    "valid inductive certificate for every production",
                    "no counterexample",
                ],
            )
        if verdict == z3.sat:
            return SoundnessReport(
                STATUS_REFUTED,
                False,
                certificate=read_model(solver.model()),
                search_bound=search_bound,
                notes=[
                    "certificate compatible but the thesis fails on the start symbol",
                    "no concrete witness found within the bound",
                ],
            )
        return SoundnessReport(STATUS_INCONCLUSIVE, False, notes=["solver UNKNOWN"])

    # ------------------------------------------------------------------ #
    # 6. Check on a single string (with a consistency guard)
    # ------------------------------------------------------------------ #
    @staticmethod
    def check_word_balance_z3(tokens: Sequence[str]) -> Tuple[bool, str, Optional[Any]]:
        """
        FOL encoding for a single string:
          Premises P: counter transitions imposed by the tokens (equalities).
          Thesis G: final counters at zero AND all prefixes non-negative.

        The prefix invariant belongs to the THESIS, not to the premises: had it ended up in
        the premises, P would become inconsistent for every malformed string and P AND NOT G
        would be UNSAT for anything at all (a vacuous check).
        Outcomes: "VALID" (UNSAT on P AND NOT G), "INVALID" (SAT, with a model),
        "INCONSISTENT_PREMISES" (incoherent P: no conclusion allowed).
        """
        n = len(tokens)
        solver = z3.Solver()
        round_vars = [z3.Int(f"round_{i}") for i in range(n + 1)]
        square_vars = [z3.Int(f"square_{i}") for i in range(n + 1)]

        solver.add(round_vars[0] == 0, square_vars[0] == 0)
        for i, tok in enumerate(tokens):
            d_r, d_s = BRACKET_EFFECT.get(tok, (0, 0))
            solver.add(
                round_vars[i + 1] == round_vars[i] + d_r,
                square_vars[i + 1] == square_vars[i] + d_s,
            )

        # Anti-vacuity guard: the premises must be coherent
        if solver.check() != z3.sat:
            return False, "INCONSISTENT_PREMISES", None

        thesis_parts = [round_vars[n] == 0, square_vars[n] == 0]
        thesis_parts.extend(round_vars[i] >= 0 for i in range(1, n + 1))
        thesis_parts.extend(square_vars[i] >= 0 for i in range(1, n + 1))
        thesis = z3.And(*thesis_parts)

        solver.push()
        solver.add(z3.Not(thesis))
        verdict = solver.check()
        if verdict == z3.unsat:
            solver.pop()
            return True, "VALID", None
        if verdict == z3.sat:
            model = solver.model()
            solver.pop()
            return False, "INVALID", model
        solver.pop()
        return False, "UNKNOWN", None
