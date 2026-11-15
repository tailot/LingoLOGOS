import json
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch


@dataclass
class Symbol:
    name: str
    is_terminal: bool = False

    def __str__(self) -> str:
        return f"'{self.name}'" if self.is_terminal else self.name

    def __repr__(self) -> str:
        return f"'{self.name}'" if self.is_terminal else self.name


@dataclass
class ProductionRule:
    lhs: Symbol
    rhs: List[Symbol]
    probability: float
    raw_probability: Optional[float] = None

    def to_string(self) -> str:
        rhs_str = " ".join(str(s) for s in self.rhs)
        return f"{self.lhs.name} -> {rhs_str}"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "rule": self.to_string(),
            "probability": round(float(self.probability), 4),
        }
        if self.raw_probability is not None and abs(self.raw_probability - self.probability) > 1e-9:
            data["probability_pre_pruning"] = round(float(self.raw_probability), 4)
        return data


# (terminal table A -> a, binary table A -> B C) ready for the discrete CYK of derive()
CNFIndex = Tuple[
    Dict[str, List[Tuple[str, "ProductionRule"]]],
    Dict[str, List[Tuple[str, str, "ProductionRule"]]],
]


class PCFGrammar:
    """
    Probabilistic context-free grammar (PCFG) representation in Chomsky Normal
    Form (CNF). Provides export helpers for JSON, YAML and Grammatical Framework (.gf).
    """

    def __init__(
        self,
        non_terminals: List[str],
        terminals: List[str],
        start_symbol: str = "S",
        rules: Optional[List[ProductionRule]] = None,
    ):
        self.non_terminals = non_terminals
        self.terminals = terminals
        self.start_symbol = start_symbol
        self.rules: List[ProductionRule] = rules if rules is not None else []

    @classmethod
    def from_model(
        cls,
        parser_model: torch.nn.Module,
        nt_labels: List[str],
        vocab_labels: List[str],
        start_symbol: str = "S",
        prune_threshold: float = 0.01,
        temperature: float = 1.0,
        renormalize: bool = True,
    ) -> "PCFGrammar":
        """
        Extracts the discrete grammar from the parser parameters through softmax and pruning.

        After pruning the probability is renormalized per left-hand side symbol
        (renormalize=True): this is a requirement for staying a valid PCFG, because
        cutting the tails of the distribution removes mass. The original model
        probability is preserved in ProductionRule.raw_probability.
        """
        grammar = cls(
            non_terminals=nt_labels,
            terminals=vocab_labels,
            start_symbol=start_symbol,
        )

        with torch.no_grad():
            p_rules = torch.exp(parser_model.get_log_rules(temperature).cpu())
            p_terms = torch.exp(parser_model.get_log_terms(temperature).cpu())
            # One host-side conversion instead of one scalar tensor per candidate production
            rule_probs = p_rules.tolist()
            term_probs = p_terms.tolist()

        # One Symbol instance per label: the same object is shared by every production
        # mentioning it, which keeps the rule index small and the exports cheap.
        non_terminal_symbols = {
            name: Symbol(name=name, is_terminal=False) for name in nt_labels
        }
        terminal_symbols = {
            name: Symbol(name=name, is_terminal=True) for name in vocab_labels
        }

        extracted_rules: List[ProductionRule] = []

        # 1. Binary rules: A -> B C
        for a, lhs_name in enumerate(nt_labels):
            lhs = non_terminal_symbols[lhs_name]
            for b, left_name in enumerate(nt_labels):
                probs_over_c = rule_probs[a][b]
                for c, right_name in enumerate(nt_labels):
                    prob = probs_over_c[c]
                    if prob >= prune_threshold:
                        extracted_rules.append(
                            ProductionRule(
                                lhs=lhs,
                                rhs=[
                                    non_terminal_symbols[left_name],
                                    non_terminal_symbols[right_name],
                                ],
                                probability=prob,
                            )
                        )

        # 2. Terminal rules: A -> a
        for a, lhs_name in enumerate(nt_labels):
            lhs = non_terminal_symbols[lhs_name]
            for tok_idx, token_name in enumerate(vocab_labels):
                prob = term_probs[a][tok_idx]
                if prob >= prune_threshold:
                    extracted_rules.append(
                        ProductionRule(lhs=lhs, rhs=[terminal_symbols[token_name]], probability=prob)
                    )

        # Per-LHS normalization after pruning: keeps the valid-PCFG property
        # (the productions of every non-terminal must sum to 1).
        if renormalize:
            mass_by_lhs: Dict[str, float] = {}
            for rule in extracted_rules:
                mass_by_lhs[rule.lhs.name] = mass_by_lhs.get(rule.lhs.name, 0.0) + rule.probability
            for rule in extracted_rules:
                total = mass_by_lhs.get(rule.lhs.name, 0.0)
                rule.raw_probability = rule.probability
                if total > 0.0:
                    rule.probability = rule.probability / total

        grammar.rules = extracted_rules
        return grammar

    def production_mass_per_lhs(self) -> Dict[str, float]:
        """
        Sum of the probabilities of every production for each non-terminal.
        PCFG validity invariant: every value must equal 1.0.
        A non-terminal without productions has mass 0.0 (incomplete grammar).
        """
        masses: Dict[str, float] = {nt: 0.0 for nt in self.non_terminals}
        for rule in self.rules:
            masses[rule.lhs.name] = masses.get(rule.lhs.name, 0.0) + float(rule.probability)
        return masses

    def is_valid_pcfg(self, tolerance: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
        """
        Checks the distribution invariant of the discrete grammar.
        Returns (valid, masses_per_LHS).
        """
        masses = self.production_mass_per_lhs()
        offenders = {
            name: mass for name, mass in masses.items()
            if mass > 0.0 and abs(mass - 1.0) > tolerance
        }
        return (len(offenders) == 0), masses

    def productions_by_lhs(self) -> Dict[str, List[ProductionRule]]:
        """Index of the productions grouped by left-hand side non-terminal."""
        index: Dict[str, List[ProductionRule]] = {}
        for rule in self.rules:
            index.setdefault(rule.lhs.name, []).append(rule)
        return index

    def cnf_index(self) -> CNFIndex:
        """
        Productions split into the terminal table (A -> a) and the binary table (A -> B C).
        Build it once when membership is tested over many strings -- the soundness check
        probes thousands of words -- and hand it to derive() to skip the rebuild per call.
        """
        terminal_rules: Dict[str, List[Tuple[str, ProductionRule]]] = {}
        binary_rules: Dict[str, List[Tuple[str, str, ProductionRule]]] = {}
        for rule in self.rules:
            lhs = rule.lhs.name
            if len(rule.rhs) == 1 and rule.rhs[0].is_terminal:
                terminal_rules.setdefault(lhs, []).append((rule.rhs[0].name, rule))
            elif len(rule.rhs) == 2 and all(not s.is_terminal for s in rule.rhs):
                binary_rules.setdefault(lhs, []).append(
                    (rule.rhs[0].name, rule.rhs[1].name, rule)
                )
        return terminal_rules, binary_rules

    def derive(
        self,
        sequence: List[str],
        index: Optional[CNFIndex] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Membership by parsing (discrete CYK) over the CNF grammar.
        Returns a derivation tree (worst case O(n^3 * |R|)) when the sequence
        is generable from the start symbol, otherwise None.
        The tree is the constructive proof of membership, so it also acts as an
        observable counterexample in the language soundness checks.
        :param index: optional prebuilt cnf_index(), for repeated membership tests.
        """
        n = len(sequence)
        if n == 0 or self.start_symbol not in self.non_terminals:
            return None

        terminal_rules, binary_rules = self.cnf_index() if index is None else index

        # chart[i][j] = {non_terminal: derivation tree for the span sequence[i:j+1]}
        chart: List[List[Dict[str, Dict[str, Any]]]] = [
            [dict() for _ in range(n)] for _ in range(n)
        ]

        for i, token in enumerate(sequence):
            for lhs, entries in terminal_rules.items():
                for term_name, rule in entries:
                    if term_name == token:
                        chart[i][i].setdefault(
                            lhs, {"type": lhs, "token": token, "rule": rule.to_string()}
                        )
                        break

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for lhs, entries in binary_rules.items():
                    for left_nt, right_nt, rule in entries:
                        for k in range(i, j):
                            if left_nt in chart[i][k] and right_nt in chart[k + 1][j]:
                                chart[i][j][lhs] = {
                                    "type": lhs,
                                    "rule": rule.to_string(),
                                    "children": [chart[i][k][left_nt], chart[k + 1][j][right_nt]],
                                }
                                break
                        if lhs in chart[i][j]:
                            break

        return chart[0][n - 1].get(self.start_symbol)

    @staticmethod
    def _write(text: str, filepath: Optional[str]) -> str:
        """Persist the exported text when a path is given, and always return it."""
        if filepath:
            Path(filepath).write_text(text, encoding="utf-8")
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grammar_metadata": {
                "type": "Probabilistic Context-Free Grammar (PCFG)",
                "normal_form": "Chomsky Normal Form (CNF)",
                "non_terminals": self.non_terminals,
                "terminals": self.terminals,
                "start_symbol": self.start_symbol,
            },
            "production_rules": [rule.to_dict() for rule in self.rules],
        }

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        return self._write(json.dumps(self.to_dict(), indent=indent), filepath)

    def to_yaml(self, filepath: Optional[str] = None) -> str:
        return self._write(yaml.dump(self.to_dict(), sort_keys=False), filepath)

    @staticmethod
    def _literal(token: str) -> str:
        """A double quoted literal of a terminal, with backslashes and quotes escaped."""
        escaped = token.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @staticmethod
    def _rule_identifier(index: int, lhs: str) -> str:
        """Deterministic GF function name of a production (index follows the rule order)."""
        return f"rule_{index}_{lhs}"

    def to_gf(self, module_name: str = "DynamicGrammarIta", filepath: Optional[str] = None) -> str:
        """
        Generates an abstract plus concrete Grammatical Framework module (.gf).
        Every production becomes one GF function; the concrete binding concatenates the
        terminals, so the linearization reproduces the yield of the derivation.
        """
        names = [
            self._rule_identifier(index, rule.lhs.name)
            for index, rule in enumerate(self.rules)
        ]

        lines = [
            "-- DynamicGrammar generated by Neuro-Symbolic Grammar Induction Engine",
            f"abstract {module_name} = {{",
            "  cat",
        ]
        # Declare the non-terminal categories
        lines.extend(f"    {nt} ;" for nt in self.non_terminals)

        lines.append("  fun")
        # One function per production rule
        for name, rule in zip(names, self.rules):
            rhs_cats = [symbol.name for symbol in rule.rhs if not symbol.is_terminal]
            signature = " -> ".join(rhs_cats + [rule.lhs.name]) if rhs_cats else rule.lhs.name
            lines.append(f"    {name} : {signature} ;")

        lines.append("}")
        lines.append("")
        lines.append(f"concrete {module_name}Eng of {module_name} = {{")
        lines.append("  lincat")
        lines.extend(f"    {nt} = Str ;" for nt in self.non_terminals)

        lines.append("  lin")
        for name, rule in zip(names, self.rules):
            args: List[str] = []
            parts: List[str] = []
            for symbol in rule.rhs:
                if symbol.is_terminal:
                    parts.append(self._literal(symbol.name))
                else:
                    argument = f"x{len(args)}"
                    args.append(argument)
                    parts.append(argument)
            argument_list = f" {' '.join(args)}" if args else ""
            lines.append(f"    {name}{argument_list} = {' ++ '.join(parts)} ;")

        lines.append("}")

        return self._write("\n".join(lines) + "\n", filepath)

    def to_lark(self) -> str:
        """
        Generates the EBNF specification for the Lark parser, used for syntactic validation.
        """
        rules_by_lhs: Dict[str, List[str]] = {}
        for rule in self.rules:
            lhs = rule.lhs.name.lower()
            rhs_parts = [
                self._literal(symbol.name) if symbol.is_terminal else symbol.name.lower()
                for symbol in rule.rhs
            ]
            rules_by_lhs.setdefault(lhs, []).append(" ".join(rhs_parts))

        lines = []
        start = self.start_symbol.lower()
        if start in rules_by_lhs:
            lines.append(f"?start: {start}")
            for lhs, prods in rules_by_lhs.items():
                lines.append(f"{lhs}: {' | '.join(prods)}")
        lines.append("%import common.WS")
        lines.append("%ignore WS")
        return "\n".join(lines) + "\n"
