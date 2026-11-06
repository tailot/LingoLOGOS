import json
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set
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
            log_rules = parser_model.get_log_rules(temperature).cpu()
            log_terms = parser_model.get_log_terms(temperature).cpu()
            p_rules = torch.exp(log_rules)
            p_terms = torch.exp(log_terms)

            extracted_rules: List[ProductionRule] = []

            # 1. Binary rules: A -> B C
            num_nt = len(nt_labels)
            for a in range(num_nt):
                lhs = Symbol(name=nt_labels[a], is_terminal=False)
                for b in range(num_nt):
                    for c in range(num_nt):
                        prob = p_rules[a, b, c].item()
                        if prob >= prune_threshold:
                            rhs = [
                                Symbol(name=nt_labels[b], is_terminal=False),
                                Symbol(name=nt_labels[c], is_terminal=False),
                            ]
                            extracted_rules.append(
                                ProductionRule(lhs=lhs, rhs=rhs, probability=prob)
                            )

            # 2. Terminal rules: A -> a
            num_terms = len(vocab_labels)
            for a in range(num_nt):
                lhs = Symbol(name=nt_labels[a], is_terminal=False)
                for tok_idx in range(num_terms):
                    prob = p_terms[a, tok_idx].item()
                    if prob >= prune_threshold:
                        rhs = [
                            Symbol(name=vocab_labels[tok_idx], is_terminal=True)
                        ]
                        extracted_rules.append(
                            ProductionRule(lhs=lhs, rhs=rhs, probability=prob)
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

    def derive(self, sequence: List[str]) -> Optional[Dict[str, Any]]:
        """
        Membership by parsing (discrete CYK) over the CNF grammar.
        Returns a derivation tree (worst case O(n^3 * |R|)) when the sequence
        is generable from the start symbol, otherwise None.
        The tree is the constructive proof of membership, so it also acts as an
        observable counterexample in the language soundness checks.
        """
        n = len(sequence)
        if n == 0 or self.start_symbol not in self.non_terminals:
            return None

        index = self.productions_by_lhs()
        terminal_rules: Dict[str, List[Tuple[str, ProductionRule]]] = {}
        binary_rules: Dict[str, List[Tuple[str, str, ProductionRule]]] = {}
        for lhs, rules in index.items():
            for rule in rules:
                if len(rule.rhs) == 1 and rule.rhs[0].is_terminal:
                    terminal_rules.setdefault(lhs, []).append((rule.rhs[0].name, rule))
                elif len(rule.rhs) == 2 and all(not s.is_terminal for s in rule.rhs):
                    binary_rules.setdefault(lhs, []).append(
                        (rule.rhs[0].name, rule.rhs[1].name, rule)
                    )

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
        data = self.to_dict()
        res = json.dumps(data, indent=indent)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(res)
        return res

    def to_yaml(self, filepath: Optional[str] = None) -> str:
        data = self.to_dict()
        res = yaml.dump(data, sort_keys=False)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(res)
        return res

    def to_gf(self, module_name: str = "DynamicGrammarIta", filepath: Optional[str] = None) -> str:
        """
        Generates an abstract plus concrete Grammatical Framework grammar (.gf).
        """
        lines = [
            f"-- DynamicGrammar generated by Neuro-Symbolic Grammar Induction Engine",
            f"abstract {module_name} = {{",
            f"  cat",
        ]
        # Declare the non-terminal categories
        for nt in self.non_terminals:
            lines.append(f"    {nt} ;")

        lines.append("  fun")
        # One function per production rule
        for idx, rule in enumerate(self.rules):
            fname = f"rule_{idx}_{rule.lhs.name}"
            rhs_cats = [s.name for s in rule.rhs if not s.is_terminal]
            if rhs_cats:
                sig = " -> ".join(rhs_cats + [rule.lhs.name])
            else:
                sig = rule.lhs.name
            lines.append(f"    {fname} : {sig} ;")

        lines.append("}")
        lines.append("")
        lines.append(f"concrete {module_name}Eng of {module_name} = {{")
        lines.append("  lincat")
        for nt in self.non_terminals:
            lines.append(f"    {nt} = Str ;")

        lines.append("  lin")
        for idx, rule in enumerate(self.rules):
            fname = f"rule_{idx}_{rule.lhs.name}"
            args = []
            rhs_lin = []
            arg_counter = 0
            for s in rule.rhs:
                if s.is_terminal:
                    escaped_term = s.name.replace('"', '\\"')
                    rhs_lin.append(f'"{escaped_term}"')
                else:
                    arg_name = f"x{arg_counter}"
                    arg_counter += 1
                    args.append(arg_name)
                    rhs_lin.append(arg_name)
            args_str = f" {' '.join(args)}" if args else ""
            body_str = " ++ ".join(rhs_lin)
            lines.append(f"    {fname}{args_str} = {body_str} ;")

        lines.append("}")

        res = "\n".join(lines) + "\n"
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(res)
        return res

    def to_lark(self) -> str:
        """
        Generates the EBNF specification for the Lark parser, used for syntactic validation.
        """
        lines = []
        rules_by_lhs: Dict[str, List[str]] = {}
        for rule in self.rules:
            lhs = rule.lhs.name.lower()
            rhs_parts = []
            for s in rule.rhs:
                if s.is_terminal:
                    escaped = s.name.replace('"', '\\"')
                    rhs_parts.append(f'"{escaped}"')
                else:
                    rhs_parts.append(s.name.lower())
            rhs_expr = " ".join(rhs_parts)
            rules_by_lhs.setdefault(lhs, []).append(rhs_expr)

        start = self.start_symbol.lower()
        if start in rules_by_lhs:
            lines.append(f"?start: {start}")
            for lhs, prods in rules_by_lhs.items():
                body = " | ".join(prods)
                lines.append(f"{lhs}: {body}")
        lines.append('%import common.WS')
        lines.append('%ignore WS')
        return "\n".join(lines) + "\n"
