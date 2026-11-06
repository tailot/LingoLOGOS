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

    def to_string(self) -> str:
        rhs_str = " ".join(str(s) for s in self.rhs)
        return f"{self.lhs.name} -> {rhs_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.to_string(),
            "probability": round(float(self.probability), 4),
        }


class PCFGrammar:
    """
    Rappresenta una grammatica formale libera da contesto probabilistica (PCFG)
    in Forma Normale di Chomsky (CNF).
    Fornisce metodi di esportazione in JSON, YAML e Grammatical Framework (.gf).
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
    ) -> "PCFGrammar":
        """
        Estrae la grammatica discreta dai parametri del parser tramite softmax e pruning.
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

            # 1. Regole binarie: A -> B C
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

            # 2. Regole terminali: A -> a
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

            # Normalizzazione opzionale per LHS dopo pruning
            grammar.rules = extracted_rules
            return grammar

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
        Genera una grammatica Grammatical Framework (.gf) astratta e concreta.
        """
        lines = [
            f"-- DynamicGrammar generated by Neuro-Symbolic Grammar Induction Engine",
            f"abstract {module_name} = {{",
            f"  cat",
        ]
        # Dichiarazione categorie non-terminali
        for nt in self.non_terminals:
            lines.append(f"    {nt} ;")

        lines.append("  fun")
        # Funzioni per ogni regola
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
        Genera la specifica EBNF per il parser Lark per validazione sintattica.
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
