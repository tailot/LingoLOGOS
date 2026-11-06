from typing import List, Dict, Any, Tuple, Optional
import lark
import z3

from .grammar import PCFGrammar
from .parser import DifferentiableCYKParser


class GrammarVerifier:
    """
    Modulo di Collaudo e Verifica Formale (Sezione 9 della Specifica):
    1. Static Syntax & Grammar Compilation Check (0 Allucinazioni Sintattiche con Lark/GF)
    2. Corpus Coverage Check (Coverage Rate = 100%, Yield(T*) = s)
    3. Z3 SMT Model Checking per Soundness logica (Solver.check(P and not G) == UNSAT)
    """

    def __init__(self, grammar: PCFGrammar):
        self.grammar = grammar

    def verify_lark_syntax(self) -> Tuple[bool, Optional[str]]:
        """
        Verifica che la grammatica discreta estratta sia compilabile come grammatica LALR/Earley
        senza errori sintattici (0 Allucinazioni Sintattiche).
        """
        ebnf = self.grammar.to_lark()
        try:
            lark_parser = lark.Lark(ebnf, parser="earley", start=self.grammar.start_symbol.lower())
            return True, None
        except Exception as e:
            return False, str(e)

    def verify_corpus_coverage(self, sequences: List[str]) -> Tuple[float, List[str]]:
        """
        Verifica la copertura del corpus: Yield(T*) = s per ogni frase s in D.
        Restituisce la percentuale di copertura e l'elenco di stringhe fallite.
        """
        ebnf = self.grammar.to_lark()
        try:
            lark_parser = lark.Lark(ebnf, parser="earley", start=self.grammar.start_symbol.lower())
        except Exception as e:
            return 0.0, sequences

        successful = 0
        failed = []
        for s in sequences:
            # Riformatta la stringa per Lark con spazi tra i caratteri terminali se necessario
            spaced = " ".join(list(s))
            try:
                tree = lark_parser.parse(spaced)
                if tree is not None:
                    successful += 1
                else:
                    failed.append(s)
            except Exception:
                failed.append(s)

        coverage = successful / max(len(sequences), 1)
        return coverage, failed

    @staticmethod
    def verify_smt_soundness(
        ast: Dict[str, Any],
        expected_balanced: bool = True,
    ) -> Tuple[bool, str, Any]:
        """
        Verification Loop con Solver SMT (Z3 Model Checking):
        Mappa l'albero sintattico (AST) o la derivazione grammaticale in formule FOL.
        Formula:
          P (Premesse): vincoli strutturali dei token parentetici (bilanciamento contatori, nesting >= 0).
          G (Goal/Tesi): bilanciamento finale uguale a 0 e validità dell'albero indotto.
        Verifica Solver.check(P and not G) -> UNSAT dimostra formalmente che G deriva necessariamente da P.
        """
        solver = z3.Solver()

        def flatten_yield(node: Dict[str, Any]) -> List[str]:
            if "token" in node:
                return [node["token"]]
            res = []
            for child in node.get("children", []):
                res.extend(flatten_yield(child))
            return res

        tokens = flatten_yield(ast)
        n = len(tokens)

        # Tracciamento contatore di profondità tonda e quadra per ogni prefisso
        # round_cnt_i, square_cnt_i
        round_vars = [z3.Int(f"round_{i}") for i in range(n + 1)]
        square_vars = [z3.Int(f"square_{i}") for i in range(n + 1)]

        # Premesse P:
        # round_0 == 0, square_0 == 0
        solver.add(round_vars[0] == 0)
        solver.add(square_vars[0] == 0)

        # Transizioni step by step secondo i token dell'AST
        for i, tok in enumerate(tokens):
            if tok == "(":
                solver.add(round_vars[i + 1] == round_vars[i] + 1)
                solver.add(square_vars[i + 1] == square_vars[i])
            elif tok == ")":
                solver.add(round_vars[i + 1] == round_vars[i] - 1)
                solver.add(square_vars[i + 1] == square_vars[i])
            elif tok == "[":
                solver.add(round_vars[i + 1] == round_vars[i])
                solver.add(square_vars[i + 1] == square_vars[i] + 1)
            elif tok == "]":
                solver.add(round_vars[i + 1] == round_vars[i])
                solver.add(square_vars[i + 1] == square_vars[i] - 1)
            else:
                solver.add(round_vars[i + 1] == round_vars[i])
                solver.add(square_vars[i + 1] == square_vars[i])

            # Invariante di non-negatività (well-bracketed)
            solver.add(round_vars[i + 1] >= 0)
            solver.add(square_vars[i + 1] >= 0)

        # Tesi G: A fine stringa, i contatori sono bilanciati a 0
        if expected_balanced:
            thesis = z3.And(round_vars[n] == 0, square_vars[n] == 0)
        else:
            thesis = z3.Or(round_vars[n] != 0, square_vars[n] != 0)

        # Test di consistenza logica formale: Solver.check(P and not G)
        solver.add(z3.Not(thesis))
        res = solver.check()

        if res == z3.unsat:
            return True, "UNSAT", None
        elif res == z3.sat:
            return False, "SAT", solver.model()
        else:
            return False, "UNKNOWN", None
