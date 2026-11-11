"""SMT-LIB 2 constraint verification and ontology integration with Z3 SMT Solver.

Part of RFC LingoLOGOS-MULTI-ORCH-v1.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import z3

from .schema import SMTConstraints, SMTVerificationError


@dataclass
class SMTVerificationResult:
    """Outcome of SMT satisfiability check performed by the orchestrator."""
    status: str
    is_sat: bool
    is_unsat: bool
    error_message: Optional[str] = None
    model_str: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "is_sat": self.is_sat,
            "is_unsat": self.is_unsat,
            "error_message": self.error_message,
            "model_str": self.model_str,
            "details": self.details,
        }


class Z3Verifier:
    """Verifies SMT-LIB 2 constraints against ontology declarations using Z3."""

    SORT_PATTERN = re.compile(r"^\s*\(\s*declare-sort\s+([A-Za-z0-9_!@$%^&*+=<>.?-]+)", re.IGNORECASE)
    CONST_PATTERN = re.compile(r"^\s*\(\s*declare-const\s+([A-Za-z0-9_!@$%^&*+=<>.?-]+)", re.IGNORECASE)
    FUN_PATTERN = re.compile(r"^\s*\(\s*declare-fun\s+([A-Za-z0-9_!@$%^&*+=<>.?-]+)", re.IGNORECASE)

    def __init__(self, smt_context: Optional["z3.Context"] = None):
        """Creates a verifier, optionally bound to a dedicated Z3 context.

        The implicit Z3 main context is shared process wide and is not safe to parse into from
        several threads at the same time. A pooled verification worker therefore gets its own
        context, which keeps concurrent ontology checks of different nodes fully isolated.
        """
        self.smt_context = smt_context

    @classmethod
    def extract_declared_identifiers(cls, smt_text: str) -> Set[str]:
        """Extracts declared sort, constant, and function names from SMT-LIB 2 text."""
        declared: Set[str] = set()
        for line in smt_text.splitlines():
            clean = line.strip()
            if not clean or clean.startswith(";"):
                continue
            m_sort = cls.SORT_PATTERN.search(clean)
            if m_sort:
                declared.add(m_sort.group(1))
                continue
            m_const = cls.CONST_PATTERN.search(clean)
            if m_const:
                declared.add(m_const.group(1))
                continue
            m_fun = cls.FUN_PATTERN.search(clean)
            if m_fun:
                declared.add(m_fun.group(1))
                continue
        return declared

    @classmethod
    def filter_duplicate_declarations(
        cls,
        declarations: List[str],
        already_declared: Set[str],
    ) -> Tuple[List[str], Set[str]]:
        """Filters out declarations for symbols that have already been declared."""
        filtered: List[str] = []
        current_declared = set(already_declared)

        for decl in declarations:
            clean = decl.strip()
            if not clean or clean.startswith(";"):
                continue

            m_sort = cls.SORT_PATTERN.search(clean)
            if m_sort:
                name = m_sort.group(1)
                if name in current_declared:
                    continue
                current_declared.add(name)
                filtered.append(decl)
                continue

            m_const = cls.CONST_PATTERN.search(clean)
            if m_const:
                name = m_const.group(1)
                if name in current_declared:
                    continue
                current_declared.add(name)
                filtered.append(decl)
                continue

            m_fun = cls.FUN_PATTERN.search(clean)
            if m_fun:
                name = m_fun.group(1)
                if name in current_declared:
                    continue
                current_declared.add(name)
                filtered.append(decl)
                continue

            # If not matching known declaration patterns, keep it
            filtered.append(decl)

        return filtered, current_declared

    def verify_constraints(
        self,
        constraints: SMTConstraints,
        ontology_smt2: Optional[str] = None,
    ) -> SMTVerificationResult:
        """Verifies SMT constraints combined with optional ontology declarations."""
        solver = z3.Solver(ctx=self.smt_context) if self.smt_context is not None else z3.Solver()
        smt_segments: List[str] = []

        ontology_symbols: Set[str] = set()
        if ontology_smt2 and ontology_smt2.strip():
            smt_segments.append(ontology_smt2)
            ontology_symbols = self.extract_declared_identifiers(ontology_smt2)

        # Filter payload declarations to avoid duplicate declaration errors in Z3
        filtered_decls, _ = self.filter_duplicate_declarations(
            constraints.declarations,
            ontology_symbols,
        )

        smt_segments.extend(filtered_decls)
        smt_segments.extend(constraints.assertions)

        full_smt_program = "\n".join(smt_segments)

        try:
            solver.from_string(full_smt_program)
            check_status = solver.check()

            if check_status == z3.sat:
                model_str = None
                try:
                    model = solver.model()
                    model_str = str(model)
                except Exception:
                    pass
                return SMTVerificationResult(
                    status="sat",
                    is_sat=True,
                    is_unsat=False,
                    model_str=model_str,
                    details={"assertions_count": len(constraints.assertions)},
                )
            elif check_status == z3.unsat:
                return SMTVerificationResult(
                    status="unsat",
                    is_sat=False,
                    is_unsat=True,
                    details={"assertions_count": len(constraints.assertions)},
                )
            else:
                return SMTVerificationResult(
                    status="unknown",
                    is_sat=False,
                    is_unsat=False,
                    details={"solver_reason": solver.reason_unknown()},
                )
        except Exception as exc:
            return SMTVerificationResult(
                status="error",
                is_sat=False,
                is_unsat=False,
                error_message=str(exc),
                details={"smt_code": full_smt_program},
            )
