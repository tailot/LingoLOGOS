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
    DECLARATION_PATTERNS = (SORT_PATTERN, CONST_PATTERN, FUN_PATTERN)

    #: Default verification budget in milliseconds; None leaves Z3 unconstrained.
    DEFAULT_TIMEOUT_MS: Optional[int] = 10_000

    def __init__(self, smt_context: Optional["z3.Context"] = None, timeout_ms: Optional[int] = None):
        """Creates a verifier, optionally bound to a dedicated Z3 context.

        The implicit Z3 main context is shared process wide and is not safe to parse into from
        several threads at the same time. A pooled verification worker therefore gets its own
        context, which keeps concurrent ontology checks of different nodes fully isolated.

        `timeout_ms` bounds the satisfiability search: without a budget a pathological set of
        assertions can hold a turn of the session forever, because the node process timeout only
        protects nodes that run as separate processes.
        """
        self.smt_context = smt_context
        self.timeout_ms = self.DEFAULT_TIMEOUT_MS if timeout_ms is None else timeout_ms

    @classmethod
    def _declared_name(cls, line: str) -> Optional[str]:
        """Symbol declared by a line, or None when the line is not a known declaration form."""
        for pattern in cls.DECLARATION_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(1)
        return None

    @classmethod
    def extract_declared_identifiers(cls, smt_text: str) -> Set[str]:
        """Extracts declared sort, constant, and function names from SMT-LIB 2 text."""
        declared: Set[str] = set()
        for line in smt_text.splitlines():
            clean = line.strip()
            if not clean or clean.startswith(";"):
                continue
            name = cls._declared_name(clean)
            if name is not None:
                declared.add(name)
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

            name = cls._declared_name(clean)
            if name is not None:
                if name in current_declared:
                    continue
                current_declared.add(name)

            # Anything that is not a known declaration form is kept untouched
            filtered.append(decl)

        return filtered, current_declared

    @staticmethod
    def _result(
        status: str,
        is_sat: bool = False,
        is_unsat: bool = False,
        error_message: Optional[str] = None,
        model_str: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> SMTVerificationResult:
        """Single construction point of the verification verdicts."""
        return SMTVerificationResult(
            status=status,
            is_sat=is_sat,
            is_unsat=is_unsat,
            error_message=error_message,
            model_str=model_str,
            details={} if details is None else details,
        )

    def verify_constraints(
        self,
        constraints: SMTConstraints,
        ontology_smt2: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> SMTVerificationResult:
        """Verifies SMT constraints combined with optional ontology declarations."""
        solver = z3.Solver(ctx=self.smt_context) if self.smt_context is not None else z3.Solver()
        budget_ms = self.timeout_ms if timeout_ms is None else timeout_ms
        if budget_ms:
            solver.set("timeout", int(budget_ms))
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
        verdict_details = {"assertions_count": len(constraints.assertions)}
        if budget_ms:
            verdict_details["timeout_ms"] = int(budget_ms)

        try:
            solver.from_string(full_smt_program)
            check_status = solver.check()

            if check_status == z3.sat:
                try:
                    model_str: Optional[str] = str(solver.model())
                except z3.Z3Exception:
                    model_str = None
                return self._result(
                    status="sat",
                    is_sat=True,
                    model_str=model_str,
                    details=verdict_details,
                )
            if check_status == z3.unsat:
                return self._result(status="unsat", is_unsat=True, details=verdict_details)
            return self._result(
                status="unknown",
                details=dict(verdict_details, solver_reason=solver.reason_unknown()),
            )
        except Exception as exc:
            return self._result(
                status="error",
                error_message=str(exc),
                details=dict(verdict_details, smt_code=full_smt_program),
            )
