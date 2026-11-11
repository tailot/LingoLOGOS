"""Centralized SMT Solver Pool for distributed verification offloading.

Allows lightweight grammar nodes to stream assertions and formulas to
a pooled cluster of dedicated Z3 SMT solver instances.

Adheres to English-only, ASCII-only rules and avoids apostrophes.
"""

from typing import List, Dict, Any, Optional
import concurrent.futures
from dataclasses import dataclass

import z3

from .schema import SMTConstraints
from .verifier import Z3Verifier, SMTVerificationResult


@dataclass
class SolverTask:
    """A task representing formal constraints to verify against an ontology."""
    task_id: str
    constraints: SMTConstraints
    ontology_smt2: str = ""
    timeout_ms: int = 10000


class SMTSolverPool:
    """Pool of worker threads executing Z3 constraint satisfiability checks concurrently."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def verify_single(self, task: SolverTask) -> SMTVerificationResult:
        """Executes a single SMT verification task inside a dedicated Z3 context.

        Each worker thread parses the ontology of one node into its own solver context, so the
        verifications of concurrent tasks never touch the shared Z3 main context.
        """
        verifier = Z3Verifier(smt_context=z3.Context())
        return verifier.verify_constraints(
            constraints=task.constraints,
            ontology_smt2=task.ontology_smt2,
        )

    def verify_batch(self, tasks: List[SolverTask]) -> Dict[str, SMTVerificationResult]:
        """Dispatches a list of verification tasks across the worker pool concurrently."""
        futures = {
            self.executor.submit(self.verify_single, t): t.task_id
            for t in tasks
        }
        results: Dict[str, SMTVerificationResult] = {}
        for future in concurrent.futures.as_completed(futures):
            t_id = futures[future]
            try:
                results[t_id] = future.result()
            except Exception as exc:
                results[t_id] = SMTVerificationResult(
                    status="error",
                    is_sat=False,
                    is_unsat=False,
                    error_message=str(exc),
                )
        return results

    def shutdown(self, wait: bool = True) -> None:
        """Shuts down the thread pool executor."""
        self.executor.shutdown(wait=wait)

    def is_shutdown(self) -> bool:
        """Reports whether the executor was released and no longer accepts verification tasks."""
        return bool(getattr(self.executor, "_shutdown", False))

