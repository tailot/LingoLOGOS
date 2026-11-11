"""Dataset D2 generator: Ontological Consistency and SMT Constraints (Z3 Integration).

Specifications:
- Formal triple: (Query_Semantica, SMT_Assertions_Block, Expected_Satisfiability_Result)
- Strict 50/50 balance between SAT and UNSAT instances.
- Injection of ontological axioms from seed_payload.json (Allen temporal intervals, partial orders, arithmetic inequalities).
- Stress-testing via finite numerical domain bounds.
- Validated via z3.Solver.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import z3

from project5.common import DatasetSample, init_seed


@dataclass
class D2Sample(DatasetSample):
    """Schema for D2 record."""
    semantic_query: str
    smt_assertions_block: str
    expected_satisfiability_result: str  # "SAT" or "UNSAT"
    verification_passed: bool


class D2Generator:
    """Generator for D2 Ontological Consistency & SMT Constraints."""

    def __init__(self, seed: int = 42, validate_with_z3: bool = True):
        self.rng = init_seed(seed)
        self.validate_with_z3 = validate_with_z3

    def _generate_allen_interval_clause(self, is_sat: bool) -> Tuple[str, str, str]:
        """Generate Allen interval relation query and SMT block."""
        start_a = self.rng.randint(0, 50)
        dur_a = self.rng.randint(5, 20)
        end_a = start_a + dur_a

        if is_sat:
            # interval B starts strictly after interval A ends (Allen 'before')
            offset = self.rng.randint(1, 15)
            start_b = end_a + offset
            dur_b = self.rng.randint(5, 20)
            end_b = start_b + dur_b
            query = f"Verify temporal precedence: Interval A [{start_a}, {end_a}] occurs strictly before Interval B [{start_b}, {end_b}]."
            smt_block = (
                f"(declare-const a_start Int)\n"
                f"(declare-const a_end Int)\n"
                f"(declare-const b_start Int)\n"
                f"(declare-const b_end Int)\n"
                f"(assert (= a_start {start_a}))\n"
                f"(assert (= a_end {end_a}))\n"
                f"(assert (= b_start {start_b}))\n"
                f"(assert (= b_end {end_b}))\n"
                f"(assert (< a_end b_start))\n"
                f"(check-sat)"
            )
            expected = "SAT"
        else:
            # UNSAT: assert that interval A is both before B and overlapping/after B
            offset = self.rng.randint(1, 15)
            start_b = end_a + offset
            dur_b = self.rng.randint(5, 20)
            end_b = start_b + dur_b
            query = f"Verify impossible temporal ordering: Interval A [{start_a}, {end_a}] strictly before Interval B [{start_b}, {end_b}] and B before A."
            smt_block = (
                f"(declare-const a_start Int)\n"
                f"(declare-const a_end Int)\n"
                f"(declare-const b_start Int)\n"
                f"(declare-const b_end Int)\n"
                f"(assert (= a_start {start_a}))\n"
                f"(assert (= a_end {end_a}))\n"
                f"(assert (= b_start {start_b}))\n"
                f"(assert (= b_end {end_b}))\n"
                f"(assert (< a_end b_start))\n"
                f"(assert (< b_end a_start))\n"
                f"(check-sat)"
            )
            expected = "UNSAT"

        return query, smt_block, expected

    def _generate_partial_order_clause(self, is_sat: bool) -> Tuple[str, str, str]:
        """Generate partial order and arithmetic inequality query and SMT block."""
        bound = self.rng.randint(10, 100)
        var_x = "node_priority_x"
        var_y = "node_priority_y"

        if is_sat:
            delta = self.rng.randint(1, 20)
            query = f"Check partial order: {var_x} < {var_y} with upper bound {bound}."
            smt_block = (
                f"(declare-const {var_x} Int)\n"
                f"(declare-const {var_y} Int)\n"
                f"(assert (< {var_x} {var_y}))\n"
                f"(assert (<= {var_y} {bound}))\n"
                f"(assert (> {var_x} 0))\n"
                f"(check-sat)"
            )
            expected = "SAT"
        else:
            query = f"Check cycle contradiction in strict partial order: {var_x} < {var_y} and {var_y} < {var_x}."
            smt_block = (
                f"(declare-const {var_x} Int)\n"
                f"(declare-const {var_y} Int)\n"
                f"(assert (< {var_x} {var_y}))\n"
                f"(assert (< {var_y} {var_x}))\n"
                f"(check-sat)"
            )
            expected = "UNSAT"

        return query, smt_block, expected

    def _verify_smt(self, smt_block: str, expected: str) -> bool:
        if not self.validate_with_z3:
            return True
        solver = z3.Solver()
        # Parse smt assertions
        try:
            assertions = z3.parse_smt2_string(smt_block)
            solver.add(assertions)
            res = solver.check()
            if res == z3.sat and expected == "SAT":
                return True
            if res == z3.unsat and expected == "UNSAT":
                return True
            return False
        except Exception:
            return False

    def generate_sample(self, index: int) -> D2Sample:
        # Strict 50/50 balance: even index -> SAT, odd index -> UNSAT
        is_sat = (index % 2 == 0)

        clause_type = (index // 2) % 2
        if clause_type == 0:
            query, smt_block, expected = self._generate_allen_interval_clause(is_sat)
        else:
            query, smt_block, expected = self._generate_partial_order_clause(is_sat)

        verification_passed = self._verify_smt(smt_block, expected)

        return D2Sample(
            dataset_id="D2_SMT_CONSISTENCY",
            sample_id=f"D2_{index:07d}",
            semantic_query=query,
            smt_assertions_block=smt_block,
            expected_satisfiability_result=expected,
            verification_passed=verification_passed,
        )

    def generate_stream(self, total: int) -> Iterator[D2Sample]:
        for i in range(total):
            yield self.generate_sample(i)
