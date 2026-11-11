"""Dataset D4 generator: Syntactic Robustness and Negative Controls (Dyck-2 and Anti-Hallucination).

Specifications:
- Triple: (Corrupted_Sequence, Error_Type_Tag, Corrected_Sequence)
- Dedicated N/2 samples to Dyck-2 sequences with arbitrary nesting depth to test internal stack counting.
- Controlled noise injection: OOV tokens, delimiter imbalance, arity violations, and type mismatches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from project5.common import DatasetSample, init_seed

ERROR_TAGS = ("OOV_TOKEN", "DELIMITER_IMBALANCE", "ARITY_VIOLATION", "TYPE_MISMATCH")


@dataclass
class D4Sample(DatasetSample):
    """Schema for D4 record."""
    corrupted_sequence: str
    error_type_tag: str
    corrected_sequence: str
    is_dyck2: bool
    nesting_depth: int


class D4Generator:
    """Generator for D4 Syntactic Robustness & Anti-Hallucination."""

    def __init__(self, seed: int = 42):
        self.rng = init_seed(seed)

    def _generate_valid_dyck2(self, max_pairs: int = 8) -> Tuple[str, int]:
        """Generate a valid Dyck-2 sequence and compute its maximum nesting depth."""
        num_pairs = self.rng.randint(1, max_pairs)
        chars: List[str] = []
        stack: List[str] = []
        max_depth = 0
        current_depth = 0

        pairs = [("(", ")"), ("[", "]")]

        for _ in range(num_pairs):
            open_c, close_c = self.rng.choice(pairs)
            chars.append(open_c)
            stack.append(close_c)
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth

            # Randomly close some open brackets
            while stack and self.rng.random() < 0.4:
                chars.append(stack.pop())
                current_depth -= 1

        while stack:
            chars.append(stack.pop())

        return "".join(chars), max_depth

    def _corrupt_dyck2(self, clean_seq: str, max_depth: int) -> Tuple[str, str, str, int]:
        error_type = self.rng.choice(ERROR_TAGS)
        if error_type == "OOV_TOKEN":
            # Insert invalid bracket or symbol
            pos = self.rng.randint(0, len(clean_seq))
            oov_char = self.rng.choice(["{", "}", "@", "$", "#", "!"])
            corrupted = clean_seq[:pos] + oov_char + clean_seq[pos:]
            return corrupted, error_type, clean_seq, max_depth

        elif error_type == "DELIMITER_IMBALANCE":
            # Drop a closing bracket or add an extra unmatched open bracket
            if len(clean_seq) > 2 and self.rng.random() < 0.5:
                # Remove random character
                pos = self.rng.randint(0, len(clean_seq) - 1)
                corrupted = clean_seq[:pos] + clean_seq[pos + 1:]
            else:
                extra = self.rng.choice(["(", "["])
                corrupted = clean_seq + extra
            return corrupted, error_type, clean_seq, max_depth

        elif error_type == "ARITY_VIOLATION":
            # For Dyck-2, simulate arity mismatch by wrapping incomplete pair
            corrupted = clean_seq + "()" + clean_seq[0] if clean_seq else "((("
            return corrupted, error_type, clean_seq, max_depth

        else:  # TYPE_MISMATCH
            # Swap a paren with a bracket causing bracket mismatch e.g. ( ]
            if "(" in clean_seq:
                pos = clean_seq.find("(")
                corrupted = clean_seq[:pos] + "]" + clean_seq[pos + 1:]
            elif "[" in clean_seq:
                pos = clean_seq.find("[")
                corrupted = clean_seq[:pos] + ")" + clean_seq[pos + 1:]
            else:
                corrupted = clean_seq + "]"
            return corrupted, error_type, clean_seq, max_depth

    def _generate_command_control(self) -> Tuple[str, str, str]:
        """Generate high-level command negative control with controlled noise."""
        valid_templates = [
            ("set_thermostat(temp=22, mode='auto')", "set_thermostat"),
            ("send_alert(level='critical', recipient='admin')", "send_alert"),
            ("schedule_task(task_id=101, delay_sec=30)", "schedule_task"),
            ("verify_hash(block_id=99, expected='a1b2c3')", "verify_hash"),
        ]
        clean, tag_name = self.rng.choice(valid_templates)
        error_type = self.rng.choice(ERROR_TAGS)

        if error_type == "OOV_TOKEN":
            corrupted = clean.replace("=", " :=? ")
        elif error_type == "DELIMITER_IMBALANCE":
            corrupted = clean.replace(")", "")
        elif error_type == "ARITY_VIOLATION":
            corrupted = clean.replace(")", ", extraneous_arg_unsupported=999)")
        else:  # TYPE_MISMATCH
            corrupted = clean.replace("temp=22", "temp='twenty_two_celsius'").replace("delay_sec=30", "delay_sec='half_minute'")
            if corrupted == clean:
                corrupted = clean.replace("level='critical'", "level=999999")

        return corrupted, error_type, clean

    def generate_sample(self, index: int) -> D4Sample:
        # N / 2 dedicated to Dyck-2 sequences
        is_dyck2 = (index % 2 == 0)

        if is_dyck2:
            clean, depth = self._generate_valid_dyck2(max_pairs=8)
            corrupted, tag, corrected, real_depth = self._corrupt_dyck2(clean, depth)
            nesting_depth = real_depth
        else:
            corrupted, tag, corrected = self._generate_command_control()
            nesting_depth = 1

        return D4Sample(
            dataset_id="D4_SYNTACTIC_ROBUSTNESS",
            sample_id=f"D4_{index:07d}",
            corrupted_sequence=corrupted,
            error_type_tag=tag,
            corrected_sequence=corrected,
            is_dyck2=is_dyck2,
            nesting_depth=nesting_depth,
        )

    def generate_stream(self, total: int) -> Iterator[D4Sample]:
        for i in range(total):
            yield self.generate_sample(i)
