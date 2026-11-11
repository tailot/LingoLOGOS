"""Dataset D5 generator: Projection and Logit Masking (VolatileLogitsMasker).

Specifications:
- Triple: (Token_Prefix_Sequence, Active_Mask_Bitset, Next_Valid_Token_Set)
- Bitset calculation determined by current state of GF + Z3 finite-state automaton.
- Bidirectional mapping between LLM vocabulary token IDs and valid formal grammar terminal symbols.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set

from project5.common import DatasetSample, init_seed

VOCABULARY = [
    "<PAD>", "<BOS>", "<EOS>", "<UNK>",
    "(", ")", "[", "]",
    "ACTION_SCHEDULE", "ACTION_SET_TEMP", "ACTION_ALERT", "ACTION_TRANSFER",
    "PARAM_TIME", "PARAM_ZONE", "PARAM_VAL", "PARAM_RECIPIENT",
    "VAL_NUM", "VAL_STR", "COMMA", "ASSIGN",
]

TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(VOCABULARY)}
VOCAB_SIZE = len(VOCABULARY)


@dataclass
class D5Sample(DatasetSample):
    """Schema for D5 record."""
    prefix_tokens: List[str]
    prefix_token_ids: List[int]
    active_mask_bitset: str  # binary string of length VOCAB_SIZE ('1' = allowed, '0' = masked)
    next_valid_tokens: List[str]
    next_valid_token_ids: List[int]
    automaton_state: str


class D5Generator:
    """Generator for D5 Projection and Dynamic Logit Masking."""

    def __init__(self, seed: int = 42, vocab_size: int = VOCAB_SIZE):
        self.rng = init_seed(seed)
        self.vocab_size = vocab_size

    def _state_transitions(self, state: str) -> tuple[List[str], str]:
        """Compute admissible next tokens and next state given current automaton state."""
        if state == "STATE_START":
            valid = ["<BOS>", "("]
            next_state = "STATE_ACTION"
        elif state == "STATE_ACTION":
            valid = ["ACTION_SCHEDULE", "ACTION_SET_TEMP", "ACTION_ALERT", "ACTION_TRANSFER"]
            next_state = "STATE_OPEN_PARAM"
        elif state == "STATE_OPEN_PARAM":
            valid = ["("]
            next_state = "STATE_PARAM_KEY"
        elif state == "STATE_PARAM_KEY":
            valid = ["PARAM_TIME", "PARAM_ZONE", "PARAM_VAL", "PARAM_RECIPIENT"]
            next_state = "STATE_ASSIGN"
        elif state == "STATE_ASSIGN":
            valid = ["ASSIGN"]
            next_state = "STATE_VAL"
        elif state == "STATE_VAL":
            valid = ["VAL_NUM", "VAL_STR", "[", "("]
            next_state = "STATE_PARAM_CONT"
        elif state == "STATE_PARAM_CONT":
            valid = ["COMMA", ")", "]"]
            next_state = "STATE_FINISH"
        else:
            valid = ["<EOS>", ")"]
            next_state = "STATE_START"

        return valid, next_state

    def generate_sample(self, index: int) -> D5Sample:
        steps = 1 + (index % 6)
        current_state = "STATE_START"
        prefix_tokens: List[str] = ["<BOS>"]

        for _ in range(steps):
            valid_next, next_state = self._state_transitions(current_state)
            chosen = self.rng.choice(valid_next)
            prefix_tokens.append(chosen)
            current_state = next_state

        # Compute next admissible tokens from final state
        admissible_tokens, final_state = self._state_transitions(current_state)
        admissible_ids = [TOKEN_TO_ID[t] for t in admissible_tokens if t in TOKEN_TO_ID]

        # Build bitset vector
        bitset_chars = ['0'] * self.vocab_size
        for tid in admissible_ids:
            if tid < self.vocab_size:
                bitset_chars[tid] = '1'
        active_mask_bitset = "".join(bitset_chars)

        prefix_ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID["<UNK>"]) for t in prefix_tokens]

        return D5Sample(
            dataset_id="D5_LOGIT_MASKING",
            sample_id=f"D5_{index:07d}",
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_ids,
            active_mask_bitset=active_mask_bitset,
            next_valid_tokens=admissible_tokens,
            next_valid_token_ids=admissible_ids,
            automaton_state=current_state,
        )

    def generate_stream(self, total: int) -> Iterator[D5Sample]:
        for i in range(total):
            yield self.generate_sample(i)
