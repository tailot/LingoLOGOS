"""Incremental AST decoding engine with in-memory deterministic pruning.

Coordinates incremental AST state transitions, volatile mask updates,
and subtree pruning when non-essential nodes are encountered.
All comments and docstrings adhere to AGENTS.md rules.
"""

from typing import Any, List, Set, Protocol, Optional
import numpy as np

from .lllayer_masker import NativeASTLogitsProcessor


class IncrementalGrammarSpec(Protocol):
    """Protocol defining the grammar interface required by IncrementalASTEngine."""

    def transition(self, current_node: str, token_str: str) -> str:
        """Computes deterministic transition delta(s_t, v)."""
        ...

    def get_node_type(self, node: str) -> str:
        """Returns the AST node type category (e.g. ESSENTIAL, OPTIONAL_MODIFIER)."""
        ...

    def is_terminal(self, node: str) -> bool:
        """Returns True if the current AST state is in an end or accepting condition."""
        ...

    def get_valid_next_tokens(self, node: str) -> Set[int]:
        """Returns the set of valid next token IDs for the given state node."""
        ...

    def get_closing_tokens(self, ast_stack: List[str]) -> Set[int]:
        """Returns the set of closing/parenthesization tokens to finalize the subtree."""
        ...


class SimpleASTGrammar:
    """Standard grammar implementation for Dyck / AST parsing within lllayer."""

    def __init__(
        self,
        token_to_id: dict[str, int],
        id_to_token: dict[int, str],
        node_types: Optional[dict[str, str]] = None,
        transitions: Optional[dict[tuple[str, str], str]] = None,
        valid_tokens: Optional[dict[str, Set[int]]] = None,
        closing_tokens: Optional[Set[int]] = None,
        terminal_nodes: Optional[Set[str]] = None,
    ):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = dict(id_to_token)
        self.node_types = node_types or {}
        self.transitions = transitions or {}
        self.valid_tokens = valid_tokens or {}
        self.closing_tokens = closing_tokens or set()
        self.terminal_nodes = terminal_nodes or {"TERMINAL", "END"}

    def transition(self, current_node: str, token_str: str) -> str:
        key = (current_node, token_str)
        if key in self.transitions:
            return self.transitions[key]
        return "TERMINAL"

    def get_node_type(self, node: str) -> str:
        return self.node_types.get(node, "ESSENTIAL")

    def is_terminal(self, node: str) -> bool:
        return node in self.terminal_nodes

    def get_valid_next_tokens(self, node: str) -> Set[int]:
        if node in self.valid_tokens:
            return set(self.valid_tokens[node])
        return set(self.token_to_id.values())

    def get_closing_tokens(self, ast_stack: List[str]) -> Set[int]:
        if self.closing_tokens:
            return set(self.closing_tokens)
        return {self.token_to_id.get(")", 0)}


class MockRunnerModel:
    """Mock/sample language model runner for testing constrained decoding."""

    def __init__(self, vocab_size: int, token_to_id: dict[str, int], id_to_token: dict[int, str]):
        self.vocab_size = vocab_size
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    def get_logits(self, tokens: List[int]) -> np.ndarray:
        # Uniform or pseudo-random initial logits
        return np.ones(self.vocab_size, dtype=np.float32)

    def sample(self, masked_logits: np.ndarray) -> int:
        # Select greedy argmax among admissible unmasked tokens
        return int(np.argmax(masked_logits))

    def decode(self, token_id_or_tokens: Any) -> Any:
        if isinstance(token_id_or_tokens, (list, tuple)):
            return " ".join(self.id_to_token.get(t, f"<{t}>") for t in token_id_or_tokens)
        return self.id_to_token.get(token_id_or_tokens, f"<{token_id_or_tokens}>")


class IncrementalASTEngine:
    """Internal orchestration engine for lllayer.

    Coordinates incremental parsing, AST pruning, and volatile logits masking.
    """

    def __init__(self, grammar_spec: IncrementalGrammarSpec, logits_processor: NativeASTLogitsProcessor):
        self.grammar = grammar_spec
        self.processor = logits_processor
        self.ast_stack = ["ROOT"]
        self.is_pruning_active = False

        # Initialize processor mask with root valid tokens
        initial_tokens = self.grammar.get_valid_next_tokens("ROOT")
        self.processor.update_mask_from_ast_state(initial_tokens)

    def on_token_generated(self, token_id: int, token_str: str) -> None:
        """Advances AST parser state and updates allowed tokens in RAM."""
        next_node = self.grammar.transition(self.ast_stack[-1], token_str)
        self.ast_stack.append(next_node)

        if self._should_prune_subtree(next_node):
            self.is_pruning_active = True

        if self.is_pruning_active:
            allowed_tokens = self.grammar.get_closing_tokens(self.ast_stack)
        else:
            allowed_tokens = self.grammar.get_valid_next_tokens(next_node)

        self.processor.update_mask_from_ast_state(allowed_tokens)

    def _should_prune_subtree(self, node: str) -> bool:
        """Determines whether to trigger AST tree pruning for ancillary subtrees."""
        return self.grammar.get_node_type(node) == "OPTIONAL_MODIFIER"

    def run_session(self, prompt: str, runner_model: Any, max_steps: int = 128) -> str:
        """Executes a constrained decoding generation session."""
        try:
            tokens: List[int] = []
            steps = 0
            while not self.grammar.is_terminal(self.ast_stack[-1]) and steps < max_steps:
                logits = runner_model.get_logits(tokens)
                masked_logits = self.processor(tokens, logits)

                # Ensure finite logits exist
                if np.all(np.isneginf(masked_logits)):
                    break

                next_token = runner_model.sample(masked_logits)
                tokens.append(next_token)
                steps += 1

                token_str = runner_model.decode(next_token)
                self.on_token_generated(next_token, token_str)

            return runner_model.decode(tokens)
        finally:
            self.processor.cleanup()
