"""Event-driven message bus and dynamic node spawning for infinite node scalability.

Supports:
  1. Decoupled asynchronous and synchronous publish/subscribe on arbitrary topics.
  2. Directed channel routing between decoupled nodes.
  3. Dynamic spawn-on-demand of sub-agents and generator nodes (N -> infinity).

Adheres to English-only, ASCII-only rules and avoids apostrophes.
"""

from typing import Dict, List, Callable, Any, Optional
import queue
import threading
import uuid

from .schema import Envelope, OrchestratorError


class EventBus:
    """Decoupled topic-based event bus for multi-agent streaming."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Envelope], None]]] = {}
        self._message_history: List[Envelope] = []
        self._lock = threading.RLock()

    def subscribe(self, topic: str, handler: Callable[[Envelope], None]) -> None:
        """Registers a handler callback for a specific topic."""
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Envelope], None]) -> None:
        """Removes a handler callback from a topic."""
        with self._lock:
            if topic in self._subscribers and handler in self._subscribers[topic]:
                self._subscribers[topic].remove(handler)

    def publish(self, topic: str, envelope: Envelope) -> None:
        """Publishes an envelope to all subscribers of a topic."""
        with self._lock:
            self._message_history.append(envelope)
            handlers = list(self._subscribers.get(topic, []))
            # Also notify wildcard subscribers if present
            wildcard_handlers = list(self._subscribers.get("*", []))

        for h in handlers:
            h(envelope)
        for h in wildcard_handlers:
            h(envelope)

    def get_history(self) -> List[Envelope]:
        """Returns recorded envelopes published through this bus."""
        with self._lock:
            return list(self._message_history)

    def clear(self) -> None:
        """Clears subscribers and message history."""
        with self._lock:
            self._subscribers.clear()
            self._message_history.clear()


class DynamicNodeSpawner:
    """Generative factory for spawning sub-agents dynamically on demand."""

    def __init__(self, base_role: str = "worker", parent_id: Optional[str] = None):
        self.base_role = base_role
        self.parent_id = parent_id
        self._spawned_count = 0

    def spawn_sub_node(self, task_descriptor: str) -> Dict[str, Any]:
        """Instantiates metadata for a new sub-agent node to handle task complexity."""
        self._spawned_count += 1
        node_id = f"sub_node_{self.parent_id or 'gen'}_{self._spawned_count}_{uuid.uuid4().hex[:6]}"
        return {
            "node_id": node_id,
            "role": f"{self.base_role}_subagent",
            "transport": "local",
            "parent_id": self.parent_id,
            "task": task_descriptor,
            "specs": {
                "grammar": "grammar.gf",
                "ontology": "ontology.smt2",
                "seed": "seed_payload.json",
            },
        }
