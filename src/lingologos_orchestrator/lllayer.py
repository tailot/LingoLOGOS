"""CLI wrapper for isolated LingoLOGOS node instances (lllayer).

RFC identifier: LingoLOGOS-MULTI-ORCH-v1
Interface Layer: lllayer (CLI Wrapper)
Executes as an independent process and communicates strictly via standard I/O streams.
Node IDs are completely arbitrary and never hardcoded.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from .schema import Envelope, VolatileState, Payload, SMTConstraints


def load_seed_payload(node_dir: Path) -> Dict[str, Any]:
    """Loads Turn 0 seed payload from the node dataset directory."""
    seed_path = node_dir / "seed_payload.json"
    if not seed_path.is_file():
        raise FileNotFoundError(f"Missing seed_payload.json in {node_dir}")
    with open(seed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_file_if_exists(file_path: Path) -> Optional[str]:
    """Reads file text if it exists."""
    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def generate_node_response(
    node_id: str,
    role: str,
    node_dir: Path,
    incoming: Envelope,
) -> Envelope:
    """Generates the next envelope from this node given the incoming message."""
    next_turn = incoming.turn
    current_tick = incoming.volatileState.internalTimeTick

    # Check if a custom script or pre-recorded turn payload exists in node_dir
    turn_file = node_dir / f"turn_{next_turn}_payload.json"
    if turn_file.is_file():
        with open(turn_file, "r", encoding="utf-8") as f:
            custom_data = json.load(f)
            # Ensure sender and volatile time tick are correctly aligned
            custom_data["sender"] = node_id
            if incoming.sender and "recipient" not in custom_data:
                custom_data["recipient"] = incoming.sender
            custom_data["turn"] = next_turn
            if "volatileState" not in custom_data:
                custom_data["volatileState"] = {"internalTimeTick": current_tick}
            return Envelope.from_dict(custom_data)

    # Deterministic generation based on role and incoming payload
    target_recipient = incoming.sender if incoming.sender != node_id else "broadcast"

    # Default declarations and assertions based on incoming constraints
    in_constraints = incoming.payload.smtConstraints
    decls = list(in_constraints.declarations)
    asserts = list(in_constraints.assertions)
    expected_status = "sat"

    if role == "opponent":
        # Opponent might contest transitions or assert safety boundary
        natural_text = f"Node {node_id} evaluating safety constraints for proposed transition."
        ast = f"ChallengeConstraint (Transition StateA StateB) (Node {node_id})"
        if not asserts:
            asserts.append("(assert true)")
    elif role == "validator":
        # Logical arbiter validates satisfiability and formal consistency
        natural_text = f"Node {node_id} validating formal SMT consistency."
        ast = f"ValidateConsistency (SystemRoot) (Node {node_id})"
        if not asserts:
            asserts.append("(assert true)")
    else:
        # Proponent or general peer
        natural_text = f"Node {node_id} formal state update."
        ast = f"StateUpdate (Step {next_turn}) (Node {node_id})"
        if not asserts:
            asserts.append("(assert true)")

    out_constraints = SMTConstraints(
        declarations=decls,
        assertions=asserts,
        z3Status=expected_status,
    )

    out_payload = Payload(
        naturalLanguage=natural_text,
        abstractSyntaxTree=ast,
        smtConstraints=out_constraints,
    )

    return Envelope(
        sender=node_id,
        recipient=target_recipient,
        turn=next_turn,
        volatileState=VolatileState(internalTimeTick=current_tick),
        payload=out_payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LingoLOGOS CLI Process Wrapper (lllayer) for isolated node execution."
    )
    parser.add_argument("--node-id", required=True, help="Arbitrary node identifier.")
    parser.add_argument("--role", default="node", help="Role of the node (e.g. proponent, opponent, validator).")
    parser.add_argument("--node-dir", required=True, help="Path to the node dataset directory.")
    args = parser.parse_args()

    node_id = args.node_id
    role = args.role
    node_dir = Path(args.node_dir).resolve()

    # Read incoming envelope from stdin
    stdin_content = sys.stdin.read().strip()

    if not stdin_content:
        # Initial turn 0 request: emit seed payload
        seed_data = load_seed_payload(node_dir)
        # Ensure sender matches node_id
        seed_data["sender"] = node_id
        envelope = Envelope.from_dict(seed_data)
    else:
        incoming_data = json.loads(stdin_content)
        incoming_envelope = Envelope.from_dict(incoming_data)

        if incoming_envelope.turn == 0:
            # Emit seed payload
            seed_data = load_seed_payload(node_dir)
            seed_data["sender"] = node_id
            envelope = Envelope.from_dict(seed_data)
        else:
            envelope = generate_node_response(
                node_id=node_id,
                role=role,
                node_dir=node_dir,
                incoming=incoming_envelope,
            )

    # Print output envelope JSON to stdout
    print(json.dumps(envelope.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
