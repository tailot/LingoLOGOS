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
from .topology_ctrl import TopologyCtrlRegister
from .lllayer_masker import NativeASTLogitsProcessor
from .lllayer_engine import IncrementalASTEngine, SimpleASTGrammar, MockRunnerModel


def discover_board_topology(
    hardware_node_id: int,
    discovered_peers_count: int,
    dataset_id: int = 0,
    routing_mask: int = 0,
) -> TopologyCtrlRegister:
    """Executes topology negotiation and returns programmed TOPOLOGY_CTRL register.

    In standalone mode (probe timeout or 0 peers):
      TOPOLOGY_CTRL.STANDALONE = 1
      DATASET_ID = primary local dataset
    In interconnected multi-chip mode (peers > 0):
      TOPOLOGY_CTRL.STANDALONE = 0
      TOTAL_NODES = 1 + discovered_peers_count
      DATASET_ID = assigned shard index
    """
    return TopologyCtrlRegister.from_handshake(
        hardware_node_id=hardware_node_id,
        discovered_peers_count=discovered_peers_count,
        dataset_id=dataset_id,
        routing_mask=routing_mask,
    )


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


def run_constrained_ast_decoding(
    node_id: str,
    role: str,
    incoming_ast: str,
    vocab_size: int = 16,
) -> str:
    """Executes a volatile in-memory AST constrained decoding session."""
    tokens = ["ROOT", "(", "Transition", "StateA", "StateB", ")", "END", "MODIFIER"]
    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for i, t in enumerate(tokens)}

    node_types = {"MODIFIER": "OPTIONAL_MODIFIER"}
    transitions = {
        ("ROOT", "("): "NODE_OPN",
        ("NODE_OPN", "Transition"): "NODE_TRANS",
        ("NODE_TRANS", "StateA"): "NODE_S1",
        ("NODE_S1", "StateB"): "NODE_S2",
        ("NODE_S2", ")"): "TERMINAL",
        ("NODE_TRANS", "MODIFIER"): "MODIFIER",
        ("MODIFIER", ")"): "TERMINAL",
    }
    valid_tokens = {
        "ROOT": {token_to_id["("]},
        "NODE_OPN": {token_to_id["Transition"]},
        "NODE_TRANS": {token_to_id["StateA"], token_to_id["MODIFIER"]},
        "NODE_S1": {token_to_id["StateB"]},
        "NODE_S2": {token_to_id[")"]},
        "MODIFIER": {token_to_id[")"]},
    }
    closing_tokens = {token_to_id[")"]}

    grammar = SimpleASTGrammar(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        node_types=node_types,
        transitions=transitions,
        valid_tokens=valid_tokens,
        closing_tokens=closing_tokens,
        terminal_nodes={"TERMINAL", "END"},
    )

    processor = NativeASTLogitsProcessor(vocab_size=max(vocab_size, len(tokens)))
    runner_model = MockRunnerModel(vocab_size=processor.vocab_size, token_to_id=token_to_id, id_to_token=id_to_token)
    engine = IncrementalASTEngine(grammar_spec=grammar, logits_processor=processor)

    return engine.run_session(prompt=incoming_ast, runner_model=runner_model)


def generate_node_response(
    node_id: str,
    role: str,
    node_dir: Path,
    incoming: Envelope,
    constrained_decoding: bool = False,
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

    if constrained_decoding:
        natural_text = f"Node {node_id} constrained decoding response."
        decoded_ast = run_constrained_ast_decoding(node_id, role, incoming.payload.abstractSyntaxTree)
        ast = f"{decoded_ast} (Node {node_id})"
        if not asserts:
            asserts.append("(assert true)")
    elif role == "opponent":
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
    parser.add_argument("--hardware-node-id", type=int, default=0, help="Physical node ID from strapping pins (0..15).")
    parser.add_argument("--probe-peers", type=int, default=0, help="Number of external peers discovered on board.")
    parser.add_argument("--dataset-id", type=int, default=0, help="Logical dataset ID assigned to this node.")
    parser.add_argument("--topology-ctrl-reg", type=str, default=None, help="Optional raw hex or JSON for TOPOLOGY_CTRL register.")
    parser.add_argument("--constrained-decoding", action="store_true", help="Enable in-memory AST-guided logits masker decoding.")
    args = parser.parse_args()

    node_id = args.node_id
    role = args.role
    node_dir = Path(args.node_dir).resolve()

    # Determine TOPOLOGY_CTRL configuration
    if args.topology_ctrl_reg:
        if args.topology_ctrl_reg.startswith("0x") or args.topology_ctrl_reg.isdigit():
            topology_ctrl = TopologyCtrlRegister.unpack(int(args.topology_ctrl_reg, 0))
        else:
            topology_ctrl = TopologyCtrlRegister.from_dict(json.loads(args.topology_ctrl_reg))
    else:
        topology_ctrl = discover_board_topology(
            hardware_node_id=args.hardware_node_id,
            discovered_peers_count=args.probe_peers,
            dataset_id=args.dataset_id,
        )

    # Read incoming envelope from stdin
    stdin_content = sys.stdin.read().strip()

    if not stdin_content:
        # Initial turn 0 request: emit seed payload
        seed_data = load_seed_payload(node_dir)
        # Ensure sender matches node_id
        seed_data["sender"] = node_id
        seed_data["topologyCtrl"] = topology_ctrl.to_dict()
        seed_data["datasetId"] = topology_ctrl.dataset_id
        envelope = Envelope.from_dict(seed_data)
    else:
        incoming_data = json.loads(stdin_content)
        incoming_envelope = Envelope.from_dict(incoming_data)

        if incoming_envelope.turn == 0:
            # Emit seed payload
            seed_data = load_seed_payload(node_dir)
            seed_data["sender"] = node_id
            seed_data["topologyCtrl"] = topology_ctrl.to_dict()
            seed_data["datasetId"] = topology_ctrl.dataset_id
            envelope = Envelope.from_dict(seed_data)
        else:
            envelope = generate_node_response(
                node_id=node_id,
                role=role,
                node_dir=node_dir,
                incoming=incoming_envelope,
                constrained_decoding=args.constrained_decoding,
            )
            # Propagate or preserve topologyCtrl and datasetId
            envelope.topologyCtrl = topology_ctrl.to_dict()
            envelope.datasetId = topology_ctrl.dataset_id

    # Print output envelope JSON to stdout
    print(json.dumps(envelope.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
