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

from .schema import Envelope, VolatileState, Payload, SMTConstraints, OrchestratorError
from .topology_ctrl import TopologyCtrlRegister
from .lllayer_masker import NativeASTLogitsProcessor
from .lllayer_engine import IncrementalASTEngine, SimpleASTGrammar, MockRunnerModel
from .lllayer_text import (
    GRAMMAR_AUTO,
    GRAMMAR_CHOICES,
    load_topology_binding,
    render_report,
    route_text,
)


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
    """Executes a volatile in-memory AST constrained decoding session.

    The built-in AST alphabet lives in :mod:`lingologos_orchestrator.lllayer_text`,
    which is the same session ``--data`` reaches from the command line. The
    ``vocab_size`` argument is retained for interface compatibility: it sizes the
    volatile mask vector and never changes the decoded tree.
    """
    del node_id, role, vocab_size
    return route_text(incoming_ast, grammar="generic").tree


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


def text_query_envelope(
    node_id: str,
    topology_ctrl: TopologyCtrlRegister,
    result: Any,
    incoming: Optional[Envelope] = None,
) -> Envelope:
    """Wraps one --data routing result in the node envelope protocol."""
    tick = incoming.volatileState.internalTimeTick if incoming is not None else 0
    turn = incoming.turn if incoming is not None else 0
    recipient = incoming.sender if incoming is not None and incoming.sender != node_id else None
    extra = {"grammarSource": result.grammar, "grammarRoute": result.to_dict()}
    out_payload = Payload(
        naturalLanguage=str(result.evidence.get("text", "")),
        abstractSyntaxTree=result.tree,
        smtConstraints=SMTConstraints(
            declarations=[],
            assertions=["(assert true)"],
            z3Status="sat",
        ),
    )
    envelope = Envelope(
        sender=node_id,
        recipient=recipient,
        turn=turn,
        volatileState=VolatileState(internalTimeTick=tick, extra=extra),
        payload=out_payload,
    )
    envelope.topologyCtrl = topology_ctrl.to_dict()
    envelope.datasetId = topology_ctrl.dataset_id
    return envelope


def run_text_query(args, node_dir: Path, topology_ctrl: TopologyCtrlRegister) -> int:
    """Handles a --data query: one sentence in, one validated syntax tree out."""
    query_text = args.data
    if query_text is None and args.data_file:
        data_path = Path(args.data_file)
        if not data_path.is_file():
            print("lllayer: text file not found: " + str(data_path), file=sys.stderr)
            return 2
        query_text = data_path.read_text(encoding="utf-8-sig")

    incoming_envelope: Optional[Envelope] = None
    if not sys.stdin.isatty():
        incoming_content = sys.stdin.read().strip()
        if incoming_content:
            incoming_envelope = Envelope.from_dict(json.loads(incoming_content))

    output_shape = args.format
    if output_shape == "auto":
        output_shape = "envelope" if incoming_envelope is not None else "report"

    try:
        result = route_text(
            query_text,
            grammar=args.grammar,
            topology_config=Path(args.topology_config) if args.topology_config else None,
            model_package=Path(args.model_package) if args.model_package else None,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            node_dir=node_dir,
            device=args.device,
            max_steps=args.max_decode_steps,
        )
    except (OrchestratorError, OSError) as error:
        print("lllayer: " + str(error), file=sys.stderr)
        return 2

    # A decoded tree may hold non-ASCII payload tokens, so the console is set to UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - depends on the host stream
        pass

    if output_shape == "tree":
        print(result.tree)
    elif output_shape == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif output_shape == "envelope":
        envelope = text_query_envelope(args.node_id, topology_ctrl, result, incoming_envelope)
        print(json.dumps(envelope.to_dict(), indent=2))
    else:
        print(render_report(result))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LingoLOGOS CLI Process Wrapper (lllayer) for isolated node execution."
    )
    parser.add_argument("--node-id", required=True, help="Arbitrary node identifier.")
    parser.add_argument("--role", default="node", help="Role of the node (e.g. proponent, opponent, validator).")
    parser.add_argument("--node-dir", default=None, help="Path to the node dataset directory (required by turn mode).")
    parser.add_argument("--hardware-node-id", type=int, default=0, help="Physical node ID from strapping pins (0..15).")
    parser.add_argument("--probe-peers", type=int, default=0, help="Number of external peers discovered on board.")
    parser.add_argument("--dataset-id", type=int, default=0, help="Logical dataset ID assigned to this node.")
    parser.add_argument("--topology-ctrl-reg", type=str, default=None, help="Optional raw hex or JSON for TOPOLOGY_CTRL register.")
    parser.add_argument("--constrained-decoding", action="store_true", help="Enable in-memory AST-guided logits masker decoding.")
    parser.add_argument("--data", type=str, default=None, help="Text to route into a syntax tree (query mode, no envelope reply needed).")
    parser.add_argument("--data-file", type=str, default=None, help="Path of a UTF-8 text file holding the text to route.")
    parser.add_argument(
        "--grammar",
        choices=list(GRAMMAR_CHOICES),
        default=GRAMMAR_AUTO,
        help="Grammar source for --data: auto, generic, topology (declared topology) or dataset (trained package).",
    )
    parser.add_argument("--topology-config", type=str, default=None, help="Topology JSON declaring the nodes and routing matrix for --data.")
    parser.add_argument("--model-package", type=str, default=None, help="Release package folder with tokenizer.json, mask_spec.json and weights.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset folder whose checkpoints hold the release package.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for the dataset grammar (cpu, cuda).")
    parser.add_argument("--max-decode-steps", type=int, default=None, help="Decode step cap for the dataset grammar.")
    parser.add_argument("--format", choices=["auto", "tree", "report", "json", "envelope"], default="auto", help="Output shape of a --data query.")
    parser.add_argument("--show-topology", action="store_true", help="Print the declared nodes and routing matrix of --topology-config and exit.")
    args = parser.parse_args()

    node_id = args.node_id
    role = args.role
    query_mode = args.data is not None or args.data_file is not None or args.show_topology
    if args.node_dir is None and not query_mode:
        # Turn mode always reads seed_payload.json, so the folder stays mandatory there.
        parser.error("the following arguments are required: --node-dir")
    node_dir = Path(args.node_dir).resolve() if args.node_dir else None

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

    # Query mode: the node answers one sentence and returns one validated tree.
    if args.show_topology:
        if not args.topology_config:
            print("lllayer: --show-topology needs --topology-config PATH", file=sys.stderr)
            return 2
        try:
            binding = load_topology_binding(Path(args.topology_config))
        except (OrchestratorError, OSError) as error:
            print("lllayer: " + str(error), file=sys.stderr)
            return 2
        declaration = binding.to_dict()
        declaration["topologyCtrl"] = topology_ctrl.to_dict()
        print(json.dumps(declaration, indent=2))
        return 0

    if args.data is not None or args.data_file is not None:
        return run_text_query(args, node_dir, topology_ctrl)

    # Turn mode: --node-dir was validated above and is therefore present here.
    assert node_dir is not None

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
