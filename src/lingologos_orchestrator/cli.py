"""Command-line interface entry point for LingoLOGOS Multi-Instance Orchestrator.

Part of RFC LingoLOGOS-MULTI-ORCH-v1.
"""

import sys
import json
import argparse
from pathlib import Path

from .orchestrator import MultiInstanceOrchestrator
from .schema import OrchestratorError
from .consolidation import ModelConsolidator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LingoLOGOS Multi-Instance Orchestrator CLI."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/topology.json",
        help="Path to topology.json configuration file.",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        default="datasets",
        help="Path to datasets root directory containing per-node subdirectories.",
    )
    parser.add_argument(
        "--runtime-state",
        "-r",
        default="runtime_state",
        help="Path to volatile runtime_state directory.",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Workspace root directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON execution result.",
    )
    parser.add_argument(
        "--consolidate",
        choices=["triad", "unified"],
        default=None,
        help="Consolidate topology into macro-nodes (triad or unified) before execution.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    datasets_dir = Path(args.datasets)
    runtime_state_dir = Path(args.runtime_state)
    workspace_dir = Path(args.workspace)

    try:
        orchestrator = MultiInstanceOrchestrator.from_config_file(
            config_path=config_path,
            datasets_dir=datasets_dir,
            runtime_state_dir=runtime_state_dir,
            workspace_dir=workspace_dir,
        )

        if args.consolidate:
            consolidator = ModelConsolidator()
            if orchestrator.federated_config:
                c_model = consolidator.consolidate_federation(
                    orchestrator.federated_config, mode=args.consolidate
                )
            else:
                c_model = consolidator.consolidate_topology(
                    orchestrator.config, mode=args.consolidate
                )
            orchestrator.config = c_model.to_topology_config(
                session_id=f"{orchestrator.config.sessionId}-consolidated",
                max_turns=orchestrator.config.maxTurns,
            )
            # A macro node has no folder of its own under the datasets root: bind it to the
            # specification folder of the physical node it represents before running the session.
            orchestrator.set_representative_nodes(c_model.representatives())

        result = orchestrator.run_session()

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Session '{result.sessionId}' completed.")
            print(f"Total turns executed: {result.total_turns}")
            print(f"Termination reason: {result.termination_reason}")
            print("Verdict: PASS" if result.success else "Verdict: FAIL")

        return 0 if result.success else 1
    except OrchestratorError as exc:
        print(f"Orchestration Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected Error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
