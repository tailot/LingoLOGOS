"""Isolated process execution wrapper for LingoLOGOS nodes.

Ensures strict process decoupling: nodes are never imported as Python modules.
Communication occurs exclusively via CLI subprocess standard I/O streams.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from .schema import Envelope, OrchestratorError, SchemaValidationError


class ProcessExecutionError(OrchestratorError):
    """Raised when an isolated node process fails or exits with an error code."""
    def __init__(self, node_id: str, exit_code: int, stderr: str, stdout: str):
        super().__init__(
            f"Node process '{node_id}' failed with exit code {exit_code}.\n"
            f"Stderr: {stderr}\nStdout: {stdout}"
        )
        self.node_id = node_id
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout


class NodeProcessRunner:
    """Manages execution of an isolated node process via CLI wrapper."""

    def __init__(
        self,
        node_id: str,
        executable: str,
        role: str,
        datasets_dir: Path,
        workspace_dir: Optional[Path] = None,
        timeout_seconds: float = 30.0,
    ):
        self.node_id = node_id
        self.executable_spec = executable
        self.role = role
        self.datasets_dir = Path(datasets_dir).resolve()
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()
        self.timeout_seconds = timeout_seconds
        self.command_path = self._resolve_executable(executable)

    def _resolve_executable(self, executable: str) -> List[str]:
        """Resolves the executable path or command args."""
        # Direct path check
        if os.path.isfile(executable) and os.access(executable, os.X_OK):
            return [os.path.abspath(executable)]

        # Check in system PATH
        resolved = shutil.which(executable)
        if resolved:
            return [resolved]

        # Special fallback for lllayer if not yet on PATH
        if executable == "lllayer":
            # Can invoke python -m lingologos_orchestrator.lllayer
            return [sys.executable, "-m", "lingologos_orchestrator.lllayer"]

        # Default: attempt invocation directly
        return [executable]

    def execute_turn(self, input_envelope: Envelope) -> Envelope:
        """Executes one turn on the isolated node process passing envelope via stdin."""
        node_dataset_dir = self.datasets_dir / self.node_id

        cmd = list(self.command_path)
        cmd.extend([
            "--node-id", self.node_id,
            "--role", self.role,
            "--node-dir", str(node_dataset_dir),
        ])

        input_json_str = json.dumps(input_envelope.to_dict())

        try:
            proc = subprocess.run(
                cmd,
                input=input_json_str,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=str(self.workspace_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise OrchestratorError(
                f"Node '{self.node_id}' timed out after {self.timeout_seconds} seconds."
            ) from exc
        except Exception as exc:
            raise OrchestratorError(
                f"Failed to spawn process for node '{self.node_id}' with command {cmd}: {exc}"
            ) from exc

        if proc.returncode != 0:
            raise ProcessExecutionError(
                node_id=self.node_id,
                exit_code=proc.returncode,
                stderr=proc.stderr,
                stdout=proc.stdout,
            )

        stdout_clean = proc.stdout.strip()
        if not stdout_clean:
            raise OrchestratorError(
                f"Node '{self.node_id}' returned empty output on stdout."
            )

        try:
            output_dict = json.loads(stdout_clean)
        except json.JSONDecodeError as exc:
            raise OrchestratorError(
                f"Node '{self.node_id}' emitted invalid JSON output: {stdout_clean[:200]}"
            ) from exc

        return Envelope.from_dict(output_dict)
