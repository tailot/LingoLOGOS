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
import httpx

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
        node_dir: Optional[Path] = None,
    ):
        self.node_id = node_id
        self.executable_spec = executable
        self.role = role
        self.datasets_dir = Path(datasets_dir).resolve()
        # node_dir overrides the datasets_dir/node_id convention: a federated node reads its
        # specification files from a git cache or a macro node runs on the dataset of the node it
        # stands for, so the folder holding ontology.smt2 is not always the one named after the id.
        self.node_dir = Path(node_dir).resolve() if node_dir else None
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()
        self.timeout_seconds = timeout_seconds
        self.command_path = self._resolve_executable(executable)
        self.hardware_node_id: int = 0
        self.probe_peers: int = 0
        self.dataset_id: int = 0
        self.topology_ctrl_reg: Optional[str] = None

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

    def _child_environment(self) -> Dict[str, str]:
        """Return an environment in which this package is importable by the node.

        A node launched as ``python -m lingologos_orchestrator.lllayer`` finds the package only
        when it is installed or when the folder holding it is on ``PYTHONPATH``. The node runs in
        its own working directory, so a relative entry inherited from the caller would not resolve
        there. The absolute folder is prepended and whatever the caller already had is kept behind
        it, so an installed deployment is unaffected.
        """
        env = dict(os.environ)
        package_parent = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        entries = [package_parent] if not existing else [package_parent, existing]
        env["PYTHONPATH"] = os.pathsep.join(entries)
        return env

    def execute_turn(self, input_envelope: Envelope) -> Envelope:
        """Executes one turn on the isolated node process passing envelope via stdin."""
        node_dataset_dir = self.node_dir if self.node_dir is not None else self.datasets_dir / self.node_id

        cmd = list(self.command_path)
        cmd.extend([
            "--node-id", self.node_id,
            "--role", self.role,
            "--node-dir", str(node_dataset_dir),
            "--hardware-node-id", str(self.hardware_node_id),
            "--probe-peers", str(self.probe_peers),
            "--dataset-id", str(self.dataset_id),
        ])
        if self.topology_ctrl_reg:
            cmd.extend(["--topology-ctrl-reg", str(self.topology_ctrl_reg)])

        input_json_str = json.dumps(input_envelope.to_dict())

        try:
            proc = subprocess.run(
                cmd,
                input=input_json_str,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=str(self.workspace_dir),
                env=self._child_environment(),
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


class HttpNodeRunner:
    """Manages execution of a remote HTTP/REST node via POST endpoint."""

    def __init__(
        self,
        node_id: str,
        endpoint_url: str,
        role: str = "node",
        auth_method: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self.node_id = node_id
        self.endpoint_url = endpoint_url
        self.role = role
        self.auth_method = auth_method
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds

    def execute_turn(self, input_envelope: Envelope) -> Envelope:
        """Sends the input envelope to the remote endpoint and parses the response envelope."""
        headers = {"Content-Type": "application/json"}
        if self.auth_method == "bearer_token" and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        payload = input_envelope.to_dict()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(self.endpoint_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return Envelope.from_dict(data)
        except Exception as exc:
            raise OrchestratorError(
                f"HTTP node '{self.node_id}' failed request to {self.endpoint_url}: {exc}"
            ) from exc
