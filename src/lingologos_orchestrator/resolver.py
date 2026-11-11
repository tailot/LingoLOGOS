"""Universal transport resolver for LingoLOGOS federated nodes.

Supports:
  1. local: Filesystem-resident resources in workspace.
  2. git: Remote git repositories dynamically cloned/updated into a local cache.
  3. http: Remote HTTP/REST endpoints for offloading computational or SMT tasks.

Follows English-only, ASCII-only rules and avoids apostrophes.
"""

import subprocess
import pathlib
from typing import Dict, Any, Union, Optional
import httpx


def _local_base_dir(node_config: Dict[str, Any], workspace_dir: Optional[pathlib.Path]) -> pathlib.Path:
    """Returns the absolute local base directory declared for a node.

    A relative local_cache_path is anchored on the workspace so that a session started from any
    working directory reads and writes the same cache folder.
    """
    base = pathlib.Path(node_config.get("local_cache_path") or ".")
    if base.is_absolute():
        return base.resolve()
    return (pathlib.Path(workspace_dir or ".") / base).resolve()


def resolve_node_transport(
    node_config: Dict[str, Any],
    workspace_dir: Optional[pathlib.Path] = None,
) -> Union[pathlib.Path, str]:
    """Resolves execution location or remote target for a given node configuration.

    For git transport the repository is synchronized into the local cache directory: a shallow
    clone on the first encounter, an incremental pull afterwards.
    """
    transport = node_config.get("transport", "local")

    if transport == "git":
        repo_url = node_config["remote_source"]
        cache_path = _local_base_dir(node_config, workspace_dir)
        branch = node_config.get("branch", "main")

        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(cache_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                ["git", "-C", str(cache_path), "pull", "origin", branch],
                check=True,
                capture_output=True,
                text=True,
            )

        return cache_path

    elif transport == "http":
        return node_config["endpoint_url"]

    return _local_base_dir(node_config, workspace_dir)


class NodeResolver:
    """Universal resolver for node resources across local, git, and http transports."""

    def __init__(self, workspace_dir: Optional[pathlib.Path] = None):
        self.workspace_dir = pathlib.Path(workspace_dir or ".").resolve()

    def resolve(self, node_config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves node transport and locates required specification files.

        Returns a dictionary containing:
          - transport: str ('local', 'git', 'http')
          - root_path: pathlib.Path or None
          - endpoint_url: str or None
          - grammar_path: pathlib.Path or str
          - ontology_path: pathlib.Path or str
          - seed_path: pathlib.Path or str
        """
        transport = node_config.get("transport", "local")
        specs = node_config.get("specs", {})
        res = resolve_node_transport(node_config, workspace_dir=self.workspace_dir)

        if transport == "http":
            return {
                "transport": "http",
                "root_path": None,
                "endpoint_url": str(res),
                "grammar_path": specs.get("grammar", ""),
                "ontology_path": specs.get("ontology", ""),
                "seed_path": specs.get("seed", ""),
                "node_dir": None,
                "auth_method": node_config.get("auth_method"),
                "auth_token_env": node_config.get("auth_token_env"),
            }

        base_dir = (self.workspace_dir / res).resolve() if not res.is_absolute() else res
        grammar_rel = specs.get("grammar", "grammar.gf")
        ontology_rel = specs.get("ontology", "ontology.smt2")
        seed_rel = specs.get("seed", "seed_payload.json")

        ontology_path = (base_dir / ontology_rel).resolve()

        return {
            "transport": transport,
            "root_path": base_dir,
            "endpoint_url": None,
            "grammar_path": (base_dir / grammar_rel).resolve(),
            "ontology_path": ontology_path,
            "seed_path": (base_dir / seed_rel).resolve(),
            "node_dir": ontology_path.parent,
            "auth_method": None,
            "auth_token_env": None,
        }
