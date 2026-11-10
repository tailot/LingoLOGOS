"""Root probes for the corpus tools.

The corpus folder is data and tooling that can sit at any depth inside the repository,
so the tools must not assume that the parent of this folder is the project root, and
they must not assume a fixed folder name either. Both roots are discovered by walking
up from this file until a folder carrying the marker files is found, and either probe
can be overridden from the environment:

    LINGOLOGOS_REPO_ROOT     the project root holding pyproject.toml and the src tree
    LINGOLOGOS_DATASET_ROOT  the corpus folder holding templates/operators.json

An override is validated before it is used, so a wrong value fails loudly instead of
reading a different corpus by accident.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent


class CorpusPathsError(Exception):
    """Raised when a root cannot be located, or an override points at a wrong folder."""


def _ancestors(start: Path) -> Tuple[Path, ...]:
    """Return start and every folder above it, nearest first."""
    return (start, *start.parents)


def repo_root(start: Optional[Path] = None) -> Path:
    """Return the project root: the ancestor holding pyproject.toml and the src tree."""
    override = os.environ.get("LINGOLOGOS_REPO_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / "pyproject.toml").is_file() or not (candidate / "src").is_dir():
            raise CorpusPathsError(
                "LINGOLOGOS_REPO_ROOT points at " + str(candidate)
                + ", which carries no pyproject.toml and src folder"
            )
        return candidate
    origin = (start if start is not None else HERE).resolve()
    for candidate in _ancestors(origin):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise CorpusPathsError("no pyproject.toml with a src folder found above " + str(origin))


def corpus_root(start: Optional[Path] = None) -> Path:
    """Return the corpus folder: the ancestor holding templates/operators.json."""
    override = os.environ.get("LINGOLOGOS_DATASET_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / "templates" / "operators.json").is_file():
            raise CorpusPathsError(
                "LINGOLOGOS_DATASET_ROOT points at " + str(candidate)
                + ", which carries no templates/operators.json"
            )
        return candidate
    origin = (start if start is not None else HERE).resolve()
    for candidate in _ancestors(origin):
        if (candidate / "templates" / "operators.json").is_file():
            return candidate
    raise CorpusPathsError("no corpus folder with templates/operators.json found above " + str(origin))


def bootstrap(start: Optional[Path] = None) -> Tuple[Path, Path]:
    """Locate both roots, put the engine and the sibling tools on sys.path, return them."""
    origin = (start if start is not None else HERE).resolve()
    repo = repo_root(origin)
    corpus = corpus_root(origin)
    for entry in (str(repo), str(repo / "src"), str(origin)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return repo, corpus
