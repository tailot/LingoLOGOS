"""Isolation rule of the workbench: a closed package that depends on nothing of the project.

The interface and the project never import each other. This gate checks that no source
under ``src`` outside ``src/gui`` imports the package, that the GUI never imports the
engine, the trainer, the orchestrator or a solver, at module level or inside a function,
that it never edits ``sys.path``, that the names of the host project appear in code only
inside the workspace profile (a docstring or a comment may still talk about the project),
that the model layer stays free of Qt, and that the dependencies of the interface are
recorded where the project keeps its dependency records.
"""

import ast
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomllib arrived with 3.11, tomli is the same reader
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
GUI = SRC / "gui"

ENGINE_PACKAGES = (
    "neuro_symbolic",
    "lingologos_orchestrator",
    "torch",
    "transformers",
    "datasets",
    "huggingface_hub",
    "z3",
    "numpy",
)


def imported_roots(path: Path) -> set:
    """Top level packages a module pulls in at import time, read from its syntax tree."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def top_level_import_roots(path: Path) -> set:
    """Only the imports a module performs as soon as it is imported."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


PROJECT_IDENTIFIERS = ("neuro_symbolic", "lingologos_orchestrator", "asic_chip")
PROFILE_MODULE = GUI / "model" / "profile.py"


def string_constants(path: Path):
    """String literals a module carries in code, docstrings left out."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.value


def engine_sources():
    for path in sorted(SRC.rglob("*.py")):
        if GUI not in path.parents and path != GUI / "__init__.py":
            yield path


def gui_sources():
    return sorted(GUI.rglob("*.py"))


def test_the_engine_and_the_orchestrator_have_sources_to_guard():
    sources = list(engine_sources())
    assert len(sources) > 20, "the existing stack should be found before it is guarded"
    assert GUI.is_dir()


def test_no_existing_source_imports_the_workbench():
    offenders = []
    for path in engine_sources():
        roots = imported_roots(path)
        if "gui" in roots:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"the interface leaked into an engine module: {offenders}"


def test_no_script_at_the_repository_root_imports_the_workbench():
    offenders = []
    for path in sorted(REPO_ROOT.glob("*.py")):
        if "gui" in imported_roots(path):
            offenders.append(path.name)
    assert offenders == []


def test_the_interface_never_imports_the_engine_at_module_level():
    offenders = []
    for path in gui_sources():
        leaked = top_level_import_roots(path) & set(ENGINE_PACKAGES)
        if leaked:
            offenders.append((str(path.relative_to(GUI)), sorted(leaked)))
    assert offenders == [], (
        "engine, trainer or solver imported at module level: "
        f"{offenders}"
    )


def test_the_interface_never_imports_the_engine_at_any_depth():
    offenders = []
    for path in gui_sources():
        leaked = imported_roots(path) & set(ENGINE_PACKAGES)
        if leaked:
            offenders.append((str(path.relative_to(GUI)), sorted(leaked)))
    assert offenders == [], (
        "the interface must stay closed, with no import of the project even inside a function: "
        f"{offenders}"
    )


def dynamic_import_offenders(path: Path):
    """Edits of ``sys.path`` and dynamic imports that are not relative to the package itself."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "path":
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                found.append("sys.path")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in ("import_module", "__import__"):
                first = node.args[0] if node.args else None
                text = ""
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    text = first.value
                elif isinstance(first, ast.JoinedStr) and first.values:
                    head = first.values[0]
                    text = head.value if isinstance(head, ast.Constant) and isinstance(head.value, str) else ""
                if not text.startswith("."):
                    found.append(name)
    return found


def test_the_interface_never_edits_the_import_path_or_imports_outside_itself():
    offenders = []
    for path in gui_sources():
        found = dynamic_import_offenders(path)
        if found:
            offenders.append((str(path.relative_to(GUI)), sorted(set(found))))
    assert offenders == [], f"path edits or non relative dynamic imports in the interface: {offenders}"


def test_the_dynamic_import_check_recognises_what_it_forbids(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import sys\nfrom importlib import import_module\n"
        "sys.path.insert(0, 'x')\nimport_module('some_project.engine')\n",
        encoding="utf-8",
    )
    assert sorted(set(dynamic_import_offenders(bad))) == ["import_module", "sys.path"]
    good = tmp_path / "good.py"
    good.write_text(
        "from importlib import import_module\nimport_module('.panel', __name__)\n"
        "import_module(f'.{1}', __name__)\n",
        encoding="utf-8",
    )
    assert dynamic_import_offenders(good) == []


def test_host_project_names_live_only_in_the_workspace_profile():
    offenders = []
    for path in gui_sources():
        if path == PROFILE_MODULE:
            continue
        for text in string_constants(path):
            hits = [name for name in PROJECT_IDENTIFIERS if name in text]
            if hits:
                offenders.append((str(path.relative_to(GUI)), hits))
    assert offenders == [], f"host project names outside the workspace profile: {offenders}"


def test_the_workspace_profile_is_where_those_names_are_kept():
    spelled = " ".join(string_constants(PROFILE_MODULE))
    for name in ("lingologos_orchestrator", "asic_chip"):
        assert name in spelled, f"the built in profile should carry {name}"
    assert "neuro_symbolic" not in spelled, "the engine package is no longer needed to find a workspace root"


def test_the_profile_module_is_in_the_qt_free_model_layer():
    assert PROFILE_MODULE.is_file()
    assert "PyQt6" not in imported_roots(PROFILE_MODULE)


def test_the_workbench_root_stays_a_leaf_that_pulls_nothing_in():
    roots = top_level_import_roots(GUI / "__init__.py")
    assert "PyQt6" not in roots
    assert not (roots & set(ENGINE_PACKAGES))
    text = (GUI / "__init__.py").read_text(encoding="utf-8")
    assert "__getattr__" in text, "module attributes are resolved lazily"


def test_the_model_layer_holds_no_qt_at_all():
    model_files = sorted((GUI / "model").rglob("*.py"))
    assert model_files, "the Qt free model layer should exist"
    for path in model_files:
        assert "PyQt6" not in path.read_text(encoding="utf-8")
        assert "PyQt6" not in imported_roots(path)
        assert "PyQt6" not in top_level_import_roots(path)


def test_the_model_layer_is_the_only_place_the_parsers_live():
    for name in ("topology.py", "hardware.py", "svdoc.py", "dataset.py", "stream.py"):
        assert (GUI / "model" / name).is_file()


def test_only_the_worker_layer_drives_threads():
    worker_files = sorted((GUI / "workers").rglob("*.py"))
    assert worker_files
    for path in worker_files:
        if path.name == "__init__.py":
            continue
        assert "QtCore" in path.read_text(encoding="utf-8") or "QThread" in path.read_text(encoding="utf-8")
    for path in gui_sources():
        if GUI / "model" in path.parents:
            continue
        if path.parent == GUI / "workers":
            continue
        text = path.read_text(encoding="utf-8")
        assert "QThread" not in text, f"{path.name} should drive a worker instead of owning one"


@pytest.fixture
def pyproject():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_the_new_dependency_is_recorded_in_pyproject(pyproject):
    gui_extra = pyproject["project"]["optional-dependencies"]["gui"]
    assert any(requirement.startswith("PyQt6") for requirement in gui_extra)
    assert "PyQt6" not in " ".join(pyproject["project"].get("dependencies", [])), (
        "the interface stays optional: the engine installs and runs without Qt"
    )


def test_the_optional_yaml_dependency_is_recorded_with_the_interface(pyproject):
    gui_extra = pyproject["project"]["optional-dependencies"]["gui"]
    assert any(requirement.lower().startswith("pyyaml") for requirement in gui_extra), (
        "YAML topology documents need PyYAML, so the interface extra must name it"
    )


def test_the_reader_of_the_dependency_records_is_recorded(pyproject):
    """``tomllib`` arrived with 3.11 while the floor of this project is 3.10.

    This gate reads ``pyproject.toml`` itself, so on 3.10 it needs ``tomli``, and that
    dependency has to be recorded with the marker that keeps it off newer interpreters:
    an unrecorded reader breaks the pipeline at collection time, not at an assertion.
    """
    dev_extra = pyproject["project"]["optional-dependencies"]["dev"]
    assert any(
        requirement.startswith("tomli") and "python_version" in requirement
        for requirement in dev_extra
    ), "the 3.10 interpreter needs tomli to read these files, with its marker"
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert any("tomli" in line and "python_version" in line for line in lines), (
        "requirements.txt carries the same marker, or a 3.10 environment misses it"
    )


def test_the_workbench_is_launchable_as_a_command(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert scripts["lingologos-workbench"] == "gui.app:main"


def test_the_test_environment_can_import_the_interface(pyproject):
    pythonpath = pyproject["tool"]["pytest"]["ini_options"]["pythonpath"]
    assert "src" in pythonpath


def test_requirements_record_the_interface_dependency():
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert any("PyQt6" in line for line in lines)
