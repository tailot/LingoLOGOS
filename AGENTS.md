# Mandatory Instructions for AI Agents (AGENTS.md)

This file defines mandatory directives and operational constraints for any AI agent or automated assistant operating in this repository.

---

## 1. Mandatory Documentation Maintenance Rule

Upon **any structural, architectural, algorithmic, or interface change**, the AI agent **MUST**:
1. Promptly update `doc/TECHNICAL_DOCUMENTATION.md` to reflect the changes.
2. Update `README.md` if the change affects system architecture, hyperparameter matrix, metrics, testing procedures, or CLI commands.
3. Document any modified mathematical formulations, equations, or formal invariants.

> **Golden Rule**: No Pull Request or Commit modifying source files under `src/` or artifacts under `artifacts/` may be finalized without corresponding updates to the technical documentation in `doc/`.

---

## 2. Mandatory Language Policy (English Only)

**English is the only allowed language for every piece of prose in this repository.** This applies to:
- Code comments, inline notes, and block comments.
- Module, class, and function docstrings.
- Console/log output, progress messages, `argparse` help text, gate labels, and the final verdict line.
- Exception messages and assertion messages (including inside test files).
- Test names, test docstrings, and helper prose in `tests/`.
- All documentation: `README.md`, `doc/`, `AGENTS.md`, and comments inside `artifacts/`.
- Commit messages and pull-request descriptions.

Additional constraints:
1. **ASCII only inside Python sources**: never introduce accented characters, smart quotes, or typographic dashes in `.py` files. Rephrase instead of using apostrophes.
2. **Values are not prose**: keep technical identifiers, dictionary keys, status constants (`UNSAT`, `SAT`, `INCONCLUSIVE`, `INVALID`), CLI flags, and exported artifact filenames byte-identical when translating surrounding text.
3. **Legacy text**: when editing a file that still contains non-English prose, translate that prose in the same change.
4. **Legitimate exceptions**: formal-language data under study (bracket tokens `()`, `[]`), third-party library names, and the historical artifact filename `artifacts/DynamicGrammarIta.gf` (retained for compatibility; its contents must still be English).

---

## 3. Repository Hygiene Standards
- **Zero bytecode cache**: Never commit `*.pyc` files, `*.egg-info/`, or `__pycache__/` directories.
- All newly introduced dependencies must be recorded in `requirements.txt`, `pyproject.toml`, and documented in `README.md` and `doc/TECHNICAL_DOCUMENTATION.md`.
- Always respect and maintain `.gitignore`.

---

## 4. Mandatory Testing and Verification Protocol

Before completing any change or submission, the AI agent **MUST** run the following verification steps:

1. **Run Unit and Integration Test Suite**:
   ```bash
   pytest -v tests/
   ```
   All tests must pass (`PASSED`) with zero errors and zero failures.

2. **Verify End-to-End Synthesis and Export Pipeline**:
   If changes affect the parser, grammar representation, or trainer, rerun synthesis:
   ```bash
   python3 src/neuro_symbolic/synthesize.py
   ```
   Verify that:
   - `artifacts/grammar.json` is valid and parsable.
   - `artifacts/grammar.yaml` matches schema.
   - `artifacts/DynamicGrammarIta.gf` is properly generated.
   - Corpus coverage rate is strictly 100%.
   - Formal Z3 SMT verification results in **UNSAT** (100% Soundness).

---

## 5. Mandatory Workbench Isolation Rule (the UI Never Depends on the Project)

The desktop workbench under `src/gui/` is a closed package. The AI agent **MUST NOT** introduce any dependency from the interface toward the rest of this repository:

1. **No imports**: no module of `src/gui/` may import any module of the project outside that package (the engine, the trainer, the orchestrator, the ASIC scripts, the tools under `datasets/`), at module level or inside a function, may not import dynamically except lazily inside `gui` itself, and may not touch `sys.path`.
2. **No project names in code**: the identifiers of the host project (module names, make targets, script paths, project folders) may appear in code string literals only inside `src/gui/model/profile.py`, the workspace profile. Docstrings and comments may still describe the project.
3. **Files and processes are the only channels**: the workbench reaches the project through files it reads from the workspace and commands it launches as separate processes, configured by the built-in profile and by an optional `workbench.json` at the workspace root. Values are never hardcoded elsewhere, and never typed into the interface.
4. **No reverse dependency**: modules of the project outside `src/gui/` must never import `gui`, and the GUI toolkit must stay in the optional `[gui]` extra of `pyproject.toml` (with `PyQt6` carrying the same limit in `requirements.txt`).
5. **Nothing remote**: no web toolkit, no embedded browser, no CDN, no remote font, icon or asset, no network call inside the interface. The style sheet, the fonts and the icons are local.

`tests/test_workbench_isolation.py` and `tests/test_workbench_offline_guard.py` enforce all of it from the syntax trees; section 16 of `doc/TECHNICAL_DOCUMENTATION.md` states the same rule as part of the design.
