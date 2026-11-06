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
