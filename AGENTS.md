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

## 2. Repository Hygiene Standards
- **Zero bytecode cache**: Never commit `*.pyc` files, `*.egg-info/`, or `__pycache__/` directories.
- All newly introduced dependencies must be recorded in `requirements.txt`, `pyproject.toml`, and documented in `README.md` and `doc/TECHNICAL_DOCUMENTATION.md`.
- Always respect and maintain `.gitignore`.

---

## 3. Mandatory Testing and Verification Protocol

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
