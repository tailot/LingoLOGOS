# LingoLOGOS Neuro-Symbolic Dataset Extraction (Full Pre-training Scale)

This directory contains the specifications, generator modules, and validation pipeline for the five neuro-symbolic datasets ($\mathcal{D}_1 \dots \mathcal{D}_5$) calibrated exclusively for **Full Pre-training Scale** ($N = 1,000,000$ per dataset, total 5,000,000 samples).

## Quick Start

Generate sample batches or full datasets:

```bash
# Generate 1,000 test samples for all 5 datasets with deterministic seed 42
python3 extract/pipeline.py --dataset ALL --num-samples 1000 --seed 42 --output-dir extract/output

# Generate specific dataset D1 (Grammatical Derivation)
python3 extract/pipeline.py --dataset D1 --num-samples 1000000 --seed 42 --output-dir /data/lingologos_pretrain
```

## Structure

- `SPECIFICATIONS.md`: Formal mathematical and architectural specifications for $\mathcal{D}_1 \dots \mathcal{D}_5$, pre-export validation gates, storage footprints, and execution parameters.
- `common.py`: Shared base schema definitions and seed initializers.
- `d1_gf_ast.py`: $\mathcal{D}_1$ Grammatical Derivation & AST Mappings generator (GF RGL, 5 languages, bounded depth $d \le 6$).
- `d2_smt_consistency.py`: $\mathcal{D}_2$ Ontological Consistency & SMT Constraints generator (Z3 solver verification, strict 50/50 SAT/UNSAT balance, Allen temporal intervals).
- `d3_command_execution.py`: $\mathcal{D}_3$ Multi-domain Command Execution generator (LLLayer CLI & multi-agent routing across 10 vertical sectors).
- `d4_syntactic_robustness.py`: $\mathcal{D}_4$ Syntactic Robustness & Anti-Hallucination generator (Dyck-2 nested structures and controlled noise injection).
- `d5_logit_masking.py`: $\mathcal{D}_5$ Dynamic Logit Masking generator (`VolatileLogitsMasker` automata states and active bitsets).
- `pipeline.py`: Unified pipeline orchestrator CLI.
