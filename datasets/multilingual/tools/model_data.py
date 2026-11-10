"""Shared corpus access for the pure-model tools: tokenizer, mask, supervised pairs.

Both ``train_model.py`` and ``evaluate_model.py`` need the same three objects, and they
have to be the same objects the mask gates were verified against, otherwise the training
numbers and gate M5 would describe different machines. This module builds them once:

* the learned tokenizer of ``datasets/multilingual/checkpoints/tokenizer/tokenizer.json``, trained on the
  ``train`` split only, so the held-out score carries no vocabulary leakage;
* the algebra automaton of the operator registry plus the leaf payloads of the ``train``
  split, optionally narrowed by the argument kinds observed in the same split;
* the supervised pairs, taken from ``datasets/multilingual/data/<language>/<split>-*.jsonl`` as the
  ``phrase`` field and the space separated ``bracket_ast``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_paths import bootstrap  # noqa: E402

REPO_ROOT, DATASET_ROOT = bootstrap(HERE)

from neuro_symbolic.masking import AlgebraMask, TokenStreamMask  # noqa: E402
from neuro_symbolic.tokenizer import Tokenizer  # noqa: E402
from neuro_symbolic.training import AlgebraExample  # noqa: E402

from verify_masking import (  # noqa: E402
    collect_leaves,
    item_rows,
    load_registry,
    reference_is_legal,
)

DEFAULT_TOKENIZER = DATASET_ROOT / "checkpoints" / "tokenizer" / "tokenizer.json"
MAX_LEAF_TOKENS = 3


def load_tokenizer(path: Optional[str] = None) -> Tokenizer:
    """Read the learned tokenizer, refusing an archive of another format."""
    return Tokenizer.load(str(Path(path) if path else DEFAULT_TOKENIZER))


def arities_of_registry() -> Dict[str, int]:
    """Operator name to arity, straight from ``datasets/multilingual/templates/operators.json``."""
    return {name: int(spec["arity"]) for name, spec in load_registry().items()}


def build_algebra_mask(refine: bool = True) -> Tuple[AlgebraMask, Dict[str, Any]]:
    """Automaton of the algebra, built from the train split exactly as gate M verifies it.

    With ``refine`` the argument kinds of each operator position are learned from the train
    trees, which shrinks the admissible sets without ever excluding a tree of that split.
    """
    from collections import Counter

    arities = arities_of_registry()
    train_rows = list(item_rows("train"))
    mask = AlgebraMask.from_leaf_payloads(arities, collect_leaves(train_rows))
    summary = mask.describe()
    summary["refined"] = bool(refine)
    if refine:
        counts: Any = Counter()
        for row in train_rows:
            AlgebraMask.observe_child_kinds(row["ast_json"], counts)
        mask = AlgebraMask(
            arities,
            mask.leaf_forms,
            mask.leaf_value_tokens,
            AlgebraMask.child_kinds_from_counts(counts),
        )
        summary = mask.describe()
        summary["refined"] = True
    summary["train_rows_for_alphabet"] = len(train_rows)
    return mask, summary


def build_token_mask(refine: bool = True) -> Tuple[TokenStreamMask, Dict[str, Any], Dict[str, int]]:
    """Token-level mask over the learned vocabulary, with its report and the registry arities."""
    mask, summary = build_algebra_mask(refine)
    token_mask = TokenStreamMask(mask, load_tokenizer())
    return token_mask, dict(token_mask.report(), algebra_refined=summary["refined"]), arities_of_registry()


def examples_of(split: str, limit: int = 0) -> List[AlgebraExample]:
    """Supervised pairs of one split: the sentence and the bracket word that means it."""
    rows = list(item_rows(split))
    if limit:
        rows = rows[: int(limit)]
    examples: List[AlgebraExample] = []
    for row in rows:
        symbols = str(row["bracket_ast"]).split()
        if not symbols:
            continue
        examples.append(
            AlgebraExample(
                identifier=str(row["id"]),
                source_text=str(row["phrase"]),
                symbols=symbols,
                language=str(row.get("language", "")),
                domain=str(row.get("domain", "")),
                logic_family=str(row.get("logic_family", "")),
            )
        )
    return examples


def length_profile(encoded: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Token-length statistics of encoded examples, to size the position budgets."""
    inputs = sorted(len(item["input_ids"]) for item in encoded)
    targets = sorted(len(item["target"]) for item in encoded)

    def summary(values: Sequence[int]) -> Dict[str, Any]:
        return {
            "minimum": values[0],
            "mean": round(sum(values) / len(values), 2),
            "median": values[len(values) // 2],
            "p99": values[min(len(values) - 1, int(0.99 * len(values)))],
            "maximum": values[-1],
        }

    return {"encoder_input": summary(inputs), "decoder_protocol": summary(targets)}


def device_report(requested: str) -> Dict[str, Any]:
    """What the run will actually use, with the memory of the accelerator when present."""
    import torch

    report: Dict[str, Any] = {"requested": requested}
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available on this machine")
        index = 0
        if ":" in requested:
            index = int(requested.split(":")[1])
        properties = torch.cuda.get_device_properties(index)
        report.update(
            {
                "resolved": "cuda:" + str(index),
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": str(properties.major) + "." + str(properties.minor),
                "multiprocessors": int(properties.multi_processor_count),
            }
        )
    else:
        report["resolved"] = "cpu"
        report["physical_cores"] = __import__("os").cpu_count()
    return report


def read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
