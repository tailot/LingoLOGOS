"""Prepare and publish the small files that make a trained adapter usable.

The model repository on Hugging Face does not need the corpus: the corpus lives in the
GitHub release of this repository. What the hub needs is the adapter, its tokenizer
files, a model card, the evaluation numbers and the convenience script that queries the
model. This script assembles exactly that set in a staging folder and, only when asked,
pushes it as a model repository.

    python dataset/tools/evaluate_sft.py --model Qwen/Qwen2.5-3B-Instruct --adapter dataset/checkpoints/sft
    python dataset/tools/publish_to_hub.py                      # preview only
    python dataset/tools/publish_to_hub.py --apply --token-env HF_TOKEN

Base weights are refused on purpose: an adapter is a few tens of megabytes and the full
model of someone else does not belong in this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
CARD_SOURCE = DATASET_ROOT / "hf_publish" / "MODEL_CARD.md"
STAGING = DATASET_ROOT / "checkpoints" / "hf_repo"

REFUSE_SUFFIXES = (".gguf", ".bin", ".pt", ".pth")


def gather(adapter_dir: Path, max_file_mb: int) -> Tuple[List[Path], List[str]]:
    files: List[Path] = []
    skipped: List[str] = []
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in REFUSE_SUFFIXES and path.suffix.lower() not in (".safetensors",):
            skipped.append(path.name + " (refused suffix)")
            continue
        size_mb = path.stat().st_size / 1_048_576.0
        if size_mb > max_file_mb:
            skipped.append(path.name + " (" + str(round(size_mb, 1)) + " MB, looks like base weights)")
            continue
        files.append(path)
    return files, skipped


def read_training_config(adapter_dir: Path) -> Dict[str, Any]:
    path = adapter_dir / "training_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def read_scores(eval_file: Path) -> Dict[str, float]:
    if not eval_file.exists():
        return {}
    try:
        payload = json.loads(eval_file.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    overall = payload.get("overall") or {}
    scores = {key: value for key, value in overall.items() if isinstance(value, (int, float))}
    if isinstance(payload.get("rows"), int):
        scores["rows"] = payload["rows"]
    return scores


def render_card(values: Dict[str, str]) -> str:
    text = CARD_SOURCE.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    leftover = [part for part in text.split() if part.startswith("{{")]
    if leftover:
        print("WARNING: unfilled placeholders: " + ", ".join(sorted(set(leftover))))
    return text


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Assemble and publish the adapter files for the hub.")
    parser.add_argument("--adapter", default=str(DATASET_ROOT / "checkpoints" / "sft"))
    parser.add_argument("--repo-id", default="tailot/LingoLOGOS", help="hub model repository")
    parser.add_argument("--model", default="", help="base chat model (read from training config if empty)")
    parser.add_argument("--dataset-tag", default="dataset-v1", help="release tag holding the corpus")
    parser.add_argument("--eval-file", default=str(DATASET_ROOT / "checkpoints" / "eval" / "results_generated.json"))
    parser.add_argument("--max-file-mb", type=int, default=500, help="refuse files larger than this")
    parser.add_argument("--token-env", default="HF_TOKEN", help="environment variable holding the write token")
    parser.add_argument("--apply", action="store_true", help="actually upload (default only prepares and lists)")
    parser.add_argument("--clean", action="store_true", help="empty the staging folder first")
    args = parser.parse_args(argv)

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_absolute() and not adapter_dir.exists():
        for base in (DATASET_ROOT, DATASET_ROOT.parent):
            if (base / adapter_dir).exists():
                adapter_dir = base / adapter_dir
                break
    if not adapter_dir.exists():
        print("ERROR: adapter folder not found: " + str(adapter_dir))
        print("train first with dataset/tools/train_sft.py")
        return 2

    if args.clean and STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    copied = 0
    files, skipped = gather(adapter_dir, args.max_file_mb)
    for note in skipped:
        print("refused: " + note)
    for path in files:
        target = STAGING / path.relative_to(adapter_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1

    training = read_training_config(adapter_dir)
    scores = read_scores(Path(args.eval_file))
    base_model = args.model or str(training.get("model", "unknown"))
    values = {
        "BASE_MODEL": base_model,
        "LORA_R": str(training.get("lora_r", "?")),
        "LORA_ALPHA": str(training.get("lora_alpha", "?")),
        "EPOCHS": str(training.get("epochs", "?")),
        "LR": str(training.get("learning_rate", "?")),
        "DATASET_TAG": args.dataset_tag,
        "EXACT": str(scores.get("exact", "not measured")),
        "STRUCTURE": str(scores.get("structure", "not measured")),
        "PAYLOAD": str(scores.get("payload", "not measured")),
        "WELL_FORMED": str(scores.get("well_formed", "not measured")),
        "EVAL_ROWS": (" (" + str(scores["rows"]) + " rows)" if "rows" in scores else ""),
    }
    (STAGING / "README.md").write_text(render_card(values), encoding="utf-8")
    shutil.copy2(HERE / "ask_model.py", STAGING / "ask_model.py")
    if Path(args.eval_file).exists():
        shutil.copy2(args.eval_file, STAGING / "evaluation.json")

    manifest = sorted(p.relative_to(STAGING).as_posix() for p in STAGING.rglob("*") if p.is_file())
    total_mb = sum(p.stat().st_size for p in STAGING.rglob("*") if p.is_file()) / 1_048_576.0
    print("staging folder: " + str(STAGING))
    print("files: " + str(len(manifest)) + " | total size " + str(round(total_mb, 1)) + " MB")
    for name in manifest:
        print("  " + name)
    print("target hub repository: " + args.repo_id + " (model)")

    if not args.apply:
        print("PREVIEW ONLY: add --apply to upload")
        return 0

    token = os.environ.get(args.token_env, "")
    if not token:
        print("ERROR: no token in environment variable " + args.token_env)
        print("create a write token at https://huggingface.co/settings/tokens and set it, then rerun")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(STAGING),
        path_in_repo="",
        commit_message="Adapter, model card and query script from the LingoLOGOS dataset pipeline",
    )
    print("uploaded to https://huggingface.co/" + args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
