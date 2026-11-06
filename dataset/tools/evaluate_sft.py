"""Evaluate a fine-tuned chat model on the held-out items of this dataset.

The score that matters is not text similarity but whether the model reproduces the
meaning: the exact bracket word, the readable pattern and the SMT assertion. The
script generates an answer for every row of an SFT split, then scores it four ways:

    exact        the whole assistant message matches the gold string
    normalized   matches after case folding and space collapsing
    structure    the bracket word is identical and well balanced
    payload      all three formal renderings are identical

It also reports the structural rate of the answers, which exposes models that write
plausible text but unbalanced brackets. Results go to dataset/checkpoints/eval.

Usage:
    python dataset/tools/evaluate_sft.py --model Qwen/Qwen2.5-7B-Instruct --adapter dataset/checkpoints/sft
    python dataset/tools/evaluate_sft.py --limit 200
    python dataset/tools/evaluate_sft.py --self-check          # scores gold against gold, no model
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import logic_map  # noqa: E402

RENDER_KEYS = ("bracket_ast", "semantic_pattern", "gf_abstract", "z3_assertion")


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def gold_payload(assistant_text: str) -> Dict[str, str]:
    parsed = extract_json(assistant_text)
    if parsed:
        return {key: str(parsed[key]) for key in RENDER_KEYS if key in parsed}
    return {}


def score_row(row: Dict[str, Any], generated: str) -> Dict[str, Any]:
    gold = row["messages"][-1]["content"]
    out: Dict[str, Any] = {
        "task": row.get("task", "?"),
        "language": row.get("language", "?"),
        "domain": row.get("domain", "?"),
    }
    out["exact"] = generated.strip() == gold.strip()
    out["normalized"] = normalize(generated) == normalize(gold)

    gold_payload_map = gold_payload(gold)
    got_payload = gold_payload(generated)
    if gold_payload_map:
        bracket = gold_payload_map.get("bracket_ast", "")
        got_bracket = got_payload.get("bracket_ast", "")
        out["structure"] = bool(got_bracket) and normalize(got_bracket) == normalize(bracket)
        skeleton = re.sub(r"[^\[\]()]", "", bracket)
        out["well_formed"] = logic_map.is_well_bracketed(skeleton)
        out["payload"] = all(
            key in got_payload and normalize(got_payload[key]) == normalize(gold_payload_map[key])
            for key in gold_payload_map
        )
    else:
        out["structure"] = out["normalized"]
        out["well_formed"] = True
        out["payload"] = out["normalized"]
    return out


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    if not scores:
        return {}
    keys = ("exact", "normalized", "structure", "payload", "well_formed")
    return {key: round(100.0 * sum(1 for row in scores if row[key]) / len(scores), 2) for key in keys}


def report_lines(scores: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    overall = aggregate(scores)
    lines.append("rows evaluated: " + str(len(scores)))
    lines.append(
        "overall: exact " + str(overall["exact"]) + " | normalized " + str(overall["normalized"])
        + " | structure " + str(overall["structure"]) + " | payload " + str(overall["payload"])
        + " | well formed " + str(overall["well_formed"])
    )
    for axis in ("task", "language", "domain"):
        values = sorted({row[axis] for row in scores})
        for value in values:
            subset = aggregate([row for row in scores if row[axis] == value])
            lines.append(
                axis + "=" + value + ": exact " + str(subset["exact"]) + " | structure " + str(subset["structure"])
                + " | payload " + str(subset["payload"]) + " | well formed " + str(subset["well_formed"])
            )
    return lines


def generate_with_model(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("base model: " + args.model + " | device: " + device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel

        print("adapter: " + args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()

    answers: List[str] = []
    for index, row in enumerate(rows):
        messages = row["messages"][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answers.append(text)
        if (index + 1) % 25 == 0:
            print("generated " + str(index + 1) + " / " + str(len(rows)))
    return answers


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Score a chat model on the held-out dataset items.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--file", default="sft/test.jsonl", help="SFT split to evaluate")
    parser.add_argument("--model", default="", help="base chat model")
    parser.add_argument("--adapter", default="", help="trained LoRA adapter folder")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out-dir", default=str(DATASET_ROOT / "checkpoints" / "eval"))
    parser.add_argument("--self-check", action="store_true", help="score gold against gold, no model loaded")
    args = parser.parse_args(argv)

    root = Path(args.dataset_root).resolve()
    data_file = Path(args.file)
    if not data_file.is_absolute():
        data_file = root / data_file
    rows = read_jsonl(data_file, args.limit)
    if not rows:
        print("ERROR: no rows found in " + str(data_file))
        return 2

    if args.self_check:
        answers = [row["messages"][-1]["content"] for row in rows]
        mode = "self-check (gold against gold)"
    else:
        if not args.model:
            print("ERROR: pass --model (and --adapter when you trained one), or use --self-check")
            return 2
        answers = generate_with_model(args, rows)
        mode = "generated"

    scores = [score_row(row, answer) for row, answer in zip(rows, answers)]
    lines = report_lines(scores)
    print("mode: " + mode)
    for line in lines:
        print(line)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "file": str(data_file),
        "rows": len(rows),
        "model": args.model,
        "adapter": args.adapter,
        "overall": aggregate(scores),
        "by_row": scores,
    }
    stamp = "self_check" if args.self_check else "generated"
    (out_dir / ("results_" + stamp + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("results written to " + str(out_dir / ("results_" + stamp + ".json")))

    overall = aggregate(scores)
    passed = overall["structure"] >= (100.0 if args.self_check else 0.0)
    print("FINAL VERDICT: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
