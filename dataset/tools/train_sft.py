"""Supervised fine-tuning of a chat LLM on the phrase / logical-form pairs of this dataset.

The training pairs live in sft/train.jsonl and sft/valid.jsonl. Every row carries a
messages list: a system prompt, a user prompt and an assistant target. The assistant
span is the only part that contributes to the loss, so the model learns the mapping
from a natural recurring phrase to the bracket algebra and the formal renderings.

The script trains a LoRA adapter with a plain PyTorch loop (no trainer framework), so
it behaves the same across library versions. Nothing in the parent repository is read
or written: the adapter goes to dataset/checkpoints/sft.

Usage:
    python dataset/tools/train_sft.py --model Qwen/Qwen2.5-7B-Instruct
    python dataset/tools/train_sft.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 400 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


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


class PairedDataset(Dataset):
    """Tokenized prompt/completion pairs with loss masking on the prompt part."""

    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples: List[Dict[str, Any]] = []
        skipped = 0
        for row in rows:
            messages = row["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                skipped += 1
                continue
            prompt_ids = tokenizer(
                tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False,
                truncation=True,
                max_length=max_length - 8,
            )["input_ids"]
            full_ids = tokenizer(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            if len(full_ids) <= len(prompt_ids):
                skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
            self.examples.append({"input_ids": full_ids, "labels": labels})
        if skipped:
            print("skipped rows without an assistant target: " + str(skipped))
        print("tokenized examples: " + str(len(self.examples)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


def make_collator(tokenizer: Any):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)
        input_ids: List[List[int]] = []
        labels: List[List[int]] = []
        attention: List[List[int]] = []
        for item in batch:
            fill = len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * (width - fill))
            labels.append(item["labels"] + [-100] * (width - fill))
            attention.append([1] * fill + [0] * (width - fill))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on the dataset SFT pairs.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="base chat model")
    parser.add_argument("--out-dir", default=str(DATASET_ROOT / "checkpoints" / "sft"))
    parser.add_argument("--train-file", default="sft/train.jsonl")
    parser.add_argument("--valid-file", default="sft/valid.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="cap rows per split (0 = all)")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50, help="max evaluation batches (0 = all)")
    parser.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (0 = only at the end)")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="force CPU even when CUDA is available")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="only validate the SFT files and exit (no model download, no heavy imports)",
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.dataset_root).resolve()
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    if not train_path.is_absolute():
        train_path = root / train_path
    if not valid_path.is_absolute():
        valid_path = root / valid_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if args.check_data:
        rows = read_jsonl(train_path, args.limit) + read_jsonl(valid_path, args.limit)
        usable = 0
        tasks: Dict[str, int] = {}
        for row in rows:
            messages = row.get("messages", [])
            task = str(row.get("task", "?"))
            tasks[task] = tasks.get(task, 0) + 1
            roles = [message.get("role") for message in messages]
            if messages and roles[-1] == "assistant" and all(role in ("system", "user", "assistant") for role in roles):
                usable += 1
        ok = bool(rows) and usable == len(rows)
        print("sft rows read: " + str(len(rows)) + " | usable for masked LM training: " + str(usable))
        print("tasks: " + ", ".join(key + "=" + str(tasks[key]) for key in sorted(tasks)))
        print("FINAL VERDICT: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1

    from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
    from transformers import (  # noqa: E402
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("base model: " + args.model)
    print("device: " + device + " | dtype: " + str(dtype))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not getattr(tokenizer, "chat_template", None):
        print("ERROR: the base model has no chat template, pick an Instruct/Chat model")
        return 2

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)

    train_rows = read_jsonl(train_path, args.limit)
    valid_rows = read_jsonl(valid_path, max(args.limit, 200))
    print("train rows: " + str(len(train_rows)) + " | valid rows: " + str(len(valid_rows)))
    train_set = PairedDataset(train_rows, tokenizer, args.max_seq_length)
    valid_set = PairedDataset(valid_rows, tokenizer, args.max_seq_length)
    collate = make_collator(tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_set) / args.batch_size))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs / args.grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_fraction),
        num_training_steps=total_steps,
    )
    print("steps per epoch: " + str(steps_per_epoch) + " | optimizer steps: " + str(total_steps))

    def evaluate() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for index, batch in enumerate(valid_loader):
                if args.eval_steps and index >= args.eval_steps:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model(**batch).loss
                total += float(loss.item())
                count += 1
        model.train()
        return total / max(count, 1)

    model.train()
    step = 0
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(math.ceil(args.epochs))):
        for micro, batch in enumerate(train_loader):
            if (epoch * steps_per_epoch + micro) >= args.epochs * steps_per_epoch:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss.item()) * args.grad_accum
            if (micro + 1) % args.grad_accum == 0 or (micro + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    print(
                        "step " + str(step) + " | loss " + format(running / args.log_every / args.grad_accum, ".4f")
                        + " | lr " + format(scheduler.get_last_lr()[0], ".2e")
                    )
                    running = 0.0
                if args.eval_every and step % args.eval_every == 0:
                    print("  eval loss " + format(evaluate(), ".4f"))
                if args.save_every and step % args.save_every == 0:
                    interim = out_dir / ("step-" + str(step))
                    interim.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(interim))
                    print("  adapter saved to " + str(interim))
    final_valid = evaluate()
    print("final valid loss: " + format(final_valid, ".4f"))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "training_config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "train_file": str(train_path),
                "valid_file": str(valid_path),
                "rows_train": len(train_rows),
                "rows_valid": len(valid_rows),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_seq_length": args.max_seq_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "seed": args.seed,
                "device": device,
                "dtype": str(dtype),
                "optimizer_steps": step,
                "final_valid_loss": final_valid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("adapter saved to " + str(out_dir))
    print("FINAL VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
