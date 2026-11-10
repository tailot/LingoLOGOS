"""Train the pure encoder-decoder on the bracket-word direction of the corpus.

The run is the fourth stage of ``doc/PURE_MODEL_ROADMAP.md``: weights come from the scratch
initialization of ``neuro_symbolic.model``, the objective is the masked cross-entropy of
``neuro_symbolic.training``, and the optimizer follows the specification of the design note
(AdamW with beta 0.9 / 0.98, epsilon 1e-9, peak rate 5e-4, linear warmup over the first
epoch, cosine decay to a tenth of the peak, weight decay 0.01 off biases and gains, global
gradient norm clipped at 1.0).

Usage
-----
    python datasets/multilingual/tools/train_model.py --dry-run
    python datasets/multilingual/tools/train_model.py --epochs 20 --batch-size 64 --amp

Outputs land in ``datasets/multilingual/checkpoints/model``: ``model.pt`` with the last weights,
``model_best.pt`` with the lowest validation loss, ``model_config.json``, ``history.json``
and ``train_metrics.json``. The held-out test split is never read here; the evaluation tool
owns it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_paths import bootstrap  # noqa: E402

REPO_ROOT, DATASET_ROOT = bootstrap(HERE)

import torch  # noqa: E402

from neuro_symbolic.model import (  # noqa: E402
    EncoderDecoderModel,
    ModelConfig,
    apply_initialization,
    initialization_report,
    save_model,
)
from neuro_symbolic.training import (  # noqa: E402
    AllowedSetTable,
    aggregate,
    batch_of,
    build_optimizer,
    cosine_schedule,
    encode_examples,
    generate_batch,
    learning_rates,
    make_batch,
    score_batch,
    training_step,
)

import model_data  # noqa: E402


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pure model of the roadmap on the corpus.")
    parser.add_argument("--tokenizer", default=str(model_data.DEFAULT_TOKENIZER), help="learned tokenizer to reuse")
    parser.add_argument("--epochs", type=int, default=20, help="full passes over the train split")
    parser.add_argument("--batch-size", type=int, default=64, help="examples per optimization step")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="examples per validation step")
    parser.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=0, help="warmup updates, 0 selects one epoch")
    parser.add_argument("--final-lr-fraction", type=float, default=0.1, help="cosine floor as a fraction of the peak")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW decoupled weight decay")
    parser.add_argument("--clip", type=float, default=1.0, help="global gradient norm cap, 0 disables it")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout of every residual block")
    parser.add_argument("--seed", type=int, default=17, help="seed of initialization and shuffling")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda[:n]")
    parser.add_argument("--amp", default="auto", choices=("auto", "off", "bf16"), help="bfloat16 autocast")
    parser.add_argument("--limit", type=int, default=0, help="read only the first N rows of each split")
    parser.add_argument("--no-refine", action="store_true", help="skip the argument-kind narrowing of the mask")
    parser.add_argument("--eval-every", type=int, default=1, help="validation period in epochs")
    parser.add_argument("--generation-samples", type=int, default=16, help="validation items decoded per epoch, 0 off")
    parser.add_argument("--out-dir", default=str(DATASET_ROOT / "checkpoints" / "model"), help="checkpoint directory")
    parser.add_argument("--dry-run", action="store_true", help="short smoke run on a small architecture, no files")
    parser.add_argument("--d-model", type=int, default=512, help="residual width")
    parser.add_argument("--encoder-layers", type=int, default=6, help="encoder block count")
    parser.add_argument("--decoder-layers", type=int, default=6, help="decoder block count")
    parser.add_argument("--attention-heads", type=int, default=8, help="attention heads per block")
    parser.add_argument("--feed-forward-size", type=int, default=2048, help="inner width of the feed forward")
    parser.add_argument("--max-encoder-positions", type=int, default=96, help="encoder position budget")
    parser.add_argument("--max-decoder-positions", type=int, default=96, help="decoder position budget")
    return parser.parse_args(argv)


def resolve(args: argparse.Namespace) -> Dict[str, Any]:
    """Fold the dry-run shorthand into the real settings."""
    settings = dict(vars(args))
    if args.dry_run:
        settings.update(
            {
                "epochs": min(3, int(args.epochs or 3)),
                "d_model": 128,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "attention_heads": 4,
                "feed_forward_size": 256,
                "batch_size": 16,
                "eval_batch_size": 16,
                "limit": args.limit or 384,
                "generation_samples": min(8, int(args.generation_samples)),
            }
        )
    return settings


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    settings = resolve(args)
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)

    hardware = model_data.device_report(
        ("cuda" if torch.cuda.is_available() else "cpu") if settings["device"] == "auto" else str(settings["device"])
    )
    device = str(hardware["resolved"])
    print("device: " + device + " | " + str(hardware.get("name", "cpu")) + " | bytes " + str(hardware.get("total_memory_bytes", 0)))

    token_mask, mask_report, arities = model_data.build_token_mask(refine=not settings["no_refine"])
    print("vocabulary: " + str(token_mask.tokenizer.vocabulary_size) + " tokens")
    print(
        "mask alphabet: operators " + str(mask_report["operators"])
        + ", leaf starts " + str(mask_report["leaf_first_tokens"])
        + ", leaf values " + str(mask_report["leaf_value_tokens"])
        + ", kind-narrowed positions " + str(mask_report["child_kind_positions"])
    )

    table = AllowedSetTable(token_mask)
    train_examples = model_data.examples_of("train", settings["limit"])
    valid_examples = model_data.examples_of("valid", settings["limit"])
    print("supervised pairs: train " + str(len(train_examples)) + ", valid " + str(len(valid_examples)))
    train_encoded = encode_examples(train_examples, token_mask, table)
    valid_encoded = encode_examples(valid_examples, token_mask, table)
    print("admissible sets: " + json.dumps(table.describe(), sort_keys=True))

    profile = model_data.length_profile(train_encoded)
    print("token lengths: " + json.dumps(profile, sort_keys=True))
    needed_encoder = profile["encoder_input"]["maximum"]
    needed_decoder = profile["decoder_protocol"]["maximum"]
    if needed_encoder > int(settings["max_encoder_positions"]):
        print("ERROR: encoder budget " + str(settings["max_encoder_positions"]) + " below the measured " + str(needed_encoder))
        return 2
    if needed_decoder > int(settings["max_decoder_positions"]):
        print("ERROR: decoder budget " + str(settings["max_decoder_positions"]) + " below the measured " + str(needed_decoder))
        return 2

    config = ModelConfig(
        vocabulary_size=token_mask.tokenizer.vocabulary_size,
        d_model=int(settings["d_model"]),
        encoder_layers=int(settings["encoder_layers"]),
        decoder_layers=int(settings["decoder_layers"]),
        attention_heads=int(settings["attention_heads"]),
        feed_forward_size=int(settings["feed_forward_size"]),
        dropout=float(settings["dropout"]),
        max_encoder_positions=int(settings["max_encoder_positions"]),
        max_decoder_positions=int(settings["max_decoder_positions"]),
    )
    model = apply_initialization(EncoderDecoderModel(config, padding_id=token_mask.pad_id), seed=seed)
    model.to(device)
    breakdown = model.parameter_breakdown()
    print("parameters: total " + str(breakdown["total"]) + " | " + json.dumps(breakdown, sort_keys=True))
    print("initialization: " + json.dumps(initialization_report(model), sort_keys=True))

    allowed_matrix = table.matrix(device)
    optimizer = build_optimizer(model, settings["lr"], settings["weight_decay"])
    steps_per_epoch = max(1, (len(train_encoded) + int(settings["batch_size"]) - 1) // int(settings["batch_size"]))
    total_steps = steps_per_epoch * int(settings["epochs"])
    warmup = int(settings["warmup_steps"]) or steps_per_epoch
    scheduler = cosine_schedule(optimizer, warmup, total_steps, settings["final_lr_fraction"])
    print(
        "schedule: " + str(steps_per_epoch) + " updates per epoch, warmup " + str(warmup)
        + ", total " + str(total_steps) + ", peak lr " + str(settings["lr"])
    )

    amp_dtype = None
    amp_choice = str(settings["amp"])
    if amp_choice == "bf16" or (amp_choice == "auto" and device.startswith("cuda")):
        amp_dtype = torch.bfloat16
    print("autocast: " + ("bf16" if amp_dtype is not None else "off"))

    if settings["dry_run"]:
        rates = learning_rates(scheduler, min(total_steps, 6))
        print("dry-run rate multipliers: " + ", ".join("{:.3f}".format(value) for value in rates))

    history: List[Dict[str, Any]] = []
    best_validation = float("inf")
    step = 0
    generation_max_steps = needed_decoder
    started = time.perf_counter()
    for epoch in range(1, int(settings["epochs"]) + 1):
        order = list(range(len(train_encoded)))
        random.Random(seed + epoch).shuffle(order)
        shuffled = [train_encoded[index] for index in order]
        epoch_metrics: List[Dict[str, Any]] = []
        epoch_started = time.perf_counter()
        samples_this_epoch = 0
        for chunk in batch_of(shuffled, settings["batch_size"]):
            batch = make_batch(chunk, token_mask, table, device)
            metrics = training_step(
                model, batch, allowed_matrix, table, optimizer, scheduler, settings["clip"], amp_dtype
            )
            epoch_metrics.append(metrics)
            step += 1
            samples_this_epoch += len(chunk)
        elapsed = time.perf_counter() - epoch_started
        summary = aggregate(epoch_metrics)
        positions = sum(item["positions"] for item in epoch_metrics)
        summary.update(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": round(elapsed, 3),
                "samples_per_second": round(samples_this_epoch / max(1e-9, elapsed), 2),
                "positions_per_second": round(positions / max(1e-9, elapsed), 2),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0,
            }
        )
        line = (
            "epoch " + str(epoch) + " | loss " + "{:.4f}".format(summary["loss"])
            + " | ppl " + "{:.2f}".format(summary["perplexity"])
            + " | exact " + "{:.3f}".format(summary["exact_sequence_accuracy"])
            + " | free-position acc " + "{:.3f}".format(summary["free_position_accuracy"])
            + " | " + "{:.0f}".format(summary["samples_per_second"]) + " samples/s"
            + " | lr " + "{:.2e}".format(summary["learning_rate"])
        )
        if epoch % int(settings["eval_every"]) == 0 or epoch == int(settings["epochs"]):
            validation = evaluate(model, valid_encoded, token_mask, table, allowed_matrix, settings["eval_batch_size"], device)
            summary["validation"] = validation
            line += " | val loss " + "{:.4f}".format(validation["loss"]) + " | val exact " + "{:.3f}".format(
                validation["exact_sequence_accuracy"]
            )
            if validation["loss"] < best_validation:
                best_validation = float(validation["loss"])
                if not settings["dry_run"]:
                    out_dir = Path(settings["out_dir"])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_model(
                        model,
                        str(out_dir / "model_best.pt"),
                        metadata=metadata_of(settings, config, breakdown, mask_report, profile, epoch, best_validation, hardware),
                    )
        if int(settings["generation_samples"]) and (epoch % int(settings["eval_every"]) == 0 or epoch == int(settings["epochs"])):
            decoded = decode_report(
                model, token_mask, valid_encoded[: int(settings["generation_samples"])], generation_max_steps, seed
            )
            summary["generation"] = decoded
            line += " | decoded exact " + "{:.3f}".format(decoded["exact_stream_accuracy"])
        history.append(summary)
        print(line)
        del samples_this_epoch

    total_seconds = time.perf_counter() - started
    summary_final = history[-1]
    if not settings["dry_run"]:
        out_dir = Path(settings["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        save_model(
            model,
            str(out_dir / "model.pt"),
            metadata=metadata_of(settings, config, breakdown, mask_report, profile, int(settings["epochs"]), best_validation, hardware),
        )
        (out_dir / "model_config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        report = {
            "config": config.to_dict(),
            "settings": {key: value for key, value in settings.items() if key != "dry_run"},
            "parameters": breakdown,
            "initialization": initialization_report(model),
            "mask_report": mask_report,
            "admissible_sets": table.describe(),
            "length_profile": profile,
            "hardware": hardware,
            "steps": step,
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": warmup,
            "total_seconds": round(total_seconds, 2),
            "samples_per_second": summary_final["samples_per_second"],
            "positions_per_second": summary_final["positions_per_second"],
            "peak_memory_bytes": summary_final["peak_memory_bytes"],
            "best_validation_loss": best_validation,
            "history": history,
        }
        (out_dir / "train_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("checkpoints written to " + str(out_dir))
    print(
        "training finished in " + "{:.1f}".format(total_seconds) + " s | best validation loss "
        + "{:.4f}".format(best_validation) + " | " + "{:.0f}".format(summary_final["samples_per_second"]) + " samples/s sustained"
    )
    return 0


def metadata_of(
    settings: Dict[str, Any],
    config: ModelConfig,
    breakdown: Dict[str, int],
    mask_report: Dict[str, Any],
    profile: Dict[str, Any],
    epoch: int,
    best_validation: float,
    hardware: Dict[str, Any],
) -> Dict[str, Any]:
    """Provenance stored next to the weights: what was trained, on what, with which mask."""
    return {
        "trained_at_epoch": int(epoch),
        "best_validation_loss": float(best_validation),
        "settings": {key: value for key, value in settings.items() if key != "dry_run"},
        "parameters": breakdown,
        "mask_report": mask_report,
        "length_profile": profile,
        "hardware": hardware,
        "config": config.to_dict(),
        "tokenizer": str(settings.get("tokenizer")),
        "vocabulary_size": int(config.vocabulary_size),
        "external_weights": "none",
    }


def evaluate(
    model: EncoderDecoderModel,
    encoded: List[Dict[str, Any]],
    token_mask: Any,
    table: AllowedSetTable,
    allowed_matrix: torch.Tensor,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Teacher-forced validation metrics, computed without gradients."""
    metrics: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for chunk in batch_of(encoded, batch_size):
            batch = make_batch(chunk, token_mask, table, device)
            loss, item = score_batch(model, batch, allowed_matrix, table)
            del loss
            metrics.append(item)
    return aggregate(metrics)


def decode_report(
    model: EncoderDecoderModel,
    token_mask: Any,
    encoded: List[Dict[str, Any]],
    max_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Greedy constrained decoding on a few validation items, as an early health signal."""
    results = generate_batch(model, token_mask, encoded, max_steps, mode="greedy", seed=seed)
    exact = 0
    steps: List[int] = []
    for result, item in zip(results, encoded):
        steps.append(result.steps)
        if result.symbols == item["symbols"]:
            exact += 1
    return {
        "items": len(results),
        "exact_stream_accuracy": exact / max(1, len(results)),
        "ended_at_eos": sum(1 for result in results if result.stopped_at_end) / max(1, len(results)),
        "mean_steps": round(sum(steps) / max(1, len(steps)), 2),
        "maximum_steps": max(steps) if steps else 0,
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
