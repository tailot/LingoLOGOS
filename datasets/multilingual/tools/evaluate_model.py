"""Held-out gates for a trained checkpoint: E1 accuracy, E2 legality, E3 formal audit, E5 ablation.

The evaluator never uses the symbolic mask as its own judge. Legality of a generated stream
is decided by ``verify_masking.reference_is_legal``, which re-derives the tree from the
bracket skeleton and the parser of ``logic_map``, and the formal gate sends the tree the
model produced to Z3 through the same builder the dataset itself was validated with.

Gates
    E1  exact tree accuracy on the held-out split, broken down by language, domain and
        logic family, plus the skeleton accuracy that separates shape from payload.
    E2  legality of every generated stream, measured with the independent oracle.
    E3  formal audit: the generated tree must be accepted by the operator registry, must be
        satisfiable and not valid in Z3, and must not invent a number the reference tree of
        that sentence does not contain.
    E5  mask ablation at identical weights: unconstrained greedy decoding under the same
        checkpoint, to show how much of the well-formedness comes from the constraint.

E4 is out of scope on purpose: the model is trained in one direction only, sentence to tree,
so no meaning side is measured here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_paths import bootstrap  # noqa: E402

REPO_ROOT, DATASET_ROOT = bootstrap(HERE)

import logic_map  # noqa: E402

from model_data import (  # noqa: E402
    build_token_mask,
    device_report,
    examples_of,
    length_profile,
)

from neuro_symbolic.masking import parse_algebra, skeleton_of_stream  # noqa: E402
from neuro_symbolic.model import load_model, load_model_metadata  # noqa: E402
from neuro_symbolic.training import (  # noqa: E402
    AllowedSetTable,
    aggregate,
    batch_of,
    encode_examples,
    generate_batch,
    make_batch,
    score_batch,
)

from verify_masking import load_registry, reference_is_legal  # noqa: E402

DEFAULT_CHECKPOINT = DATASET_ROOT / "checkpoints" / "model" / "model.pt"
DEFAULT_REPORT = DATASET_ROOT / "checkpoints" / "evaluation"


class EvaluationError(RuntimeError):
    """Raised when the checkpoint and the corpus do not describe the same machine."""


@dataclass
class Scored:
    """One held-out sentence: what the model emitted, what the dataset meant, and the verdict."""

    identifier: str
    language: str
    domain: str
    logic_family: str
    generated: List[str]
    reference: List[str]
    steps: int
    ended_at_end: bool
    mean_probability: float
    exact: bool
    exact_shape: bool
    legal: bool
    probabilities: List[float] = field(default_factory=list, repr=False)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the held-out gates of a trained model.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="weights to evaluate")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"), help="split to score")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda[:n]")
    parser.add_argument("--batch-size", type=int, default=32, help="examples per scored or decoded batch")
    parser.add_argument("--limit", type=int, default=0, help="cap the number of scored examples")
    parser.add_argument("--seed", type=int, default=23, help="seed of the decoding sampler")
    parser.add_argument("--sampling", action="store_true", help="sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--min-exact-tree-accuracy", type=float, default=0.0, help="threshold for gate E1")
    parser.add_argument(
        "--max-invented-number-rate",
        type=float,
        default=0.02,
        help="gate E3 tolerance for generated numbers absent from the reference tree",
    )
    parser.add_argument(
        "--max-noncontingent-rate",
        type=float,
        default=0.02,
        help="gate E3 tolerance for generated rules that Z3 calls unsatisfiable or valid",
    )
    parser.add_argument("--no-z3", action="store_true", help="skip the formal pass of gate E3")
    parser.add_argument("--no-ablation", action="store_true", help="skip gate E5")
    parser.add_argument("--no-refine", action="store_true", help="do not narrow the mask by argument kind")
    parser.add_argument("--out", default="", help="report path, by default datasets/multilingual/checkpoints/evaluation")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> Tuple[str, Dict[str, Any]]:
    """Device of the run: an explicit request, else the accelerator when one exists."""
    import torch

    if requested != "auto":
        report = device_report(requested)
        return str(report["resolved"]), report
    if torch.cuda.is_available():
        report = device_report("cuda:0")
        return str(report["resolved"]), report
    return "cpu", {"requested": "auto", "resolved": "cpu"}


# ---------------------------------------------------------------------------
# Scoring and decoding of one split
# ---------------------------------------------------------------------------
def teacher_forced(
    model: Any,
    token_mask: Any,
    table: AllowedSetTable,
    encoded: Sequence[Dict[str, Any]],
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Teacher-forced metrics over the whole split, batch by batch, without gradients."""
    import torch

    matrix = table.matrix(device)
    model.eval()
    metrics: List[Dict[str, Any]] = []
    with torch.no_grad():
        for chunk in batch_of(list(encoded), int(batch_size)):
            batch = make_batch(chunk, token_mask, table, device)
            _, item = score_batch(model, batch, matrix, table)
            metrics.append(item)
    return aggregate(metrics)


def decode_split(
    model: Any,
    token_mask: Any,
    encoded: Sequence[Dict[str, Any]],
    max_steps: int,
    batch_size: int,
    sampling: bool,
    temperature: float,
    seed: int,
    arities: Dict[str, int],
) -> List[Scored]:
    """Constrained decoding of the whole split, in batches to keep the memory flat."""
    gold_of = {item["identifier"]: list(item["symbols"]) for item in encoded}
    scored: List[Scored] = []
    for start in range(0, len(encoded), int(batch_size)):
        chunk = list(encoded[start : start + int(batch_size)])
        decoded = generate_batch(
            model,
            token_mask,
            chunk,
            max_steps=int(max_steps),
            mode="sample" if sampling else "greedy",
            temperature=float(temperature),
            seed=int(seed) + start,
        )
        for entry in decoded:
            reference = gold_of[entry.identifier]
            scored.append(
                Scored(
                    identifier=entry.identifier,
                    language=entry.language,
                    domain=entry.domain,
                    logic_family=entry.logic_family,
                    generated=list(entry.symbols),
                    reference=reference,
                    steps=entry.steps,
                    ended_at_end=entry.stopped_at_end,
                    mean_probability=(
                        sum(entry.probabilities) / len(entry.probabilities) if entry.probabilities else 0.0
                    ),
                    exact=entry.symbols == reference,
                    exact_shape=skeleton_of_stream(entry.symbols) == skeleton_of_stream(reference),
                    legal=reference_is_legal(entry.symbols, arities),
                    probabilities=list(entry.probabilities),
                )
            )
    return scored


def unmasked_decoding(
    model: Any,
    token_mask: Any,
    encoded: Sequence[Dict[str, Any]],
    max_steps: int,
    batch_size: int,
    arities: Dict[str, int],
) -> Dict[str, Any]:
    """Gate E5: the same weights with the constraint removed, greedy over the whole vocabulary."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    legal = 0
    ended = 0
    outside_alphabet = 0
    rows = 0
    sample_failures: List[str] = []
    with torch.no_grad():
        for start in range(0, len(encoded), int(batch_size)):
            chunk = list(encoded[start : start + int(batch_size)])
            width = max(len(item["input_ids"]) for item in chunk)
            input_ids = torch.full((len(chunk), width), token_mask.pad_id, dtype=torch.long).to(device)
            input_mask = torch.zeros((len(chunk), width), dtype=torch.bool).to(device)
            for row, item in enumerate(chunk):
                source = item["input_ids"]
                input_ids[row, : len(source)] = torch.tensor(source, dtype=torch.long, device=device)
                input_mask[row, : len(source)] = True
            memory = model.encode(input_ids, input_mask)
            prefixes: List[List[int]] = [[token_mask.rule_start_id] for _ in chunk]
            alive = [True] * len(chunk)
            for _ in range(int(max_steps)):
                active = [row for row, flag in enumerate(alive) if flag]
                if not active:
                    break
                longest = max(len(prefixes[row]) for row in active)
                decoder_input = torch.full(
                    (len(active), longest), token_mask.pad_id, dtype=torch.long, device=device
                )
                for slot, row in enumerate(active):
                    prefix = prefixes[row]
                    decoder_input[slot, : len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=device)
                rows_index = torch.tensor(active, dtype=torch.long, device=device)
                logits = model.decode(decoder_input, memory[rows_index], input_mask[rows_index])
                for slot, row in enumerate(active):
                    chosen = int(torch.argmax(logits[slot, len(prefixes[row]) - 1]).item())
                    prefixes[row].append(chosen)
                    if chosen == token_mask.eos_id:
                        alive[row] = False
            for protocol in prefixes:
                rows += 1
                stream = token_mask.tokenizer.decode_symbol_stream(protocol)
                if any(token not in token_mask.symbol_ids for token in stream):
                    outside_alphabet += 1
                finished = protocol[-1] == token_mask.eos_id
                ended += 1 if finished else 0
                if finished and reference_is_legal(stream, arities):
                    legal += 1
                elif len(sample_failures) < 5:
                    sample_failures.append(" ".join(stream[:24]))
    return {
        "examples": rows,
        "legal_streams": legal,
        "legality": legal / max(1, rows),
        "ended_at_end_marker": ended,
        "ended_at_end_rate": ended / max(1, rows),
        "rows_outside_the_algebra_alphabet": outside_alphabet,
        "failure_examples": sample_failures,
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def numeric_leaves(tree: Any) -> Set[float]:
    """The numbers a tree commits to, read from its leaves and not from the surface text."""
    return {
        float(logic_map.numeric_value(node))
        for node in logic_map.walk(tree)
        if logic_map.is_leaf_node(node) and logic_map.is_numeric_leaf(node)
    }


def accuracy_breakdown(scored: Sequence[Scored], attribute: str) -> Dict[str, Any]:
    """Exact-tree accuracy grouped by one field of the dataset row."""
    buckets: Dict[str, List[int]] = {}
    for entry in scored:
        bucket = str(getattr(entry, attribute) or "unknown")
        counts = buckets.setdefault(bucket, [0, 0])
        counts[0] += int(entry.exact)
        counts[1] += 1
    return {
        bucket: {"examples": total, "exact_trees": good, "accuracy": round(good / max(1, total), 4)}
        for bucket, (good, total) in sorted((key, tuple(value)) for key, value in buckets.items())
    }


def accuracy_gate(scored: Sequence[Scored]) -> Dict[str, Any]:
    """E1: tree and skeleton accuracy of the decoding, with its length profile."""
    if not scored:
        return {
            "examples": 0,
            "exact_trees": 0,
            "exact_tree_accuracy": 0.0,
            "exact_skeleton_accuracy": 0.0,
            "ended_at_end_marker": 0,
            "truncated": 0,
            "steps": {"minimum": 0, "mean": 0.0, "median": 0, "maximum": 0},
            "mean_selected_probability": 0.0,
        }
    steps = sorted(entry.steps for entry in scored)
    with_probability = [entry for entry in scored if entry.probabilities]
    return {
        "examples": len(scored),
        "exact_trees": sum(1 for entry in scored if entry.exact),
        "exact_tree_accuracy": sum(1 for entry in scored if entry.exact) / max(1, len(scored)),
        "exact_skeleton_accuracy": sum(1 for entry in scored if entry.exact_shape) / max(1, len(scored)),
        "ended_at_end_marker": sum(1 for entry in scored if entry.ended_at_end),
        "truncated": sum(1 for entry in scored if not entry.ended_at_end),
        "steps": {
            "minimum": steps[0],
            "mean": round(sum(steps) / len(steps), 2),
            "median": steps[len(steps) // 2],
            "maximum": steps[-1],
        },
        "mean_selected_probability": round(
            sum(entry.mean_probability for entry in with_probability) / max(1, len(with_probability)), 4
        ),
    }


def legality_gate(scored: Sequence[Scored]) -> Dict[str, Any]:
    """E2: every generated stream checked by the skeleton oracle and the parser."""
    failures = [entry.identifier + ": " + " ".join(entry.generated) for entry in scored if not entry.legal]
    return {
        "examples": len(scored),
        "legal_streams": sum(1 for entry in scored if entry.legal),
        "legality": sum(1 for entry in scored if entry.legal) / max(1, len(scored)),
        "failures": failures[:10],
    }


def formal_gate(
    scored: Sequence[Scored],
    registry: Dict[str, Dict[str, Any]],
    use_z3: bool,
) -> Dict[str, Any]:
    """E3: registry acceptance, Z3 contingency, and the invented-number audit."""
    trees: Dict[str, Any] = {}
    rejected: List[str] = []
    for entry in scored:
        if not entry.legal:
            continue
        tree = parse_algebra(" ".join(entry.generated))
        try:
            logic_map.check_tree(tree, registry)
        except Exception as error:  # a tree the registry rejects fails the gate, not the run
            rejected.append(entry.identifier + ": " + type(error).__name__ + " " + str(error))
            continue
        trees[entry.identifier] = (tree, entry)

    invented: List[str] = []
    numbers_checked = 0
    for identifier, (tree, entry) in trees.items():
        try:
            produced = numeric_leaves(tree)
            expected = numeric_leaves(parse_algebra(" ".join(entry.reference)))
        except Exception as error:
            invented.append(identifier + ": " + type(error).__name__ + " " + str(error))
            continue
        numbers_checked += 1
        if not produced <= expected:
            invented.append(identifier + ": " + ", ".join(sorted(str(value) for value in produced - expected)))

    report: Dict[str, Any] = {
        "trees_audited": len(trees),
        "registry_rejections": len(rejected),
        "registry_rejection_examples": rejected[:5],
        "numbers_checked": numbers_checked,
        "invented_numbers": len(invented),
        "invented_number_rate": len(invented) / max(1, numbers_checked),
        "invented_number_examples": invented[:5],
    }
    if not use_z3:
        report["z3"] = "skipped"
        return report
    try:
        import z3
    except ImportError:
        report["z3"] = "not installed"
        return report

    solver = z3.Solver()
    refuter = z3.Solver()
    unsatisfiable: List[str] = []
    valid: List[str] = []
    broken: List[str] = []
    contingent = 0
    started = time.time()
    for identifier, (tree, _entry) in trees.items():
        try:
            assertion = logic_map.build_z3(tree, z3, registry, {})
        except Exception as error:
            broken.append(identifier + ": " + type(error).__name__ + " " + str(error))
            continue
        solver.push()
        solver.add(assertion)
        satisfiable = solver.check()
        refuter.push()
        refuter.add(z3.Not(assertion))
        refutable = refuter.check()
        solver.pop()
        refuter.pop()
        if satisfiable != z3.sat:
            unsatisfiable.append(identifier)
        elif refutable != z3.sat:
            valid.append(identifier)
        else:
            contingent += 1
    report["z3"] = {
        "forms": len(trees),
        "contingent": contingent,
        "unsatisfiable": len(unsatisfiable),
        "valid": len(valid),
        "build_errors": len(broken),
        "noncontingent": len(unsatisfiable) + len(valid),
        "noncontingent_rate": (len(unsatisfiable) + len(valid)) / max(1, len(trees)),
        "unsatisfiable_examples": unsatisfiable[:5],
        "valid_examples": valid[:5],
        "build_error_examples": broken[:5],
        "seconds": round(time.time() - started, 1),
    }
    return report


def print_verdicts(report: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Print one line per gate and return the overall verdict."""
    gates = report["gates"]
    e1, e2, e3 = gates["E1"], gates["E2"], gates["E3"]
    verdict = True
    recorded: Dict[str, Any] = {}

    passed = e1["exact_tree_accuracy"] >= float(args.min_exact_tree_accuracy)
    verdict = verdict and passed
    print(
        "GATE E1 " + ("PASS" if passed else "FAIL")
        + " | exact tree " + format(e1["exact_tree_accuracy"], ".4f")
        + " of " + str(e1["examples"])
        + " | threshold " + format(float(args.min_exact_tree_accuracy), ".4f")
        + " | skeleton " + format(e1["exact_skeleton_accuracy"], ".4f")
        + " | truncated " + str(e1["truncated"])
    )
    recorded["E1"] = {
        "passed": passed,
        "exact_tree_accuracy": e1["exact_tree_accuracy"],
        "threshold": float(args.min_exact_tree_accuracy),
    }
    for bucket, values in e1["by_language"].items():
        print("  " + bucket + ": " + format(values["accuracy"], ".4f") + " over " + str(values["examples"]))

    passed = e2["legal_streams"] == e2["examples"]
    verdict = verdict and passed
    print("GATE E2 " + ("PASS" if passed else "FAIL") + " | legal streams " + str(e2["legal_streams"]) + "/" + str(e2["examples"]))
    for message in e2["failures"][:5]:
        print("  illegal: " + message)
    recorded["E2"] = {"passed": passed, "legal_streams": e2["legal_streams"], "examples": e2["examples"]}

    z3 = e3["z3"]
    structural = e3["registry_rejections"] == 0 and (not isinstance(z3, dict) or z3["build_errors"] == 0)
    passed = (
        structural
        and e3["invented_number_rate"] <= float(args.max_invented_number_rate)
        and (
            not isinstance(z3, dict)
            or z3["noncontingent_rate"] <= float(args.max_noncontingent_rate)
        )
    )
    verdict = verdict and passed
    detail = (
        " | audited " + str(e3["trees_audited"])
        + " | registry rejections " + str(e3["registry_rejections"])
        + " | invented numbers " + str(e3["invented_numbers"])
        + " (" + format(e3["invented_number_rate"], ".4f") + ")"
    )
    if isinstance(z3, dict):
        detail += (
            " | contingent " + str(z3["contingent"])
            + " | unsatisfiable " + str(z3["unsatisfiable"])
            + " | valid " + str(z3["valid"])
            + " | build errors " + str(z3["build_errors"])
            + " | " + format(z3["seconds"], ".1f") + " s"
        )
    else:
        detail += " | formal pass " + str(z3)
    print("GATE E3 " + ("PASS" if passed else "FAIL") + detail)
    recorded["E3"] = {
        "passed": passed,
        "structural": structural,
        "max_invented_number_rate": float(args.max_invented_number_rate),
        "max_noncontingent_rate": float(args.max_noncontingent_rate),
    }

    if "E5" in gates:
        e5 = gates["E5"]
        passed = e2["legality"] > e5["legality"]
        verdict = verdict and passed
        print(
            "GATE E5 " + ("PASS" if passed else "FAIL")
            + " | legality with mask " + format(e2["legality"], ".4f")
            + " | legality without mask " + format(e5["legality"], ".4f")
            + " | rows outside the algebra alphabet " + str(e5["rows_outside_the_algebra_alphabet"])
        )
        recorded["E5"] = {
            "passed": passed,
            "legality_with_mask": e2["legality"],
            "legality_without_mask": e5["legality"],
        }
    report["verdicts"] = recorded
    report["final_verdict"] = "PASS" if verdict else "FAIL"
    return verdict


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise EvaluationError("no checkpoint at " + str(checkpoint))

    device, hardware = resolve_device(args.device)
    token_mask, mask_report, arities = build_token_mask(refine=not args.no_refine)
    model = load_model(str(checkpoint), device=device)
    if int(model.config.vocabulary_size) != int(token_mask.tokenizer.vocabulary_size):
        raise EvaluationError("the checkpoint vocabulary differs from the learned tokenizer")

    examples = examples_of(args.split, args.limit)
    if not examples:
        raise EvaluationError("no examples in split " + args.split)
    table = AllowedSetTable(token_mask)
    encoded = encode_examples(examples, token_mask, table)
    budget = int(model.config.max_decoder_positions)
    longest = max(len(item["target"]) for item in encoded)
    if longest > budget:
        raise EvaluationError("the decoder position budget can not hold the longest protocol")

    report: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": load_model_metadata(str(checkpoint)),
        "model_config": model.config.to_dict(),
        "device": hardware,
        "split": args.split,
        "examples": len(encoded),
        "mask_report": mask_report,
        "admissible_sets": table.describe(),
        "token_lengths": length_profile(encoded),
        "decoding": {
            "mode": "sample" if args.sampling else "greedy",
            "temperature": float(args.temperature),
            "max_steps": budget - 1,
            "position_budget": budget,
            "longest_protocol": longest,
            "seed": int(args.seed),
        },
    }
    print("checkpoint: " + checkpoint.name + " | " + json.dumps(report["checkpoint_metadata"], sort_keys=True))
    print("device: " + str(hardware.get("resolved")) + " | split " + args.split + " | examples " + str(len(encoded)))

    forced = teacher_forced(model, token_mask, table, encoded, args.batch_size, device)
    report["teacher_forced"] = forced
    print(
        "teacher forced: loss {loss:.4f} | perplexity {perplexity:.3f} | free-position accuracy {free:.3f}"
        " | inadmissible argmax {bad:.4f}".format(
            loss=forced["loss"],
            perplexity=forced["perplexity"],
            free=forced["free_position_accuracy"],
            bad=forced["illegal_unmasked_argmax_rate"],
        )
    )

    scored = decode_split(
        model,
        token_mask,
        encoded,
        max_steps=budget - 1,
        batch_size=args.batch_size,
        sampling=bool(args.sampling),
        temperature=float(args.temperature),
        seed=int(args.seed),
        arities=arities,
    )
    report["gates"] = {"E1": accuracy_gate(scored)}
    report["gates"]["E1"]["by_language"] = accuracy_breakdown(scored, "language")
    report["gates"]["E1"]["by_domain"] = accuracy_breakdown(scored, "domain")
    report["gates"]["E1"]["by_logic_family"] = accuracy_breakdown(scored, "logic_family")
    report["gates"]["E2"] = legality_gate(scored)
    report["gates"]["E3"] = formal_gate(scored, load_registry(), use_z3=not args.no_z3)
    if not args.no_ablation:
        report["gates"]["E5"] = unmasked_decoding(
            model, token_mask, encoded, budget - 1, args.batch_size, arities
        )

    verdict = print_verdicts(report, args)

    out = Path(args.out).resolve() if args.out else DEFAULT_REPORT / (checkpoint.stem + "-" + args.split + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("report: " + str(out))
    print("FINAL VERDICT " + ("PASS" if verdict else "FAIL"))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
