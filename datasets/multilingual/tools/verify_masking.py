"""Verify the constrained decoding mask: everything it allows is legal, everything illegal is refused.

The mask in ``src/neuro_symbolic/masking.py`` is what removes structural
hallucination, so it gets the same kind of hard acceptance gates the rest of the
repository uses. Run it after ``datasets/multilingual/tools/train_tokenizer.py``, because gate
M5 works on the learned vocabulary.

Usage:
    python datasets/multilingual/tools/train_tokenizer.py
    python datasets/multilingual/tools/verify_masking.py
    python datasets/multilingual/tools/verify_masking.py --samples 400 --seed 7
    python datasets/multilingual/tools/verify_masking.py --refine
    python datasets/multilingual/tools/verify_masking.py --list-only

Gates
-----
M1 the bracket word of every held-out item is accepted by the symbol-level mask
M2 every corrupted bracket word is refused (bracket swap, delete, insert, missing
   child, wrong operator, unknown leaf, over-long leaf, truncated stream)
M3 randomly sampled mask-constrained streams are all well-formed trees
M4 the operator registry and the corpus agree, and the mask serialization is
   byte-identical to the stored bracket words
M5 on the learned vocabulary the token-level mask admits the gold target token by
   token, never offers the closing marker before the tree is complete, and offers
   it exactly once the tree is complete

``--refine`` additionally learns which node kind fills each operator argument
position and reports whether the narrowed mask still covers the held-out items.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_paths import bootstrap  # noqa: E402

REPO_ROOT, DATASET_ROOT = bootstrap(HERE)

from neuro_symbolic.masking import (  # noqa: E402
    AlgebraMask,
    AlgebraMaskError,
    TokenStreamMask,
    algebra_word,
    parse_algebra,
    skeleton_of_stream,
    stream_of_tree,
    tokenize_algebra,
)
from neuro_symbolic.tokenizer import Tokenizer, TokenizerError  # noqa: E402

import logic_map  # noqa: E402

MAX_LEAF_TOKENS = 3


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def item_rows(scope: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for language in sorted((DATASET_ROOT / "data").iterdir()):
        if not language.is_dir():
            continue
        for path in sorted(language.glob(scope + "-*.jsonl")):
            rows.extend(read_jsonl(path))
    return rows


def load_registry() -> Dict[str, Dict[str, Any]]:
    """Operator registry of the dataset: name to its specification."""
    registry = json.loads((DATASET_ROOT / "templates" / "operators.json").read_text(encoding="utf-8"))
    return dict(registry["operators"])


def collect_leaves(rows: List[Dict[str, Any]]) -> List[List[str]]:
    payloads: List[List[str]] = []

    def walk(node: Dict[str, Any]) -> None:
        if "leaf" in node:
            payloads.append([str(token) for token in node["leaf"]])
            return
        for child in node.get("args", []):
            walk(child)

    for row in rows:
        walk(row["ast_json"])
    return payloads


# ---------------------------------------------------------------------------
# Negative controls over the symbol stream
# ---------------------------------------------------------------------------
BRACKET_PAIR = {"(": ")", ")": "(", "[": "]", "]": "["}
CLOSE_BRACKETS = (")", "]")
OPEN_BRACKETS = ("(", "[")


def corrupt_stream(
    stream: Sequence[str],
    mode: str,
    rng: random.Random,
    arities: Dict[str, int],
    leaf_tokens: Sequence[str],
) -> List[str]:
    """Return a deliberately broken variant of a legal bracket word."""
    symbols = list(stream)
    bracket_positions = [index for index, symbol in enumerate(symbols) if symbol in ("(", ")", "[", "]")]
    if mode == "swap_bracket" and bracket_positions:
        index = rng.choice(bracket_positions)
        symbols[index] = BRACKET_PAIR[symbols[index]]
        return symbols
    if mode == "delete_bracket" and bracket_positions:
        index = rng.choice(bracket_positions)
        del symbols[index]
        return symbols
    if mode == "insert_bracket":
        index = rng.randrange(len(symbols) + 1)
        symbols.insert(index, rng.choice(OPEN_BRACKETS + CLOSE_BRACKETS))
        return symbols
    if mode == "truncate" and len(symbols) > 2:
        cut = rng.randrange(1, len(symbols))
        return symbols[:cut]
    if mode == "unknown_operator":
        positions = [index + 1 for index, symbol in enumerate(symbols) if symbol == "("]
        if positions:
            symbols[rng.choice(positions)] = "NOT_AN_OPERATOR"
            return symbols
    if mode == "unknown_leaf":
        positions = [index + 1 for index, symbol in enumerate(symbols) if symbol == "["]
        if positions:
            symbols[rng.choice(positions)] = "zz_unknown_leaf"
            return symbols
    if mode == "long_leaf":
        positions = [index + 1 for index, symbol in enumerate(symbols) if symbol == "["]
        if positions:
            index = rng.choice(positions)
            end = next((i for i in range(index, len(symbols)) if symbols[i] == "]"), None)
            if end is not None and end - index <= 3:
                symbols.insert(end, symbols[index])
                return symbols
    if mode == "wrong_arity":
        positions = [index + 1 for index, symbol in enumerate(symbols) if symbol == "("]
        candidates = []
        for index in positions:
            name = symbols[index]
            if name not in arities:
                continue
            wanted = arities[name]
            alternatives = [other for other, value in arities.items() if value != wanted]
            if alternatives:
                candidates.append((index, rng.choice(alternatives)))
        if candidates:
            index, replacement = rng.choice(candidates)
            symbols[index] = replacement
            return symbols
    return []


def stream_is_legal(mask: AlgebraMask, stream: Sequence[str]) -> bool:
    try:
        mask.validate_stream(stream)
    except (AlgebraMaskError, ValueError):
        return False
    return True


def reference_is_legal(stream: Sequence[str], arities: Dict[str, int]) -> bool:
    """Independent structural oracle, written without the decoding automaton.

    It uses only the bracket balance of the Dyck-2 projection, the tree reader and
    the operator registry, so it can tell whether a perturbation really produced an
    illegal word without asking the mask that is under test.
    """
    if not logic_map.is_well_bracketed(skeleton_of_stream(stream)):
        return False
    try:
        tree = parse_algebra(" ".join(stream))
    except AlgebraMaskError:
        return False
    stack: List[Dict[str, Any]] = [tree]
    while stack:
        node = stack.pop()
        if "leaf" in node:
            if not 1 <= len(node["leaf"]) <= MAX_LEAF_TOKENS:
                return False
            continue
        if node["op"] not in arities or arities[node["op"]] != len(node["args"]):
            return False
        stack.extend(node["args"])
    return True


def run_negative_controls(
    mask: AlgebraMask,
    rows: List[Dict[str, Any]],
    arities: Dict[str, int],
    leaf_tokens: List[str],
    rng: random.Random,
) -> Dict[str, Any]:
    modes = (
        "swap_bracket",
        "delete_bracket",
        "insert_bracket",
        "truncate",
        "unknown_operator",
        "unknown_leaf",
        "long_leaf",
        "wrong_arity",
    )
    rejected = {mode: 0 for mode in modes}
    attempted = {mode: 0 for mode in modes}
    not_negative: Counter = Counter()
    missed: List[str] = []
    for row in rows:
        stream = row["bracket_ast"].split()
        for mode in modes:
            for _ in range(2):
                broken = corrupt_stream(stream, mode, rng, arities, leaf_tokens)
                # Keep only perturbations the independent oracle calls illegal:
                # asking the mask itself would make this gate vacuous.
                if not broken or reference_is_legal(broken, arities):
                    not_negative[mode] += 1
                    continue
                attempted[mode] += 1
                if stream_is_legal(mask, broken):
                    missed.append(mode + ": " + " ".join(broken))
                else:
                    rejected[mode] += 1
    return {
        "modes": modes,
        "attempted": attempted,
        "rejected": rejected,
        "attempted_total": sum(attempted.values()),
        "rejected_total": sum(rejected.values()),
        "discarded_not_negative": dict(not_negative),
        "missed": missed[:5],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the acceptance gates of the symbolic decoding mask.")
    parser.add_argument(
        "--tokenizer",
        default=str(DATASET_ROOT / "checkpoints" / "tokenizer" / "tokenizer.json"),
        help="learned tokenizer used by the token level gate M5",
    )
    parser.add_argument("--samples", type=int, default=300, help="streams to sample for gate M3")
    parser.add_argument("--seed", type=int, default=17, help="seed of the sampling and corruption runs")
    parser.add_argument("--refine", action="store_true", help="also test the argument-kind refinement")
    parser.add_argument("--max-depth", type=int, default=5, help="tree depth budget of a sampled stream")
    parser.add_argument("--max-symbols", type=int, default=400, help="hard cap of a sampled stream")
    parser.add_argument("--list-only", action="store_true", help="report without writing files")
    parser.add_argument("--lenient", action="store_true", help="return 0 even if a gate fails")
    parser.add_argument("--out-dir", default=str(DATASET_ROOT / "checkpoints" / "masking"))
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    registry_specs = load_registry()
    arities = {name: int(spec["arity"]) for name, spec in registry_specs.items()}
    train_rows = list(item_rows("train"))
    held_rows = list(item_rows("valid")) + list(item_rows("test"))
    print("items read: train " + str(len(train_rows)) + ", held out " + str(len(held_rows)))

    train_leaves = collect_leaves(train_rows)
    held_leaves = collect_leaves(held_rows)
    leaf_tokens = sorted({token for payload in train_leaves + held_leaves for token in payload})

    mask = AlgebraMask.from_leaf_payloads(arities, train_leaves)
    print("mask alphabet: " + json.dumps(mask.describe()))

    report: Dict[str, Any] = {
        "config": {"seed": args.seed, "samples": args.samples, "refine": args.refine},
        "alphabet": mask.describe(),
        "gates": {},
        "scores": {},
    }
    gates: Dict[str, bool] = report["gates"]
    scores: Dict[str, Any] = report["scores"]

    # M1: the mask has to accept every held-out bracket word of the corpus.
    accepted = 0
    rejections: List[str] = []
    for row in held_rows:
        stream = row["bracket_ast"].split()
        if stream_is_legal(mask, stream):
            accepted += 1
        elif len(rejections) < 5:
            rejections.append(row["bracket_ast"])
    scores["held_out_items"] = len(held_rows)
    scores["held_out_accepted"] = accepted
    scores["held_out_rejected_examples"] = rejections
    gates["M1_held_out_accepted"] = accepted == len(held_rows)
    print("M1 held-out bracket words accepted: " + str(accepted) + " of " + str(len(held_rows)))

    # M2: corrupted bracket words have to be refused.
    controls = run_negative_controls(mask, held_rows, arities, leaf_tokens, rng)
    scores["negative_controls"] = {
        "attempted": controls["attempted_total"],
        "rejected": controls["rejected_total"],
        "per_mode": {mode: [controls["attempted"][mode], controls["rejected"][mode]] for mode in controls["modes"]},
        "discarded_not_negative": controls["discarded_not_negative"],
    }
    gates["M2_negative_controls_refused"] = (
        controls["rejected_total"] == controls["attempted_total"] and controls["attempted_total"] > 0
    )
    print(
        "M2 corrupted bracket words refused: " + str(controls["rejected_total"]) + " of "
        + str(controls["attempted_total"])
    )
    if controls["missed"]:
        for entry in controls["missed"]:
            print("  accepted although illegal: " + entry)

    # M3: constrained sampling can only produce well-formed trees.
    sampled_ok = 0
    skeleton_ok = 0
    failures: List[str] = []
    for _ in range(args.samples):
        try:
            stream = mask.sample_stream(rng, args.max_depth, args.max_symbols)
            tree = mask.validate_stream(stream)
        except AlgebraMaskError as error:
            failures.append(str(error))
            continue
        sampled_ok += 1
        skeleton = skeleton_of_stream(stream)
        if logic_map.is_well_bracketed(skeleton) and logic_map.bracket_skeleton(tree) == skeleton:
            skeleton_ok += 1
        elif len(failures) < 5:
            failures.append("skeleton is not Dyck-2: " + skeleton)
    scores["sampled_streams"] = args.samples
    scores["sampled_valid"] = sampled_ok
    scores["sampled_well_bracketed"] = skeleton_ok
    scores["sampling_failures"] = failures[:5]
    gates["M3_sampled_streams_valid"] = (
        sampled_ok == args.samples and skeleton_ok == args.samples and args.samples > 0
    )
    print(
        "M3 sampled streams valid: " + str(sampled_ok) + ", well bracketed skeletons: " + str(skeleton_ok)
        + " of " + str(args.samples)
    )

    # M4: registry, corpus and mask serialization agree with each other.
    registry_mismatches: List[str] = []
    serialization_mismatches: List[str] = []
    observed_arities: Counter = Counter()
    for row in train_rows + held_rows:
        stream = row["bracket_ast"].split()
        tree = parse_algebra(" ".join(stream))
        # The symbol stream the mask reasons about must be exactly what the
        # dataset stores and exactly what the dataset serializer produces.
        if algebra_word(tree) != row["bracket_ast"]:
            serialization_mismatches.append("reader and writer: " + row["bracket_ast"])
        if algebra_word(row["ast_json"]) != logic_map.bracket_ast(row["ast_json"]):
            serialization_mismatches.append("not equal to the dataset serializer: " + row["bracket_ast"])
        if algebra_word(row["ast_json"]) != row["bracket_ast"]:
            serialization_mismatches.append("stored tree: " + row["bracket_ast"])
        if not logic_map.is_well_bracketed(skeleton_of_stream(stream)):
            serialization_mismatches.append("skeleton: " + row["bracket_ast"])
        try:
            logic_map.check_tree(tree, registry_specs)
        except logic_map.LogicTreeError as error:
            serialization_mismatches.append(str(error) + " | " + row["bracket_ast"])
        stack: List[Dict[str, Any]] = [tree]
        while stack:
            node = stack.pop()
            if "leaf" in node:
                continue
            observed_arities[(node["op"], len(node["args"]))] += 1
            if node["op"] not in arities or arities[node["op"]] != len(node["args"]):
                registry_mismatches.append(node["op"] + " with " + str(len(node["args"])) + " children")
            stack.extend(node["args"])
    gates["M4_registry_agrees_with_corpus"] = not registry_mismatches and not serialization_mismatches
    scores["distinct_operator_arities"] = len(observed_arities)
    scores["registry_mismatches"] = registry_mismatches[:5]
    scores["serialization_mismatches"] = serialization_mismatches[:5]
    print(
        "M4 operator/arity pairs observed: " + str(len(observed_arities)) + ", mismatches: "
        + str(len(registry_mismatches) + len(serialization_mismatches))
    )

    # M5: token level mask over the learned vocabulary.
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print("M5 skipped: no learned tokenizer at " + str(tokenizer_path))
        print("run python datasets/multilingual/tools/train_tokenizer.py first")
        gates["M5_token_level_protocol"] = False
        scores["tokenizer_file"] = str(tokenizer_path)
    else:
        tokenizer = Tokenizer.load(tokenizer_path)
        try:
            token_mask = TokenStreamMask(mask, tokenizer)
        except TokenizerError as error:
            print("M5 failed to build the token level mask: " + str(error))
            gates["M5_token_level_protocol"] = False
            token_mask = None
        if token_mask is not None:
            checked = 0
            allowed_by_gold = 0
            forced_completion = 0
            premature_end = 0
            off_protocol: List[str] = []
            for row in held_rows:
                stream = row["bracket_ast"].split()
                ids = token_mask.encode_target(stream)
                end_position = len(ids) - 2
                expected = ids + [token_mask.pad_id]
                for index, want in enumerate(expected):
                    allowed = token_mask.allowed_next_ids(ids[:index])
                    checked += 1
                    if want in allowed:
                        allowed_by_gold += 1
                    elif len(off_protocol) < 5:
                        off_protocol.append(
                            row["bracket_ast"] + " | position " + str(index) + " of " + str(len(ids))
                        )
                if token_mask.allowed_next_ids(ids[:end_position]) == {token_mask.rule_end_id}:
                    forced_completion += 1
                for index in range(1, end_position):
                    if token_mask.rule_end_id in token_mask.allowed_next_ids(ids[:index]):
                        premature_end += 1
            scores["token_steps_checked"] = checked
            scores["token_steps_allowed"] = allowed_by_gold
            scores["completion_forces_rule_end"] = forced_completion
            scores["premature_rule_end_offers"] = premature_end
            scores["off_protocol_examples"] = off_protocol
            gates["M5_token_level_protocol"] = (
                allowed_by_gold == checked
                and forced_completion == len(held_rows)
                and premature_end == 0
                and not off_protocol
            )
            print(
                "M5 token decisions accepted: " + str(allowed_by_gold) + " of " + str(checked)
            )
            print(
                "M5 the closing marker is forced at completion in " + str(forced_completion) + " of "
                + str(len(held_rows)) + " items, and offered too early " + str(premature_end) + " times"
            )
            report["token_alphabet"] = token_mask.report()

    # Optional refinement: which kind fills which operator argument position.
    if args.refine:
        counts: Counter = Counter()
        for row in train_rows:
            AlgebraMask.observe_child_kinds(row["ast_json"], counts)
        narrowed = AlgebraMask(
            arities,
            mask.leaf_forms,
            mask.leaf_value_tokens,
            AlgebraMask.child_kinds_from_counts(counts),
        )
        narrowed_accepted = sum(
            1 for row in held_rows if stream_is_legal(narrowed, row["bracket_ast"].split())
        )
        scores["refined_held_out_accepted"] = narrowed_accepted
        scores["refined_held_out_items"] = len(held_rows)
        scores["refined_positions"] = len(narrowed.child_kinds)
        print(
            "refinement argument kinds learned: " + str(len(narrowed.child_kinds)) + ", held-out accepted "
            + str(narrowed_accepted) + " of " + str(len(held_rows))
        )
        report["refinement_usable"] = narrowed_accepted == len(held_rows)

    all_ok = all(gates.values())
    print("gates: " + ", ".join(name + "=" + ("PASS" if value else "FAIL") for name, value in gates.items()))
    print("FINAL VERDICT: " + ("PASS" if all_ok else "FAIL"))

    out_dir = Path(args.out_dir)
    if not args.list_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "masking_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("artifacts written to " + str(out_dir))

    if not all_ok and not args.lenient:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
