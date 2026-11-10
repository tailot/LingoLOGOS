"""Train the project tokenizer from scratch on the project corpus.

No pretrained vocabulary is downloaded: the merges are learned from the corpus
files of this dataset and nothing else. The structural alphabet of the bracket
algebra is kept atomic on purpose, so the constrained decoder of
``src/neuro_symbolic/masking.py`` can talk about ``( ) [ ]``, operator names and
leaf identifiers one token at a time.

The corpus is closed and small (1.9k distinct word forms, every one of them
occurring at least twice), so the merge loop stops before it reaches the
vocabulary target. The report prints where it saturated instead of inflating the
vocabulary with tokens that never occur.

Usage:
    python datasets/multilingual/tools/train_tokenizer.py
    python datasets/multilingual/tools/train_tokenizer.py --vocab-size 6000 --min-frequency 2
    python datasets/multilingual/tools/train_tokenizer.py --list-only
    python datasets/multilingual/tools/train_tokenizer.py --all-splits

Merges are learned from the train split only, so the scores reported on the valid
and test splits measure generalization of the vocabulary, not memorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_paths import bootstrap  # noqa: E402

REPO_ROOT, DATASET_ROOT = bootstrap(HERE)

from neuro_symbolic.tokenizer import (  # noqa: E402
    ALGEBRA_BRACKETS,
    Tokenizer,
    TokenizerError,
    pre_tokenize,
)

SPLITS = ("train", "valid", "test")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sft_texts(split: str) -> List[str]:
    """Every message content of one fine-tuning split, exactly as the model reads it."""
    path = DATASET_ROOT / "sft" / (split + ".jsonl")
    texts: List[str] = []
    for row in read_jsonl(path):
        for message in row.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
    return texts


def item_rows(scope: str) -> List[Dict[str, Any]]:
    """Dataset items of one split, across the five languages."""
    rows: List[Dict[str, Any]] = []
    for language in sorted((DATASET_ROOT / "data").iterdir()):
        if not language.is_dir():
            continue
        for path in sorted(language.glob(scope + "-*.jsonl")):
            rows.extend(read_jsonl(path))
    return rows


def leaf_payloads(rows: List[Dict[str, Any]]) -> List[List[str]]:
    """Square-node payloads of every item, the alphabet the leaf grammar must cover."""
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


def algebra_symbols(rows: List[Dict[str, Any]]) -> List[str]:
    """Distinct algebra symbols of the items: brackets, operators, leaf tokens."""
    seen: Set[str] = set()

    def walk(node: Dict[str, Any]) -> None:
        if "leaf" in node:
            seen.add("[")
            seen.add("]")
            seen.update(str(token) for token in node["leaf"])
            return
        seen.add("(")
        seen.add(")")
        seen.add(node["op"])
        for child in node.get("args", []):
            walk(child)

    for row in rows:
        walk(row["ast_json"])
    return sorted(seen)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Learn the BPE tokenizer from the dataset corpus.")
    parser.add_argument("--out-dir", default=str(DATASET_ROOT / "checkpoints" / "tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=6000, help="vocabulary target")
    parser.add_argument("--min-frequency", type=int, default=2, help="stop merging below this count")
    parser.add_argument("--list-only", action="store_true", help="report without writing files")
    parser.add_argument("--lenient", action="store_true", help="return 0 even if a gate fails")
    parser.add_argument("--all-splits", action="store_true", help="learn also from valid and test")
    parser.add_argument("--limit", type=int, default=0, help="read only the first N rows of a split")
    args = parser.parse_args(argv)

    train_rows = list(item_rows("train"))
    held_rows: List[Dict[str, Any]] = list(item_rows("valid")) + list(item_rows("test"))
    if args.limit:
        train_rows = train_rows[: args.limit]
        held_rows = held_rows[: args.limit]

    learn_splits = list(SPLITS) if args.all_splits else ["train"]
    texts: List[str] = []
    for split in learn_splits:
        texts.extend(sft_texts(split))

    registry = json.loads((DATASET_ROOT / "templates" / "operators.json").read_text(encoding="utf-8"))
    operators = sorted(registry["operators"])

    # The algebra alphabet learned from the train split only, so the held-out
    # score below says whether the symbolic alphabet generalizes at all.
    atomic_symbols: Set[str] = set(ALGEBRA_BRACKETS)
    atomic_symbols.update(operators)
    atomic_symbols.update(registry.get("literal_leaves", []))
    for payload in leaf_payloads(train_rows):
        atomic_symbols.update(payload)

    print("corpus texts read: " + str(len(texts)) + " from splits " + ", ".join(learn_splits))
    print("atomic algebra symbols declared: " + str(len(atomic_symbols)))

    try:
        tokenizer = Tokenizer.train(
            texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            atomic_symbols=atomic_symbols,
        )
    except TokenizerError as error:
        print("ERROR: tokenizer training failed: " + str(error))
        return 2

    stats = dict(tokenizer.statistics)
    print("vocabulary: " + str(tokenizer.vocabulary_size) + " tokens, target " + str(args.vocab_size))
    print("merges learned: " + str(stats["merges_learned"]) + " (" + str(stats["stop_reason"]) + ")")
    print("vocabulary saturated at: " + str(stats["saturated_at"]))

    report: Dict[str, Any] = {
        "config": {
            "splits": learn_splits,
            "vocab_size": args.vocab_size,
            "min_frequency": args.min_frequency,
            "limit": args.limit,
        },
        "statistics": stats,
        "gates": {},
        "scores": {},
    }

    gates: Dict[str, bool] = report["gates"]
    scores: Dict[str, Any] = report["scores"]

    # T1 and T2: reversibility and unknown tokens on every split.
    failures_total = 0
    unknown_total = 0
    tokens_total = 0
    for split in SPLITS:
        split_texts = sft_texts(split)
        failures, samples = tokenizer.round_trip_failures(split_texts)
        unknown, total = tokenizer.unknown_tokens(split_texts)
        failures_total += failures
        unknown_total += unknown
        tokens_total += total
        print(
            "split " + split + ": " + str(len(split_texts)) + " texts, round trip failures "
            + str(failures) + ", unknown tokens " + str(unknown) + " of " + str(total)
        )
        if samples:
            for sample in samples:
                print("  not reversible: " + sample)
    scores["round_trip_failures"] = failures_total
    scores["unknown_tokens"] = unknown_total
    scores["tokens_produced"] = tokens_total
    gates["T1_reversible"] = failures_total == 0
    gates["T2_no_unknown"] = unknown_total == 0

    # T3 and T4: the symbolic alphabet stays atomic and the train alphabet covers
    # the whole corpus, including the held-out splits.
    all_rows = train_rows + held_rows
    symbols = algebra_symbols(all_rows)
    encodable, not_encodable = tokenizer.encodable_symbols(symbols)
    scores["algebra_symbols"] = len(symbols)
    scores["algebra_symbols_atomic"] = len(encodable)
    scores["algebra_symbols_not_atomic"] = not_encodable[:20]
    gates["T3_alphabet_atomic"] = not not_encodable

    train_symbols = set(algebra_symbols(train_rows))
    held_symbols = [symbol for symbol in symbols if symbol not in train_symbols]
    scores["held_out_algebra_symbols"] = held_symbols[:20]
    gates["T4_alphabet_generalizes"] = not held_symbols
    if held_symbols:
        print("algebra symbols that only the held-out splits use: " + ", ".join(held_symbols[:20]))

    # T5: the target space of the model is the algebra region only, so the whole
    # bracket word of an item must survive a token round trip.
    broken_words = 0
    example = ""
    for row in all_rows:
        word = row["bracket_ast"]
        stream = word.split()
        try:
            ids = tokenizer.encode_algebra(stream)
        except TokenizerError:
            broken_words += 1
            example = example or word
            continue
        if tokenizer.decode_symbol_stream(ids) != stream:
            broken_words += 1
            example = example or word
    scores["bracket_words_checked"] = len(all_rows)
    scores["bracket_words_broken"] = broken_words
    if example:
        print("bracket word that does not survive tokenization: " + example)
    gates["T5_algebra_token_round_trip"] = broken_words == 0

    # Size report: the spec range 8k-32k is not reachable on this corpus, and
    # padding the vocabulary with unused tokens would only hide that.
    report["size_note"] = {
        "vocab_target": args.vocab_size,
        "vocabulary_size": tokenizer.vocabulary_size,
        "saturated_at": stats["saturated_at"],
        "target_above_saturation": args.vocab_size > int(stats["saturated_at"]),
    }
    if report["size_note"]["target_above_saturation"]:
        print(
            "note: the vocabulary saturated at " + str(stats["saturated_at"])
            + " tokens, so the requested " + str(args.vocab_size)
            + " cannot be reached without unused entries"
        )

    all_ok = all(gates.values())
    print("gates: " + ", ".join(name + "=" + ("PASS" if value else "FAIL") for name, value in gates.items()))
    print("FINAL VERDICT: " + ("PASS" if all_ok else "FAIL"))

    out_dir = Path(args.out_dir)
    if not args.list_only:
        tokenizer.save(out_dir / "tokenizer.json")
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("artifacts written to " + str(out_dir))

    if not all_ok and not args.lenient:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
