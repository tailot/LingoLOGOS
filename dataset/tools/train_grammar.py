"""Grammar induction on the bracket algebra of this dataset.

The dataset stores every instruction as a balanced word over the four Dyck-2
terminals, so the generator corpus of the LingoLOGOS engine can be trained on
the structures this dataset really means. The script reuses the engine in the
parent repository untouched (it only adds it to sys.path) and writes its own
artifacts under dataset/checkpoints, never into the project artifacts folder.

By default the corpus mixes the dataset words with synthetic Dyck-2 words: the
dataset supplies 29 structural words, and synthetic sampling gives the PCFG
enough statistical variety to be identifiable. The coverage gate G2 is always
measured on the dataset words alone.

Usage:
    python dataset/tools/train_grammar.py
    python dataset/tools/train_grammar.py --epochs 120 --synthetic 900
    python dataset/tools/train_grammar.py --init random --num-nt 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
REPO_ROOT = DATASET_ROOT.parent
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch  # noqa: E402

from neuro_symbolic.grammar import PCFGrammar  # noqa: E402
from neuro_symbolic.parser import DifferentiableCYKParser  # noqa: E402
from neuro_symbolic.reference import (  # noqa: E402
    GOLD_NON_TERMINALS,
    apply_gold_prior,
    generate_negative_controls,
    gold_rule_f1,
)
from neuro_symbolic.trainer import DyckDataset, GrammarTrainer  # noqa: E402
from neuro_symbolic.verification import GrammarVerifier  # noqa: E402

VOCAB = ["(", ")", "[", "]"]


def read_words(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").split()


def encode(words: List[str]) -> List[List[int]]:
    return [[DyckDataset.VOCAB_MAP[char] for char in word] for word in words]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Induce the PCFG behind the dataset bracket algebra.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="dataset directory")
    parser.add_argument("--out-dir", default=str(DATASET_ROOT / "checkpoints" / "grammar"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--init", choices=("prior", "random"), default="prior")
    parser.add_argument("--num-nt", type=int, default=5)
    parser.add_argument("--synthetic", type=int, default=900, help="extra random Dyck-2 words (0 disables)")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--entropy-penalty", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--prune", type=float, default=0.01)
    parser.add_argument("--search-bound", type=int, default=6)
    parser.add_argument("--lenient", action="store_true", help="return 0 even if a gate fails")
    parser.add_argument("--list-only", action="store_true", help="report without writing artifacts")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    corpus_dir = root / "corpus"

    dataset_words = read_words(corpus_dir / "dyck2_all.txt")
    train_words = read_words(corpus_dir / "dyck2_train.txt")
    holdout_words = sorted(set(read_words(corpus_dir / "dyck2_valid.txt")) | set(read_words(corpus_dir / "dyck2_test.txt")))
    negative_words = read_words(corpus_dir / "dyck2_negative_controls.txt")
    print("dataset bracket words: " + str(len(dataset_words)) + " unique")
    print("train words " + str(len(train_words)) + " / holdout words " + str(len(holdout_words)))

    mixed = list(dict.fromkeys(train_words))
    if args.synthetic > 0:
        filler = DyckDataset.generate_dyck2(
            num_samples=args.synthetic,
            max_depth=args.max_depth,
            max_length=args.max_length,
            seed=args.seed,
        )
        mixed = list(dict.fromkeys(mixed + filler.as_strings()))
    print("training words: " + str(len(mixed)))

    num_nt = args.num_nt
    if args.init == "prior" and num_nt == len(GOLD_NON_TERMINALS):
        nt_labels = list(GOLD_NON_TERMINALS)
    else:
        nt_labels = ["S"] + ["NT_" + str(index) for index in range(1, num_nt)]

    net = DifferentiableCYKParser(num_non_terminals=num_nt, vocab_size=len(VOCAB), temperature=args.tau)
    with torch.no_grad():
        if args.init == "prior" and num_nt == len(GOLD_NON_TERMINALS):
            print("initialization: structured prior on the gold Dyck-2 CNF")
            apply_gold_prior(net, nt_labels)
        else:
            print("initialization: random weights")
            net.rule_logits.data.normal_(0.0, 0.5)
            net.term_logits.data.normal_(0.0, 0.5)
            net.family_logits.data.zero_()

    train_set = DyckDataset(encode(mixed))
    holdout_set = DyckDataset(encode(holdout_words))
    print("corpus: " + str(len(train_set)) + " training words, " + str(len(holdout_set)) + " holdout words")

    trainer = GrammarTrainer(
        parser=net,
        learning_rate=args.lr,
        temperature_tau=args.tau_start,
        entropy_penalty=args.entropy_penalty,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        tau_now = args.tau_start + (args.tau - args.tau_start) * progress
        trainer.temperature_tau = tau_now
        net.temperature = tau_now
        stats = trainer.train_epoch(train_set, batch_size=args.batch_size)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            holdout = trainer.evaluate(holdout_set)
            print(
                "epoch " + str(epoch + 1) + " | tau " + format(tau_now, ".3f")
                + " | loss " + format(stats["avg_loss"], ".4f")
                + " | holdout nll/symbol " + format(holdout["mean_nll_per_symbol"], ".4f")
                + " | ppl/symbol " + format(holdout["ppl_per_symbol"], ".4f")
            )

    metrics_holdout = trainer.evaluate(holdout_set)

    grammar = PCFGrammar.from_model(
        net,
        nt_labels=nt_labels,
        vocab_labels=VOCAB,
        start_symbol="S",
        prune_threshold=args.prune,
        temperature=args.tau,
        renormalize=True,
    )
    print("extracted grammar: " + str(len(grammar.rules)) + " rules")

    verifier = GrammarVerifier(grammar)
    lark_ok, lark_error = verifier.verify_lark_compilation()
    print("G1a lark compilation: " + ("PASS" if lark_ok else "FAIL (" + str(lark_error) + ")"))
    pcfg_ok, masses = verifier.verify_pcfg_distribution()
    print("G1b mass per non-terminal: " + ("PASS" if pcfg_ok else "FAIL") + " -> " + str(masses))
    coverage, failed = verifier.verify_corpus_coverage(dataset_words)
    print("G2 coverage of the dataset bracket words: " + format(coverage * 100.0, ".2f") + "%")
    if failed:
        print("   failed words: " + str(failed[:5]))
    negatives = negative_words or generate_negative_controls(dataset_words, max_controls=200, seed=args.seed)
    rejection, accepted_wrong = verifier.verify_negative_controls(negatives)
    print("G3 rejection of " + str(len(negatives)) + " ill-bracketed controls: " + format(rejection * 100.0, ".2f") + "%")
    precision, recall, f1, f1_details = gold_rule_f1(grammar, probability_floor=args.prune)
    print("G3b gold rule F1: " + format(f1, ".4f"))
    report = verifier.verify_language_soundness(search_bound=args.search_bound)
    print("G4 inclusion L(G) subset Dyck-2: " + str(report))

    summary: Dict[str, Any] = {
        "config": {
            "seed": args.seed,
            "init": args.init,
            "num_nt": num_nt,
            "nt_labels": nt_labels,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "tau": args.tau,
            "tau_start": args.tau_start,
            "entropy_penalty": args.entropy_penalty,
            "weight_decay": args.weight_decay,
            "prune_threshold": args.prune,
            "search_bound": args.search_bound,
            "synthetic_words": args.synthetic,
            "dataset_words": len(dataset_words),
            "train_words": len(train_set),
            "holdout_words": len(holdout_set),
        },
        "metrics_holdout": metrics_holdout,
        "grammar": {
            "num_rules": len(grammar.rules),
            "non_terminals": grammar.non_terminals,
            "valid_pcfg": pcfg_ok,
            "lhs_masses": masses,
            "lark_compiles": lark_ok,
        },
        "language": {
            "dataset_word_coverage": coverage,
            "coverage_failures": failed,
            "negative_controls": len(negatives),
            "negative_rejection_rate": rejection,
            "gold_rule_precision": precision,
            "gold_rule_recall": recall,
            "gold_rule_f1": f1,
        },
        "soundness": {
            "status": report.status,
            "sound": report.sound,
            "counterexample": report.counterexample,
            "notes": report.notes,
        },
        "gates": {
            "G1_grammar_valid": bool(lark_ok and pcfg_ok),
            "G2_dataset_words_covered": bool(coverage >= 1.0),
            "G3_negative_controls_rejected": bool(rejection >= 0.999),
            "G4_z3_unsat_certificate": bool(report.sound and report.status == "UNSAT"),
        },
    }

    if args.list_only:
        print(json.dumps(summary, indent=2))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "rule_logits": net.rule_logits.detach().cpu(),
                "term_logits": net.term_logits.detach().cpu(),
                "family_logits": net.family_logits.detach().cpu(),
                "metadata": summary["config"],
            },
            out_dir / "grammar_weights.pt",
        )
        grammar.to_json(str(out_dir / "grammar.json"))
        grammar.to_yaml(str(out_dir / "grammar.yaml"))
        grammar.to_gf("DatasetGrammar", str(out_dir / "DatasetGrammar.gf"))
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("artifacts written to " + str(out_dir))

    all_ok = all(summary["gates"].values())
    print("FINAL VERDICT: " + ("PASS" if all_ok else "FAIL"))
    if not all_ok and not args.lenient:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
