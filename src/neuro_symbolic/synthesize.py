"""
Grammar induction script, extraction of the discrete grammar and generation of the
artifacts. Compliant with sections 4, 5, 6, 8 and 9 of the technical specification.

Usage:
    python src/neuro_symbolic/synthesize.py                     # reproducible run (seed 42)
    python src/neuro_symbolic/synthesize.py --init random       # no prior on the rules
    python src/neuro_symbolic/synthesize.py --epochs 40 --tau 0.5
    python src/neuro_symbolic/synthesize.py --list-only         # report only, no file written

Acceptance gates (in strict agreement with the documentation):
    G1 exported grammar loads and is a valid PCFG (mass 1 per non-terminal)
    G2 corpus coverage 100% (discrete CYK)
    G3 negative control: rejection of the strings outside Dyck-k
    G4 SMT check of the inclusion L(G) subset Dyck-2 -> UNSAT verdict
With `--lenient` the script does not exit with code != 0 when a gate fails.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# Ensures the repository root and the src folder are on sys.path
# (src/ layout: the package is importable both from the repo root and from src/)
repo_root_path = Path(__file__).resolve().parents[2]
for path_entry in (str(repo_root_path), str(repo_root_path / "src")):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neuro-symbolic grammar induction on Dyck-2")
    parser.add_argument("--seed", type=int, default=42, help="seed for initialization and dataset")
    parser.add_argument(
        "--init",
        choices=["prior", "random"],
        default="prior",
        help="'prior' initializes the gold Dyck-2 CNF, 'random' starts from random weights",
    )
    parser.add_argument("--num-nt", type=int, default=5, help="number of non-terminals K")
    parser.add_argument("--num-samples", type=int, default=100, help="unique Dyck-2 sentences to generate")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="final temperature (also used for the grammar extraction)",
    )
    parser.add_argument(
        "--tau-start",
        type=float,
        default=1.0,
        help="initial temperature: linear annealing tau-start -> tau during training",
    )
    parser.add_argument("--entropy-penalty", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--prune", type=float, default=0.01, help="tau threshold for rule pruning")
    parser.add_argument("--search-bound", type=int, default=6, help="counterexample search bound")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="output folder (relative to the repo root, not to the CWD)",
    )
    parser.add_argument("--lenient", action="store_true", help="do not exit with code != 0 when a gate fails")
    parser.add_argument("--list-only", action="store_true", help="do not write artifacts")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = repo_root_path / artifacts_dir
    if not args.list_only:
        os.makedirs(artifacts_dir, exist_ok=True)

    num_nt = args.num_nt
    vocab_size = len(VOCAB)

    # Non-terminal names are only labels: they carry no semantic meaning.
    # With the 'prior' initialization we use the gold names, with 'random' neutral names.
    if args.init == "prior" and num_nt == len(GOLD_NON_TERMINALS):
        nt_labels = list(GOLD_NON_TERMINALS)
    else:
        nt_labels = ["S"] + [f"NT_{i}" for i in range(1, num_nt)]

    print(f"Initializing DifferentiableCYKParser (K={num_nt}, |V|={vocab_size})...")
    parser = DifferentiableCYKParser(
        num_non_terminals=num_nt,
        vocab_size=vocab_size,
        temperature=args.tau,
    )

    with torch.no_grad():
        if args.init == "prior" and num_nt == len(GOLD_NON_TERMINALS):
            print("Initialization: structured prior on the gold Dyck-2 CNF")
            apply_gold_prior(parser, nt_labels)
        else:
            print("Initialization: random weights (N(0, 0.5))")
            parser.rule_logits.data.normal_(0.0, 0.5)
            parser.term_logits.data.normal_(0.0, 0.5)
            parser.family_logits.data.zero_()

    print("Generating the synthetic Dyck-2 dataset...")
    dataset = DyckDataset.generate_dyck2(
        num_samples=args.num_samples,
        max_depth=args.max_depth,
        max_length=args.max_length,
        seed=seed,
    )
    train_set, test_set = dataset.split(test_fraction=args.test_fraction, seed=seed)
    print(
        f"Dataset: {len(dataset)} unique sentences -> train {len(train_set)} / test {len(test_set)}"
    )

    trainer = GrammarTrainer(
        parser=parser,
        learning_rate=args.lr,
        temperature_tau=args.tau_start,
        entropy_penalty=args.entropy_penalty,
        weight_decay=args.weight_decay,
    )

    print("Training in progress (temperature annealing)...")
    for epoch in range(args.epochs):
        # Linear annealing of tau: the soft temperature at the beginning keeps the gradient
        # on competing productions alive, then it tightens towards a 0/1 distribution
        # consistent with the discrete reading (pruning) that follows.
        progress = epoch / max(args.epochs - 1, 1)
        tau_now = args.tau_start + (args.tau - args.tau_start) * progress
        trainer.temperature_tau = tau_now
        parser.temperature = tau_now

        stats = trainer.train_epoch(train_set, batch_size=args.batch_size)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            holdout = trainer.evaluate(test_set)
            print(
                f"Epoch {epoch+1:02d} | tau {tau_now:.3f} | loss {stats['avg_loss']:.4f} | "
                f"entropy {stats['avg_entropy']:.4f} | grad {stats['grad_norm']:.2f} | "
                f"TEST nll/simb {holdout['mean_nll_per_symbol']:.4f} | "
                f"PPL/simb {holdout['ppl_per_symbol']:.4f} | "
                f"sparsity {holdout['rule_sparsity']:.4f} | "
                f"mass error {holdout['max_lhs_mass_error']:.2e}"
            )

    metrics_train = trainer.evaluate(train_set)
    metrics_test = trainer.evaluate(test_set)
    print("Model metrics train:", metrics_train)
    print("Model metrics test :", metrics_test)

    # 1. Extraction of the discrete grammar (pruning + renormalization per LHS)
    grammar = PCFGrammar.from_model(
        parser,
        nt_labels=nt_labels,
        vocab_labels=VOCAB,
        start_symbol="S",
        prune_threshold=args.prune,
        temperature=args.tau,
        renormalize=True,
    )
    print(
        f"Extracted grammar: {len(grammar.rules)} rules "
        f"(non-terminals used: {len(grammar.non_terminals)})"
    )

    # 2. Formal verification and quality metrics of the induced language
    verifier = GrammarVerifier(grammar)

    lark_ok, lark_err = verifier.verify_lark_compilation()
    print(f"G1a Lark compilation: {'PASS' if lark_ok else f'FAIL ({lark_err})'}")

    pcfg_ok, masses = verifier.verify_pcfg_distribution()
    print(f"G1b valid PCFG (mass per LHS): {'PASS' if pcfg_ok else 'FAIL'} -> {masses}")

    corpus_strings = dataset.as_strings()
    coverage, failed = verifier.verify_corpus_coverage(corpus_strings)
    print(
        f"G2 corpus coverage: {coverage * 100:.2f}% (target 100%)"
        + ("" if not failed else f" | failed sentences: {failed[:5]}")
    )

    negatives = generate_negative_controls(corpus_strings, max_controls=200, seed=seed)
    rejection, accepted_wrong = verifier.verify_negative_controls(negatives)
    print(
        f"G3 negative controls ({len(negatives)} strings outside Dyck-k): "
        f"rejection {rejection * 100:.2f}%"
        + ("" if not accepted_wrong else f" | wrongly accepted: {accepted_wrong[:5]}")
    )

    precision, recall, f1, f1_details = gold_rule_f1(grammar, probability_floor=args.prune)
    print(f"G3b gold rule F1: {f1:.4f} (P {precision:.4f} / R {recall:.4f})")

    report = verifier.verify_language_soundness(search_bound=args.search_bound)
    print(f"G4 SMT check of the inclusion L(G) subset Dyck-2: {report}")
    if report.certificate:
        print(f"   certificate (delta_round, delta_square, min_round, min_square): {report.certificate}")
    if report.derivation:
        print(f"   counterexample tree: {json.dumps(report.derivation)[:400]}")

    summary: Dict[str, object] = {
        "config": {
            "seed": seed,
            "init": args.init,
            "num_nt": num_nt,
            "vocab": VOCAB,
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
            "num_samples": len(dataset),
            "train_size": len(train_set),
            "test_size": len(test_set),
            "max_depth": args.max_depth,
            "max_length": args.max_length,
        },
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "grammar": {
            "num_rules": len(grammar.rules),
            "num_binary_rules": sum(1 for r in grammar.rules if len(r.rhs) == 2),
            "num_terminal_rules": sum(1 for r in grammar.rules if len(r.rhs) == 1),
            "non_terminals": grammar.non_terminals,
            "valid_pcfg": pcfg_ok,
            "lhs_masses": masses,
            "lark_compiles": lark_ok,
        },
        "language": {
            "corpus_coverage": coverage,
            "corpus_failures": failed,
            "negative_controls": len(negatives),
            "negative_rejection_rate": rejection,
            "negatives_wrongly_accepted": accepted_wrong,
            "gold_rule_precision": precision,
            "gold_rule_recall": recall,
            "gold_rule_f1": f1,
            "gold_rule_details": f1_details,
        },
        "soundness": {
            "status": report.status,
            "sound": report.sound,
            "certificate": {k: list(v) for k, v in report.certificate.items()},
            "counterexample": report.counterexample,
            "derivation": report.derivation,
            "search_bound": report.search_bound,
            "notes": report.notes,
        },
        "gates": {
            "G1_grammar_valid": bool(lark_ok and pcfg_ok),
            "G2_corpus_coverage_100": bool(coverage >= 1.0),
            "G3_negative_controls_rejected": bool(rejection >= 0.999),
            "G4_z3_unsat_certificate": bool(report.sound and report.status == "UNSAT"),
        },
    }

    if args.list_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("Mode --list-only: no artifact written.")
    else:
        # 3. Export of the artifacts
        weights_path = artifacts_dir / "grammar_weights.pt"
        torch.save(
            {
                "model_state_dict": parser.state_dict(),
                "rule_logits": parser.rule_logits.detach().cpu(),
                "term_logits": parser.term_logits.detach().cpu(),
                "family_logits": parser.family_logits.detach().cpu(),
                "metadata": summary["config"],
            },
            weights_path,
        )
        print(f"Weights saved in {weights_path}")

        json_path = artifacts_dir / "grammar.json"
        yaml_path = artifacts_dir / "grammar.yaml"
        gf_path = artifacts_dir / "DynamicGrammarIta.gf"
        metrics_path = artifacts_dir / "metrics.json"

        grammar.to_json(str(json_path))
        grammar.to_yaml(str(yaml_path))
        grammar.to_gf("DynamicGrammarIta", str(gf_path))
        metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"Grammar exported to {json_path}, {yaml_path}, {gf_path}")
        print(f"Metrics + verification report in {metrics_path}")

    gates = summary["gates"]
    all_ok = all(gates.values())
    print("Final gates:", json.dumps(gates, indent=2))
    print("FINAL VERDICT:", "PASS" if all_ok else "FAIL")
    if not all_ok and not args.lenient:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
