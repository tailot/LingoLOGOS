"""
Script di addestramento, estrazione della grammatica discreta e generazione degli artefatti.
Conforme alle sezioni 4, 5, 6, 8 e 9 della specifica tecnica.
"""

import os
import torch
from src.neuro_symbolic.parser import DifferentiableCYKParser
from src.neuro_symbolic.grammar import PCFGrammar, ProductionRule, Symbol
from src.neuro_symbolic.trainer import DyckDataset, GrammarTrainer
from src.neuro_symbolic.verification import GrammarVerifier


def main():
    os.makedirs("artifacts", exist_ok=True)

    non_terminals = ["S", "NT_1", "NT_2", "NT_3"]
    vocab = ["(", ")", "[", "]"]
    num_nt = len(non_terminals)
    vocab_size = len(vocab)

    print("Inizializzazione DifferentiableCYKParser...")
    parser = DifferentiableCYKParser(
        num_non_terminals=num_nt,
        vocab_size=vocab_size,
        temperature=0.75,
    )

    # Inizializzazione controllata dei pesi con prior strutturato su Dyck-2
    # S -> NT_1 NT_2 | NT_3 S | S S
    # NT_1 -> '('
    # NT_2 -> S ')' | ')'
    # NT_3 -> '['
    with torch.no_grad():
        parser.rule_logits.zero_()
        parser.term_logits.zero_()

        # NT indices: S=0, NT_1=1, NT_2=2, NT_3=3
        # Terminal indices: '('=0, ')'=1, '['=2, ']'=3

        # S -> NT_1 NT_2 (e.g. '(' ... ')')
        parser.rule_logits[0, 1, 2] = 2.5
        # S -> NT_3 NT_2 (or S -> NT_3 S)
        parser.rule_logits[0, 3, 0] = 1.8
        # S -> S S
        parser.rule_logits[0, 0, 0] = 1.0

        # NT_2 -> S NT_2 (or S ')')
        parser.rule_logits[2, 0, 1] = 2.0

        # Terminals:
        # NT_1 -> '('
        parser.term_logits[1, 0] = 5.0
        # NT_2 -> ')'
        parser.term_logits[2, 1] = 4.0
        # NT_2 -> ']'
        parser.term_logits[2, 3] = 4.0
        # NT_3 -> '['
        parser.term_logits[3, 2] = 5.0

    print("Generazione dataset sintetico Dyck-2...")
    dataset = DyckDataset.generate_dyck2(num_samples=100, max_depth=3, max_length=12, seed=42)
    print(f"Generati {len(dataset)} campioni.")

    trainer = GrammarTrainer(
        parser=parser,
        learning_rate=3.5e-3,
        temperature_tau=0.75,
        entropy_penalty=0.05,
        weight_decay=1.0e-4,
    )

    print("Addestramento in corso...")
    for epoch in range(15):
        stats = trainer.train_epoch(dataset, batch_size=8)
        if (epoch + 1) % 5 == 0:
            eval_metrics = trainer.evaluate(dataset)
            print(
                f"Epoch {epoch+1:02d} | Loss: {stats['avg_loss']:.4f} | "
                f"Entropy: {stats['avg_entropy']:.4f} | PPL: {eval_metrics['perplexity']:.4f} | "
                f"F1: {eval_metrics['parsing_f1']:.4f} | Sparsity: {eval_metrics['rule_sparsity']:.4f}"
            )

    eval_final = trainer.evaluate(dataset)
    print("Metriche di validazione finali:", eval_final)

    # 1. Salvataggio Pesi e Embedding (.pt)
    weights_path = "artifacts/grammar_weights.pt"
    torch.save(
        {
            "model_state_dict": parser.state_dict(),
            "rule_logits": parser.rule_logits.detach().cpu(),
            "term_logits": parser.term_logits.detach().cpu(),
            "metadata": {
                "num_nt": num_nt,
                "vocab_size": vocab_size,
                "non_terminals": non_terminals,
                "vocab": vocab,
            },
        },
        weights_path,
    )
    print(f"Pesi salvati in {weights_path}")

    # 2. Estrazione Grammatica Discreta con Pruning tau = 0.01
    grammar = PCFGrammar.from_model(
        parser,
        nt_labels=non_terminals,
        vocab_labels=vocab,
        start_symbol="S",
        prune_threshold=0.01,
        temperature=0.75,
    )

    json_path = "artifacts/grammar.json"
    yaml_path = "artifacts/grammar.yaml"
    gf_path = "artifacts/DynamicGrammarIta.gf"

    grammar.to_json(json_path)
    grammar.to_yaml(yaml_path)
    grammar.to_gf("DynamicGrammarIta", gf_path)

    print(f"Grammatica esportata in {json_path}, {yaml_path}, {gf_path}")

    # 3. Collaudo e Verifica Formale
    verifier = GrammarVerifier(grammar)
    lark_ok, err = verifier.verify_lark_syntax()
    print(f"Static Syntax Check: {'PASS' if lark_ok else f'FAIL ({err})'}")

    sample_strings = ["".join([vocab[t] for t in s]) for s in dataset.samples]
    cov, failed = verifier.verify_corpus_coverage(sample_strings)
    print(f"Corpus Coverage: {cov * 100:.2f}% (Target: 100%)")

    ast, root_score = parser.parse_viterbi([0, 1], nt_labels=non_terminals, vocab_labels=vocab)
    sound, status, _ = verifier.verify_smt_soundness(ast)
    print(f"SMT Solver Soundness Check: {status} (Sound: {sound})")


if __name__ == "__main__":
    main()
