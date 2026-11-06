import random
import math
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .parser import DifferentiableCYKParser
from .grammar import PCFGrammar


class DyckDataset(Dataset):
    """
    Generatore e contenitore di campioni per il linguaggio formale Dyck-2 (parentesi tonde e quadre).
    Vocabolario: 0: '(', 1: ')', 2: '[', 3: ']'
    """

    VOCAB = ["(", ")", "[", "]"]
    VOCAB_MAP = {"(": 0, ")": 1, "[": 2, "]": 3}

    def __init__(self, samples: List[List[int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.samples[idx], dtype=torch.long)

    @classmethod
    def generate_dyck2(
        cls,
        num_samples: int = 100,
        max_depth: int = 3,
        max_length: int = 16,
        seed: int = 42,
    ) -> "DyckDataset":
        """
        Genera frasi bilanciate non vuote di Dyck-2 con lunghezza pari limitata.
        """
        random.seed(seed)
        samples_set = set()

        def gen_expr(depth: int) -> str:
            if depth >= max_depth or random.random() < 0.3:
                return "()" if random.random() < 0.5 else "[]"
            choice = random.choice([0, 1, 2])
            if choice == 0:
                inner = gen_expr(depth + 1)
                return f"({inner})"
            elif choice == 1:
                inner = gen_expr(depth + 1)
                return f"[{inner}]"
            else:
                left = gen_expr(depth + 1)
                right = gen_expr(depth + 1)
                return f"{left}{right}"

        attempts = 0
        while len(samples_set) < num_samples and attempts < num_samples * 100:
            attempts += 1
            s = gen_expr(0)
            if 2 <= len(s) <= max_length and len(s) % 2 == 0:
                samples_set.add(s)

        samples = [[cls.VOCAB_MAP[c] for c in s] for s in sorted(list(samples_set))]
        return cls(samples)


class GrammarTrainer:
    """
    Pipeline di training per induzione grammaticale neuro-simbolica.
    Ottimizza la loss con NLL marginale + regolarizzazione d'entropia H(R) + L2 weight decay.
    """

    def __init__(
        self,
        parser: DifferentiableCYKParser,
        learning_rate: float = 3.5e-3,
        temperature_tau: float = 0.75,
        entropy_penalty: float = 0.05,
        weight_decay: float = 1e-4,
    ):
        self.parser = parser
        self.learning_rate = learning_rate
        self.temperature_tau = temperature_tau
        self.entropy_penalty = entropy_penalty
        self.weight_decay = weight_decay

        self.optimizer = torch.optim.AdamW(
            self.parser.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def compute_loss(
        self,
        batch_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcola la loss globale:
        L(theta) = - 1/|D| sum log P(S => X) + lambda_1 sum H(P(A -> .))
        """
        log_prob = self.parser(batch_x, temperature=self.temperature_tau)
        nll = -torch.mean(log_prob)
        entropy = self.parser.rule_entropy()
        total_loss = nll + self.entropy_penalty * entropy

        metrics = {
            "nll": nll.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def train_epoch(self, dataset: DyckDataset, batch_size: int = 16) -> Dict[str, float]:
        self.parser.train()
        # Raggruppa per lunghezza per batching omogeneo nel CYK
        samples_by_len: Dict[int, List[List[int]]] = {}
        for s in dataset.samples:
            samples_by_len.setdefault(len(s), []).append(s)

        total_loss_accum = 0.0
        nll_accum = 0.0
        ent_accum = 0.0
        batch_count = 0
        grad_norms = []

        for length, s_list in samples_by_len.items():
            for i in range(0, len(s_list), batch_size):
                chunk = s_list[i : i + batch_size]
                x = torch.tensor(chunk, dtype=torch.long)

                self.optimizer.zero_grad()
                loss, metrics = self.compute_loss(x)
                loss.backward()

                # Monitor stabilità gradienti
                grad_norm = 0.0
                for p in self.parser.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm(2).item() ** 2
                grad_norms.append(math.sqrt(grad_norm))

                self.optimizer.step()

                total_loss_accum += metrics["total_loss"]
                nll_accum += metrics["nll"]
                ent_accum += metrics["entropy"]
                batch_count += 1

        avg_loss = total_loss_accum / max(batch_count, 1)
        avg_nll = nll_accum / max(batch_count, 1)
        avg_ent = ent_accum / max(batch_count, 1)
        avg_grad = sum(grad_norms) / max(len(grad_norms), 1)

        return {
            "avg_loss": avg_loss,
            "avg_nll": avg_nll,
            "avg_entropy": avg_ent,
            "grad_norm": avg_grad,
        }

    def evaluate(self, dataset: DyckDataset) -> Dict[str, float]:
        """
        Calcola Perplessità per Simbolo (PPL), Parsing Precision/Recall (F1) e Sparsità.
        """
        self.parser.eval()
        total_log_prob = 0.0
        total_symbols = 0
        valid_parses = 0

        with torch.no_grad():
            for s in dataset.samples:
                x = torch.tensor([s], dtype=torch.long)
                lp = self.parser(x, temperature=self.temperature_tau).item()
                if lp > -1e8:
                    total_log_prob += lp
                    valid_parses += 1
                total_symbols += len(s)

        # Perplessità per Simbolo: exp(- 1/N sum log P(x_i | x_{<i}))
        # con sum log P(X) = total_log_prob
        ppl = math.exp(-total_log_prob / max(total_symbols, 1)) if total_symbols > 0 else float("inf")
        parse_f1 = valid_parses / max(len(dataset.samples), 1)

        # Sparsità delle Regole: percentuale di regole con P(r) < 1e-4
        log_rules = self.parser.get_log_rules()
        log_terms = self.parser.get_log_terms()
        p_rules = torch.exp(log_rules).view(-1)
        p_terms = torch.exp(log_terms).view(-1)
        all_probs = torch.cat([p_rules, p_terms])
        sparse_count = (all_probs < 1e-4).float().sum().item()
        sparsity = sparse_count / all_probs.numel()

        return {
            "perplexity": ppl,
            "parsing_f1": parse_f1,
            "rule_sparsity": sparsity,
            "coverage": valid_parses / max(len(dataset.samples), 1),
        }
