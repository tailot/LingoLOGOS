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
    Sample generator and container for the formal Dyck-2 language (round and square brackets).
    Vocabulary: 0: '(', 1: ')', 2: '[', 3: ']'
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
        Generates non-empty balanced Dyck-2 sentences of bounded even length.
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

    def split(
        self,
        test_fraction: float = 0.25,
        seed: int = 42,
    ) -> Tuple["DyckDataset", "DyckDataset"]:
        """
        Random train / test split OVER SENTENCES (the dataset holds unique sentences, so the
        split is duplicate-free by construction).
        Returns (train, test) as two disjoint DyckDataset objects.
        """
        shuffled = list(self.samples)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        cut = max(1, int(round(len(shuffled) * (1.0 - test_fraction))))
        return DyckDataset(shuffled[:cut]), DyckDataset(shuffled[cut:])

    def as_strings(self) -> List[str]:
        """Decodes the dataset sentences into strings of terminals."""
        inverse = {index: symbol for symbol, index in self.VOCAB_MAP.items()}
        return ["".join(inverse[token] for token in sample) for sample in self.samples]


class GrammarTrainer:
    """
    Training pipeline for neuro-symbolic grammar induction.
    Optimizes marginal NLL + entropy regularizer H(R) + L2 weight decay.
    """

    def __init__(
        self,
        parser: DifferentiableCYKParser,
        learning_rate: float = 3.5e-3,
        temperature_tau: float = 0.75,
        entropy_penalty: float = 0.05,
        weight_decay: float = 1e-4,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.parser = parser
        self.learning_rate = learning_rate
        self.temperature_tau = temperature_tau
        self.entropy_penalty = entropy_penalty
        self.weight_decay = weight_decay
        # Gradient clipping: PCFG logits can grow a lot during temperature annealing;
        # without clipping the last steps diverge.
        self.max_grad_norm = max_grad_norm

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
        Computes the global objective:
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
        # Group by length to keep CYK batches homogeneous
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

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parser.parameters(), self.max_grad_norm
                    )

                # Monitor gradient stability
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
        Model-level metrics on a dataset (report the TEST block, not the training one).

        - `mean_nll_per_symbol` : -(1 / N_symbols) * sum_X log P(S =>* X) over the corpus.
          This is a cross-entropy: the lower bound is 0 and there is no upper bound
          (an untrained parser can exceed the 2 bits per symbol of log2(|vocab|)).
          Under the distribution invariant (the productions of every non-terminal sum to 1)
          the value is however comparable with the perplexities reported in the literature.
        - `ppl_per_symbol`      : exp(mean_nll_per_symbol). This is a MARGINAL perplexity
          per symbol: exp(-(1/N) sum log P(X)), not the autoregressive conditional
          perplexity exp(-(1/N) sum log P(x_i | x_<i)). A chart parser does not expose
          position-wise conditional distributions, so the definition must be stated.
        - `model_coverage`      : fraction of sentences with non-zero inside probability.
          NB: this is NOT parsing accuracy. A dense PCFG has coverage ~1 on any string;
          correctness is measured by the negative controls
          (GrammarVerifier.verify_negative_controls) and by the gold rule F1.
        - `rule_sparsity`       : fraction of productions with P < 1e-4.
        - `max_lhs_mass_error`  : largest deviation from unit mass per non-terminal.
        """
        self.parser.eval()
        total_log_prob = 0.0
        total_symbols = 0
        parsable = 0

        with torch.no_grad():
            for s in dataset.samples:
                x = torch.tensor([s], dtype=torch.long)
                lp = self.parser(x, temperature=self.temperature_tau).item()
                if lp > -1e8:
                    total_log_prob += lp
                    parsable += 1
                total_symbols += len(s)

        # Information-theoretic cost per symbol: -(1/N) * sum_X log P(S =>* X)
        mean_nll_per_symbol = -total_log_prob / max(total_symbols, 1)
        ppl = math.exp(min(mean_nll_per_symbol, 60.0)) if total_symbols > 0 else float("inf")

        # Fraction of productions with P(r) < 1e-4 over the joint distribution
        # (the family gate is already included in those probabilities)
        log_rules = self.parser.get_log_rules(self.temperature_tau)
        log_terms = self.parser.get_log_terms(self.temperature_tau)
        p_rules = torch.exp(log_rules).view(-1)
        p_terms = torch.exp(log_terms).view(-1)
        all_probs = torch.cat([p_rules, p_terms])
        sparse_count = (all_probs < 1e-4).float().sum().item()
        sparsity = sparse_count / all_probs.numel()

        # Learned distribution invariant: the mass per non-terminal must equal 1
        masses = self.parser.production_mass_per_lhs(self.temperature_tau)

        return {
            "mean_nll_per_symbol": mean_nll_per_symbol,
            "ppl_per_symbol": ppl,
            "model_coverage": parsable / max(len(dataset.samples), 1),
            "rule_sparsity": sparsity,
            "max_lhs_mass_error": max(abs(m - 1.0) for m in masses.tolist()),
        }

