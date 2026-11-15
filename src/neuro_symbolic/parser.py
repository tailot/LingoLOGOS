import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

# Sentinel log-probability of an impossible span inside the CYK chart. A finite value keeps
# logsumexp stable and lets the chart stay a plain dense tensor; UNPARSABLE_FLOOR is the
# cutoff that separates a real log-likelihood from that sentinel (see GrammarTrainer.evaluate).
CHART_IMPOSSIBLE = -1e9
UNPARSABLE_FLOOR = -1e8

# Temperature guard: dividing by a temperature that small would overflow the logits.
MIN_TEMPERATURE = 1e-5


def default_nt_labels(num_non_terminals: int) -> List[str]:
    """Neutral non-terminal labels: index 0 always names the start symbol S."""
    if num_non_terminals <= 0:
        return []
    return ["S"] + [f"NT_{index}" for index in range(1, num_non_terminals)]


def effective_temperature(temperature: Optional[float], fallback: float) -> float:
    """Resolve an optional temperature argument against the module default, with the guard."""
    return max(temperature if temperature is not None else fallback, MIN_TEMPERATURE)


class DifferentiableCYKParser(nn.Module):
    """
    Differentiable parsing engine based on the CYK algorithm (Inside Step).
    Computes the log-likelihood of input sequences given a parameterized PCFG.
    It also performs deterministic Viterbi parsing (AST extraction, entropy H(R)).

    DISTRIBUTION INVARIANT (requirement for PCFG validity):
        for every non-terminal A, the sum over ALL its productions must equal 1:
            sum_{B,C} P(A -> B C) + sum_{a} P(A -> a) = 1
    Joint normalization is obtained through a Bernoulli gate per non-terminal:
        P(A -> B C) = pi_A * softmax_rule[A,B,C]
        P(A -> a) = (1 - pi_A) * softmax_term[A,a]
    Historical note: the two families used to be normalized with two separate
    softmaxes, so the mass per non-terminal equalled 2 and P(S =>* X) was not a
    string probability (NLL and perplexity had no information-theoretic meaning).
    """

    def __init__(
        self,
        num_non_terminals: int,
        vocab_size: int,
        temperature: float = 1.0,
    ):
        super(DifferentiableCYKParser, self).__init__()
        self.num_nt = num_non_terminals
        self.vocab_size = vocab_size
        self.temperature = temperature

        # Non-terminal parameters: P(A -> B, C) in log-space
        # Shape: [NT, NT, NT] -> (A, B, C)
        self.rule_logits = nn.Parameter(
            torch.randn(self.num_nt, self.num_nt, self.num_nt) * 0.1
        )

        # Terminal parameters: P(A -> a) in log-space
        # Shape: [NT, Vocab] -> (A, a)
        self.term_logits = nn.Parameter(
            torch.randn(self.num_nt, self.vocab_size) * 0.1
        )

        # Family gate for each non-terminal A:
        #   pi_A = sigmoid(family_logits[A]) = P(A chooses a binary production)
        # P(A -> a) is therefore weighted by (1 - pi_A). Initialized to 0 (pi_A = 0.5).
        # Under a well-learned CNF regime the gate converges to 1 for the structural
        # non-terminals and to 0 for the ones emitting terminals.
        self.family_logits = nn.Parameter(torch.zeros(self.num_nt))

    def family_mass(self) -> torch.Tensor:
        """
        Returns pi_A = P(A -> binary production) for every non-terminal.
        The remaining mass (1 - pi_A) is reserved for the terminal productions A -> a.
        """
        return torch.sigmoid(self.family_logits)

    def get_log_rules(self, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Computes log P(A -> B C) with optional temperature.
        Shape: [NT, NT, NT]
        Joint normalized form: P(A -> B C) = pi_A * softmax_{B,C}(rule_logits[A] / T).
        """
        temp = effective_temperature(temperature, self.temperature)
        scaled_logits = self.rule_logits / temp
        # Normalization over all possible binary productions (B, C) for each A
        flat = F.log_softmax(scaled_logits.view(self.num_nt, -1), dim=-1)
        log_rules = flat.view(self.num_nt, self.num_nt, self.num_nt)
        # Bernoulli mass of the binary family (pi_A)
        log_binary_mass = F.logsigmoid(self.family_logits).view(self.num_nt, 1, 1)
        return log_rules + log_binary_mass

    def get_log_terms(self, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Computes log P(A -> a) with optional temperature.
        Shape: [NT, Vocab]
        Joint normalized form: P(A -> a) = (1 - pi_A) * softmax_a(term_logits[A] / T).
        """
        temp = effective_temperature(temperature, self.temperature)
        log_terms = F.log_softmax(self.term_logits / temp, dim=-1)
        # logsigmoid(-x) = log(1 - sigmoid(x)) = log(1 - pi_A)
        log_term_mass = F.logsigmoid(-self.family_logits).view(self.num_nt, 1)
        return log_terms + log_term_mass

    def production_entropy_per_lhs(self) -> torch.Tensor:
        """
        Joint entropy H(P(A -> .)) for every non-terminal A, computed as the
        entropy of the mixture over the two production families:
            H(A) = H(pi_A) + pi_A * H_rules(A) + (1 - pi_A) * H_terms(A)
        Shape: [NT]
        """
        pi = self.family_mass()

        p_rules = F.softmax(self.rule_logits.view(self.num_nt, -1), dim=-1)
        log_p_rules = F.log_softmax(self.rule_logits.view(self.num_nt, -1), dim=-1)
        h_rules = -torch.sum(p_rules * log_p_rules, dim=-1)

        p_terms = F.softmax(self.term_logits, dim=-1)
        log_p_terms = F.log_softmax(self.term_logits, dim=-1)
        h_terms = -torch.sum(p_terms * log_p_terms, dim=-1)

        h_gate = -(pi * F.logsigmoid(self.family_logits)) - (
            (1 - pi) * F.logsigmoid(-self.family_logits)
        )
        return h_gate + pi * h_rules + (1 - pi) * h_terms

    def production_mass_per_lhs(self, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Total probability mass P for every non-terminal A, summed over all
        productions (binary + terminal). For a valid PCFG it must equal 1.
        Shape: [NT]
        """
        log_rules = self.get_log_rules(temperature)
        log_terms = self.get_log_terms(temperature)
        binary_mass = torch.exp(log_rules).sum(dim=(-2, -1))
        term_mass = torch.exp(log_terms).sum(dim=-1)
        return binary_mass + term_mass

    def rule_entropy(self) -> torch.Tensor:
        """
        Entropy regularizer H(R) = sum_{A in V_N} H(P(A -> .)).
        H(P(A -> .)) is the joint mixture entropy over the productions of A
        (see production_entropy_per_lhs), so minimizing it pushes both the
        family choice (Bernoulli gate) and the individual production
        towards deterministic configurations.
        """
        return torch.sum(self.production_entropy_per_lhs())

    def forward(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
        return_chart: bool = False,
    ) -> torch.Tensor:
        """
        Runs the Inside pass in a fully differentiable way.
        :param x: tensor of shape [Batch_Size, Seq_Len] holding vocabulary indices.
        :param temperature: temperature used to scale the logits.
        :param return_chart: when True it also returns the parsing chart.
        :return: log-likelihood of the sequence derived from the start symbol S (index 0).
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Normalization in log-probability space
        log_rules = self.get_log_rules(temperature)
        log_terms = self.get_log_terms(temperature)

        # CYK Inside table: chart[batch, len, i, A]
        # Dimension 1: len_idx (0 to seq_len - 1)
        # Dimension 2: start_idx i (0 to seq_len - 1)
        # Dimension 3: non-terminal NT
        chart = torch.full(
            (batch_size, seq_len, seq_len, self.num_nt),
            fill_value=CHART_IMPOSSIBLE,
            device=device,
            dtype=self.rule_logits.dtype,
        )

        # 1. Base case: length 1. One gather over the terminal table replaces the
        # per-position / per-non-terminal assignment loop: log_terms[:, x] has shape
        # [NT, Batch, Seq_Len], the chart slice wants [Batch, Seq_Len, NT].
        chart[:, 0, :, :] = log_terms[:, x].permute(1, 2, 0)

        # View of the rule table broadcast over the batch, hoisted out of the CYK loops
        log_rules_batch = log_rules.unsqueeze(0)

        # 2. Recursive step: lengths from 2 up to seq_len
        for length in range(2, seq_len + 1):
            l_idx = length - 1
            for i in range(seq_len - length + 1):
                j = i + length - 1

                # Collector over the possible split points k
                split_logs = []
                for k in range(i, j):
                    left_len = k - i
                    right_len = j - (k + 1)

                    # Inside left: [Batch, NT_B] -> [Batch, NT_B, 1]
                    left_inside = chart[:, left_len, i, :].unsqueeze(2)
                    # Inside right: [Batch, NT_C] -> [Batch, 1, NT_C]
                    right_inside = chart[:, right_len, k + 1, :].unsqueeze(1)

                    # Combine Left and Right: [Batch, NT_B, NT_C]
                    pair_inside = left_inside + right_inside

                    # Adds the log-probabilities of the rules [NT_A, NT_B, NT_C]
                    # Resulting shape: [Batch, NT_A]
                    combined = torch.logsumexp(
                        pair_inside.unsqueeze(1) + log_rules_batch,
                        dim=(-2, -1),
                    )
                    split_logs.append(combined)

                # LogSumExp over the split points k
                chart[:, l_idx, i, :] = torch.logsumexp(
                    torch.stack(split_logs, dim=-1), dim=-1
                )

        # Reads the log-probability of the whole sequence off the start symbol S (index 0)
        root_log_prob = chart[:, seq_len - 1, 0, 0]
        if return_chart:
            return root_log_prob, chart
        return root_log_prob

    def parse_viterbi(
        self,
        tokens: List[int],
        nt_labels: Optional[List[str]] = None,
        vocab_labels: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Runs Viterbi CYK parsing (max instead of logsumexp) to extract the
        most probable derivation tree (AST).
        """
        seq_len = len(tokens)
        if seq_len == 0:
            return None, float("-inf")

        log_rules = self.get_log_rules(temperature).detach().cpu()
        log_terms = self.get_log_terms(temperature).detach().cpu()

        # The search touches every (span, split, A, B, C) candidate: reading the tables
        # through scalar tensor indexing would allocate one tensor object per access, so
        # the two tables and the chart are carried as plain Python float lists.
        impossible = float("-inf")
        rule_scores: List[List[List[float]]] = log_rules.tolist()
        term_scores: List[List[float]] = log_terms.tolist()

        # best_score[l_idx][i][nt]
        best_score: List[List[List[float]]] = [
            [[impossible] * self.num_nt for _ in range(seq_len)] for _ in range(seq_len)
        ]
        # backpointer[(l_idx, i, nt)] = (k, B, C) or the terminal leaf
        bp: Dict[Tuple[int, int, int], Any] = {}

        if nt_labels is None:
            nt_labels = default_nt_labels(self.num_nt)
        if vocab_labels is None:
            vocab_labels = [str(token) for token in range(self.vocab_size)]

        # Base case
        for i, token in enumerate(tokens):
            for nt in range(self.num_nt):
                best_score[0][i][nt] = term_scores[nt][token]
                bp[(0, i, nt)] = ("terminal", vocab_labels[token])

        # Recursive step
        for length in range(2, seq_len + 1):
            l_idx = length - 1
            for i in range(seq_len - length + 1):
                j = i + length - 1
                for nt_a in range(self.num_nt):
                    max_val = impossible
                    best_cand = None
                    rules_for_a = rule_scores[nt_a]

                    for k in range(i, j):
                        left_len = k - i
                        right_len = j - (k + 1)
                        left_span = best_score[left_len][i]
                        right_span = best_score[right_len][k + 1]
                        for nt_b in range(self.num_nt):
                            s_left = left_span[nt_b]
                            if s_left == impossible:
                                continue
                            rules_for_b = rules_for_a[nt_b]
                            for nt_c in range(self.num_nt):
                                s_right = right_span[nt_c]
                                if s_right == impossible:
                                    continue
                                score = rules_for_b[nt_c] + s_left + s_right
                                if score > max_val:
                                    max_val = score
                                    best_cand = (k, left_len, right_len, nt_b, nt_c)

                    best_score[l_idx][i][nt_a] = max_val
                    if best_cand is not None:
                        bp[(l_idx, i, nt_a)] = best_cand

        root_score = best_score[seq_len - 1][0][0]
        if root_score == impossible:
            return None, impossible

        def build_tree(l_idx: int, i: int, nt: int) -> Dict[str, Any]:
            node_label = nt_labels[nt]
            info = bp.get((l_idx, i, nt))
            if info is None:
                return {"type": node_label, "children": []}
            if info[0] == "terminal":
                return {"type": node_label, "token": info[1]}

            k, left_len, right_len, nt_b, nt_c = info
            left_subtree = build_tree(left_len, i, nt_b)
            right_subtree = build_tree(right_len, k + 1, nt_c)
            return {
                "type": node_label,
                "children": [left_subtree, right_subtree],
            }

        ast = build_tree(seq_len - 1, 0, 0)
        return ast, root_score
