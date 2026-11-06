import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List


class DifferentiableCYKParser(nn.Module):
    """
    Motore di parsing differenziabile basato sull'algoritmo CYK (Inside Step).
    Calcola la log-verosimiglianza delle sequenze di input data una PCFG parametrizzata.
    Include anche Viterbi parsing deterministico per l'estrazione di AST e calcolo dell'entropia H(R).
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

        # Parametri non-terminali: P(A -> B, C) in Log-space
        # Shape: [NT, NT, NT] -> (A, B, C)
        self.rule_logits = nn.Parameter(
            torch.randn(self.num_nt, self.num_nt, self.num_nt) * 0.1
        )

        # Parametri terminali: P(A -> a) in Log-space
        # Shape: [NT, Vocab] -> (A, a)
        self.term_logits = nn.Parameter(
            torch.randn(self.num_nt, self.vocab_size) * 0.1
        )

    def get_log_rules(self, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Calcola log P(A -> B C) con temperatura opzionale.
        Shape: [NT, NT, NT]
        """
        temp = temperature if temperature is not None else self.temperature
        scaled_logits = self.rule_logits / max(temp, 1e-5)
        # Normalizzazione su tutte le possibili produzioni binarie (B, C) per ogni A
        log_rules = F.log_softmax(scaled_logits.view(self.num_nt, -1), dim=-1).view(
            self.num_nt, self.num_nt, self.num_nt
        )
        return log_rules

    def get_log_terms(self, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Calcola log P(A -> a) con temperatura opzionale.
        Shape: [NT, Vocab]
        """
        temp = temperature if temperature is not None else self.temperature
        scaled_logits = self.term_logits / max(temp, 1e-5)
        log_terms = F.log_softmax(scaled_logits, dim=-1)
        return log_terms

    def rule_entropy(self) -> torch.Tensor:
        """
        Calcola la regolarizzazione di entropia H(R) = sum_{A in V_N} H(P(A -> .)).
        H(P(A -> .)) = - sum_alpha P(A -> alpha) * log P(A -> alpha)
        Include sia produzioni non-terminali che terminali.
        """
        # Entropia sulle regole binarie P(A -> B C)
        p_rules = F.softmax(self.rule_logits.view(self.num_nt, -1), dim=-1)
        log_p_rules = F.log_softmax(self.rule_logits.view(self.num_nt, -1), dim=-1)
        h_rules = -torch.sum(p_rules * log_p_rules, dim=-1)

        # Entropia sulle emissioni terminali P(A -> a)
        p_terms = F.softmax(self.term_logits, dim=-1)
        log_p_terms = F.log_softmax(self.term_logits, dim=-1)
        h_terms = -torch.sum(p_terms * log_p_terms, dim=-1)

        return torch.sum(h_rules + h_terms)

    def forward(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
        return_chart: bool = False,
    ) -> torch.Tensor:
        """
        Esegue il passaggio Inside in modo differenziabile.
        :param x: Tensor di forma [Batch_Size, Seq_Len] con gli indici del vocabolario.
        :param temperature: Temperatura per lo scaling dei logits.
        :param return_chart: Se True restituisce anche la chart di parsing.
        :return: Log-likelihood della sequenza generata dal simbolo iniziale S (index 0).
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Normalizzazione in log-probabilities
        log_rules = self.get_log_rules(temperature)
        log_terms = self.get_log_terms(temperature)

        # Tabella CYK Inside: chart[batch, len, i, A]
        # Dimensione 1: len_idx (0 to seq_len - 1)
        # Dimensione 2: start_idx i (0 to seq_len - 1)
        # Dimensione 3: non-terminal NT
        chart = torch.full(
            (batch_size, seq_len, seq_len, self.num_nt),
            fill_value=-1e9,
            device=device,
            dtype=self.rule_logits.dtype,
        )

        # 1. Caso Base: Lunghezza = 1
        for i in range(seq_len):
            tokens = x[:, i]  # [Batch]
            for nt in range(self.num_nt):
                chart[:, 0, i, nt] = log_terms[nt, tokens]

        # 2. Passo Ricorsivo: Lunghezze da 2 a seq_len
        for length in range(2, seq_len + 1):
            l_idx = length - 1
            for i in range(seq_len - length + 1):
                j = i + length - 1

                # Vettore dei possibili punti di split k
                split_logs = []
                for k in range(i, j):
                    left_len = k - i
                    right_len = j - (k + 1)

                    # Inside left: [Batch, NT_B] -> [Batch, NT_B, 1]
                    left_inside = chart[:, left_len, i, :].unsqueeze(2)
                    # Inside right: [Batch, NT_C] -> [Batch, 1, NT_C]
                    right_inside = chart[:, right_len, k + 1, :].unsqueeze(1)

                    # Combina Left e Right: [Batch, NT_B, NT_C]
                    pair_inside = left_inside + right_inside

                    # Aggiunge le log-probabilità delle regole [NT_A, NT_B, NT_C]
                    # Resulting shape: [Batch, NT_A]
                    combined = torch.logsumexp(
                        pair_inside.unsqueeze(1) + log_rules.unsqueeze(0),
                        dim=(-2, -1),
                    )
                    split_logs.append(combined)

                # LogSumExp sui punti di split k
                chart[:, l_idx, i, :] = torch.logsumexp(
                    torch.stack(split_logs, dim=-1), dim=-1
                )

        # Estraggo la log-probabilità dal Simbolo Iniziale S (indice 0) per l'intera sequenza
        root_log_prob = chart[:, seq_len - 1, 0, 0]
        if return_chart:
            return root_log_prob, chart
        return root_log_prob

    def parse_viterbi(
        self,
        tokens: List[int],
        nt_labels: Optional[List[str]] = None,
        vocab_labels: Optional[List[str]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Esegue il Viterbi CYK Parsing (max invece di logsumexp)
        per estrarre l'albero di derivazione più probabile (AST).
        """
        seq_len = len(tokens)
        if seq_len == 0:
            return None, float("-inf")

        log_rules = self.get_log_rules().detach().cpu()
        log_terms = self.get_log_terms().detach().cpu()

        # best_score[len_idx, i, nt]
        best_score = torch.full(
            (seq_len, seq_len, self.num_nt), fill_value=float("-inf")
        )
        # backpointer[len_idx, i, nt] = (k, B, C)
        bp = {}

        if nt_labels is None:
            nt_labels = [f"NT_{i}" if i > 0 else "S" for i in range(self.num_nt)]
        if vocab_labels is None:
            vocab_labels = [str(t) for t in range(self.vocab_size)]

        # Base case
        for i in range(seq_len):
            tok = tokens[i]
            for nt in range(self.num_nt):
                best_score[0, i, nt] = log_terms[nt, tok].item()
                bp[(0, i, nt)] = ("terminal", vocab_labels[tok])

        # Recursive step
        for length in range(2, seq_len + 1):
            l_idx = length - 1
            for i in range(seq_len - length + 1):
                j = i + length - 1
                for nt_a in range(self.num_nt):
                    max_val = float("-inf")
                    best_cand = None

                    for k in range(i, j):
                        left_len = k - i
                        right_len = j - (k + 1)
                        for nt_b in range(self.num_nt):
                            s_left = best_score[left_len, i, nt_b].item()
                            if s_left == float("-inf"):
                                continue
                            for nt_c in range(self.num_nt):
                                s_right = best_score[right_len, k + 1, nt_c].item()
                                if s_right == float("-inf"):
                                    continue
                                score = (
                                    log_rules[nt_a, nt_b, nt_c].item()
                                    + s_left
                                    + s_right
                                )
                                if score > max_val:
                                    max_val = score
                                    best_cand = (k, left_len, right_len, nt_b, nt_c)

                    best_score[l_idx, i, nt_a] = max_val
                    if best_cand is not None:
                        bp[(l_idx, i, nt_a)] = best_cand

        root_score = best_score[seq_len - 1, 0, 0].item()
        if root_score == float("-inf"):
            return None, float("-inf")

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
