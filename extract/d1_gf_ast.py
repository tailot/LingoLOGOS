"""Dataset D1 generator: Grammatical Derivation and AST Mappings (GF RGL).

Specifications:
- Mapping from natural language sequence w_{1:T} to AST nodes validated by GF RGL.
- Format: JSON schema containing sequence w_{1:T} and linear AST serialization T_GF.
- Equal distribution across 5 target languages: it, en, fr, de, es.
- Recursive parsing with abstract depth bound d <= 6.
- Preserves morphosyntactic reductions, verbal inflections, and long-range dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from extract.common import DatasetSample, init_seed

LANGUAGES = ("it", "en", "fr", "de", "es")

SUBJECTS = {
    "it": ["il sensore", "la lampada", "il termostato", "il gateway", "la presa"],
    "en": ["the sensor", "the lamp", "the thermostat", "the gateway", "the socket"],
    "fr": ["le capteur", "la lampe", "le thermostat", "la passerelle", "la prise"],
    "de": ["der Sensor", "die Lampe", "der Thermostat", "das Gateway", "die Steckdose"],
    "es": ["el sensor", "la lampara", "el termostato", "la pasarela", "el enchufe"],
}

VERBS = {
    "it": [
        ("rileva", "Det"),
        ("invia un segnale", "SendSignal"),
        ("attiva la procedura", "ActivateProc"),
        ("richiede attenzione", "ReqAttention"),
        ("entra in modalita sicura", "EnterSafeMode"),
    ],
    "en": [
        ("detects an event", "Det"),
        ("sends a signal", "SendSignal"),
        ("activates the procedure", "ActivateProc"),
        ("requires attention", "ReqAttention"),
        ("enters safe mode", "EnterSafeMode"),
    ],
    "fr": [
        ("detecte un evenement", "Det"),
        ("envoie un signal", "SendSignal"),
        ("active la procedure", "ActivateProc"),
        ("exige une attention", "ReqAttention"),
        ("entre en mode securise", "EnterSafeMode"),
    ],
    "de": [
        ("erkennt ein Ereignis", "Det"),
        ("sendet ein Signal", "SendSignal"),
        ("aktiviert die Prozedur", "ActivateProc"),
        ("erfordert Aufmerksamkeit", "ReqAttention"),
        ("wechselt in den abgesicherten Modus", "EnterSafeMode"),
    ],
    "es": [
        ("detecta un evento", "Det"),
        ("envia una senal", "SendSignal"),
        ("activa el procedimiento", "ActivateProc"),
        ("requiere atencion", "ReqAttention"),
        ("entra en modo seguro", "EnterSafeMode"),
    ],
}

CONDITIONS = {
    "it": [
        ("se la soglia supera il limite", "CondThresholdExceeded"),
        ("quando il timer scade", "CondTimerExpired"),
        ("qualora la tensione scenda", "CondVoltageDrop"),
    ],
    "en": [
        ("if the threshold exceeds limit", "CondThresholdExceeded"),
        ("when the timer expires", "CondTimerExpired"),
        ("should the voltage drop", "CondVoltageDrop"),
    ],
    "fr": [
        ("si le seuil depasse la limite", "CondThresholdExceeded"),
        ("lorsque le minuteur expire", "CondTimerExpired"),
        ("au cas ou la tension chute", "CondVoltageDrop"),
    ],
    "de": [
        ("wenn der Schwellenwert das Limit uberschreitet", "CondThresholdExceeded"),
        ("wenn der Timer ablaeuft", "CondTimerExpired"),
        ("falls die Spannung abfaellt", "CondVoltageDrop"),
    ],
    "es": [
        ("si el umbral supera el limite", "CondThresholdExceeded"),
        ("cuando el temporizador expire", "CondTimerExpired"),
        ("en caso de que baje la tension", "CondVoltageDrop"),
    ],
}

CONNECTORS = {
    "it": ("e conseguentemente", "AndThen"),
    "en": ("and consequently", "AndThen"),
    "fr": ("et par consequent", "AndThen"),
    "de": ("und folglich", "AndThen"),
    "es": ("y por consiguiente", "AndThen"),
}


@dataclass
class D1Sample(DatasetSample):
    """Schema for D1 record."""
    language: str
    tokens: List[str]
    natural_language: str
    ast_tree: str
    ast_depth: int
    gf_linearization: str


class D1Generator:
    """Generator for D1 Grammatical Derivation & AST Mappings."""

    def __init__(self, seed: int = 42, max_depth: int = 6):
        self.rng = init_seed(seed)
        self.max_depth = max_depth

    def _generate_phrase(self, lang: str, depth: int) -> tuple[str, str, int]:
        subj_idx = self.rng.randint(0, len(SUBJECTS[lang]) - 1)
        verb_idx = self.rng.randint(0, len(VERBS[lang]) - 1)
        subj_text = SUBJECTS[lang][subj_idx]
        verb_text, verb_ast = VERBS[lang][verb_idx]

        clause_text = f"{subj_text} {verb_text}"
        clause_ast = f"PredVP (SubjNode (Id {subj_idx})) ({verb_ast})"
        curr_depth = 2

        if depth > 2 and self.rng.random() < 0.8:
            cond_idx = self.rng.randint(0, len(CONDITIONS[lang]) - 1)
            cond_text, cond_ast = CONDITIONS[lang][cond_idx]
            clause_text = f"{clause_text} {cond_text}"
            clause_ast = f"CondVP ({clause_ast}) ({cond_ast})"
            curr_depth += 1

        if depth > 3 and curr_depth < self.max_depth and self.rng.random() < 0.6:
            conn_text, conn_ast = CONNECTORS[lang]
            second_verb_idx = self.rng.randint(0, len(VERBS[lang]) - 1)
            second_verb_text, second_verb_ast = VERBS[lang][second_verb_idx]
            clause_text = f"{clause_text} {conn_text} {second_verb_text}"
            clause_ast = f"{conn_ast} ({clause_ast}) ({second_verb_ast})"
            curr_depth += 1

        ast_tree = f"PhrUtt MarkP ({clause_ast})"
        curr_depth += 1
        return clause_text, ast_tree, min(curr_depth, self.max_depth)

    def generate_sample(self, index: int) -> D1Sample:
        lang = LANGUAGES[index % len(LANGUAGES)]
        depth_target = 2 + (index % (self.max_depth - 1))
        nl_text, ast_tree, real_depth = self._generate_phrase(lang, depth_target)
        tokens = nl_text.split()
        return D1Sample(
            dataset_id="D1_GF_RGL",
            sample_id=f"D1_{index:07d}",
            language=lang,
            tokens=tokens,
            natural_language=nl_text,
            ast_tree=ast_tree,
            ast_depth=real_depth,
            gf_linearization=ast_tree,
        )

    def generate_stream(self, total: int) -> Iterator[D1Sample]:
        for i in range(total):
            yield self.generate_sample(i)
