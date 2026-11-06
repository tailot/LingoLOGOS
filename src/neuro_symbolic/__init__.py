"""
Neuro-symbolic grammar induction and synthesis engine.
"""

from .parser import DifferentiableCYKParser
from .grammar import PCFGrammar, ProductionRule, Symbol
from .trainer import GrammarTrainer, DyckDataset
from .verification import (
    GrammarVerifier,
    SoundnessReport,
    counter_signature,
    is_well_bracketed,
)
from .reference import (
    GOLD_NON_TERMINALS,
    GOLD_RULE_TRIPLES,
    gold_dyck2_grammar,
    generate_negative_controls,
    gold_rule_f1,
)

__all__ = [
    "DifferentiableCYKParser",
    "PCFGrammar",
    "ProductionRule",
    "Symbol",
    "GrammarTrainer",
    "DyckDataset",
    "GrammarVerifier",
    "SoundnessReport",
    "counter_signature",
    "is_well_bracketed",
    "GOLD_NON_TERMINALS",
    "GOLD_RULE_TRIPLES",
    "gold_dyck2_grammar",
    "generate_negative_controls",
    "gold_rule_f1",
]
