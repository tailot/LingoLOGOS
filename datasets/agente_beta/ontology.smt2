; SMT-LIB 2 Ontology for agente_beta
(declare-sort State)
(declare-fun safe_transition (State State) Bool)
(declare-const s_bad State)
(assert (not (safe_transition s_bad s_bad)))
