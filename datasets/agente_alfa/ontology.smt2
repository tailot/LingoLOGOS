; SMT-LIB 2 Ontology for agente_alfa
(declare-sort State)
(declare-fun safe_transition (State State) Bool)
(declare-const s_init State)
(assert (safe_transition s_init s_init))
