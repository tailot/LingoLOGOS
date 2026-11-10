; SMT-LIB 2 Ontology for arbitro_logico
(declare-sort State)
(declare-fun safe_transition (State State) Bool)
(declare-fun valid_state (State) Bool)
(assert (forall ((s State)) (=> (valid_state s) (safe_transition s s))))
