-- DynamicGrammar concrete rules for arbitro_logico
abstract GrammarArbitro = {
  cat
    State ;
    Action ;
  fun
    ArbitrateConsensus : State -> Action ;
    ValidationRoot : State ;
}

concrete GrammarArbitroEng of GrammarArbitro = {
  lincat
    State = Str ;
    Action = Str ;
  lin
    ArbitrateConsensus state = "Arbitrate consensus on " ++ state ;
    ValidationRoot = "ValidationRoot" ;
}
