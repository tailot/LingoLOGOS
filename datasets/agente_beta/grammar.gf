-- DynamicGrammar concrete rules for agente_beta
abstract GrammarBeta = {
  cat
    State ;
    Action ;
  fun
    InitMonitor : State -> Action ;
    SafetyBoundary : State ;
}

concrete GrammarBetaEng of GrammarBeta = {
  lincat
    State = Str ;
    Action = Str ;
  lin
    InitMonitor state = "Initialize monitor on " ++ state ;
    SafetyBoundary = "SafetyBoundary" ;
}
