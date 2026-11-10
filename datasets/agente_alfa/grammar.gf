-- DynamicGrammar concrete rules for agente_alfa
abstract GrammarAlfa = {
  cat
    State ;
    Action ;
  fun
    StartAnalysis : State -> Action ;
    SystemRoot : State ;
}

concrete GrammarAlfaEng of GrammarAlfa = {
  lincat
    State = Str ;
    Action = Str ;
  lin
    StartAnalysis state = "Start analysis of " ++ state ;
    SystemRoot = "SystemRoot" ;
}
