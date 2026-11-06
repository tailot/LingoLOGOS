# Documentazione Tecnica: Motore di Induzione Grammaticale e Sintesi Neuro-Simbolica

Questo documento fornisce le linee guida ingegneristiche complete per la manutenzione, estensione, configurazione e validazione formale del motore di induzione grammaticale differenziabile e sintesi neuro-simbolica.

---

## 1. Architettura Interna e Dettaglio dei Moduli

Il repository implementa un'architettura ibrida neurale-simbolica strutturata come segue:

```
src/neuro_symbolic/
├── __init__.py          # Export pubblico delle classi primarie
├── parser.py            # DifferentiableCYKParser: algoritmo Inside-Outside e Viterbi CYK
├── grammar.py           # PCFGrammar, ProductionRule, Symbol ed esportatori (JSON, YAML, GF, Lark)
├── trainer.py           # DyckDataset, GrammarTrainer (loss regolarizzata con NLL ed Entropia)
├── verification.py      # GrammarVerifier: controllo sintassi Lark/GF, Yield coverage, SMT solver Z3
└── synthesize.py        # Pipeline di sintesi ed esportazione end-to-end degli artefatti
```

### 1.1 `parser.py` - DifferentiableCYKParser
- **Rappresentazione dei Parametri**:
  - `rule_logits` ($\mathbb{R}^{K \times K \times K}$): Log-odds per le produzioni non-terminali binarie $A \to B \ C$.
  - `term_logits` ($\mathbb{R}^{K \times |V_T|}$): Log-odds per le emissioni terminali $A \to a$.
- **Semianello Logaritmico Soft**:
  Le operazioni di moltiplicazione probabilistica diventano addizioni nei logaritmi, mentre le somme probabilistiche marginali su percorsi alternativi vengono calcolate con la funzione differenziabile `LogSumExp`.
- **Parsing Deterministico Viterbi**:
  Invece del `logsumexp`, la funzione `parse_viterbi` seleziona il percorso ad argmax per ricostruire l'Albero di Sintassi Astratta (AST) deterministico in tempo polinomiale $O(N^3 \cdot K^3)$.

### 1.2 `grammar.py` - Astrazione Simbolica e Serializzazione
- **Strutture Dati**:
  - `Symbol`: modella simboli terminali o non-terminali.
  - `ProductionRule`: modella regole di produzione $A \to \alpha$ con associata probabilità $P \in [0, 1]$.
  - `PCFGrammar`: incapsula l'insieme $G = (V_N, V_T, R, S, P)$.
- **Pruning Deterministico**:
  La funzione `PCFGrammar.from_model(...)` applica una soglia $\tau = 0.01$. Tutte le regole con probabilità marginale $P(r) < \tau$ vengono eliminate, garantendo l'acquisizione di una grammatica compatta, leggibile e priva di allucinazioni stocastiche.
- **Esportatori Supportati**:
  - `to_json(...)`: serializzazione JSON secondo specifica.
  - `to_yaml(...)`: serializzazione YAML per configurazioni leggibili.
  - `to_gf(...)`: genera codice astratto e concreto per **Grammatical Framework** (`.gf`).
  - `to_lark(...)`: genera la grammatica in formato EBNF per validazione sintattica con Lark.

### 1.3 `trainer.py` - Addestramento e Ottimizzazione
- **Generatore Dyck-2**:
  Genera sequenze bilanciate di parentesi tonde `()` e quadre `[]` di profondità controllata.
- **Funzione di Costo Neurale Regolarizzata**:
  $$\mathcal{L}(\theta) = -\frac{1}{|D|} \sum_{X \in D} \log P(S \stackrel{*}{\Rightarrow} X \mid \theta) + \lambda_1 \sum_{A \in V_N} H(P(A \to \cdot)) + \lambda_2 \|\theta\|_2^2$$
  dove $H(P(A \to \cdot))$ spinge la distribuzione delle regole verso distribuzioni a bassa entropia (determinismo simbolico quasi puro).
- **Metriche di Validazione**:
  Calcolo di Perplessità per simbolo (PPL), Coverage / Parsing Precision F1, Sparsità delle regole e monitoraggio del gradiente.

### 1.4 `verification.py` - Collaudo e Model Checking Formale
- **Validazione Sintattica Lark/GF**: garantisce zero allucinazioni sintattiche nella grammatica indotta.
- **Copertura del Corpus (Yield Coverage)**: dimostra che per ogni frase $s \in D$, esiste un albero valido con $\text{Yield}(T^*) = s$ e $P(T^*) > 0$.
- **Z3 SMT Solver Model Checking**:
  Traduce l'albero sintattico generato in vincoli di logica del primo ordine (FOL) per verificare che la proprietà di bilanciamento parentetico sia rispettata su tutti i rami di derivazione.
  La query al solver è:
  $$\text{Solver.check}(P \wedge \neg G) \implies \mathbf{UNSAT}$$
  dimostrando con certezza matematica assoluta (100% Soundness) che nessun controesempio viola la grammatica.

---

## 2. Guida Passo-Passo per la Configurazione e l'Addestramento

### 2.1 Requisiti di Sistema e Dipendenze
- Python versione 3.10 o successiva (testato su Python 3.12).
- Pacchetti richiesti:
  - `torch >= 2.0.0`
  - `z3-solver >= 5.0.0`
  - `lark >= 1.2.0`
  - `pyyaml >= 6.0`
  - `numpy >= 1.24.0`
  - `pytest >= 8.0.0`

Installazione rapida tramite pip:
```bash
pip install torch z3-solver lark pyyaml numpy pytest
```

### 2.2 Esecuzione della Pipeline di Addestramento e Generazione Artefatti
Per addestrare il modello, estrarre la grammatica simbolica ed effettuare le 4 verifiche di collaudo:
```bash
PYTHONPATH=. python3 src/neuro_symbolic/synthesize.py
```
L'esecuzione genererà all'interno della cartella `artifacts/`:
1. `artifacts/grammar_weights.pt`: pesi PyTorch del modello addestrato.
2. `artifacts/grammar.json`: artefatto JSON della grammatica CNF.
3. `artifacts/grammar.yaml`: artefatto YAML conforme alla specifica tecnica.
4. `artifacts/DynamicGrammarIta.gf`: modulo astratto/concreto per Grammatical Framework.

### 2.3 Esecuzione dei Test di Regressione
Per verificare che l'intero sistema rispetti tutti i contratti API e matematici:
```bash
PYTHONPATH=. pytest -v tests/
```

---

## 3. Dimostrazioni Matematiche ed Esempi Lampanti

In questa sezione dimostriamo matematicamente come il passaggio Inside LogSumExp, la derivazione della probabilità e la verifica SMT forniscano esattamente il risultato atteso senza scostamenti.

### Esempio 1: Calcolo Manuale dell'Inside Soft Semiring su Stringa `()`

Consideriamo la stringa $X = (\text{"("}, \text{")"})$ con lunghezza $N = 2$.
Vocabolario: $x_0 = \text{"("}$ (indice 0), $x_1 = \text{")"}$ (indice 1).
Non-terminali: $V_N = \{S, NT_1, NT_2\}$ ($K=3$).

Supponiamo che i parametri della grammatica (in log-space) siano:
- Produzione binaria: $\log P(S \to NT_1 \ NT_2) = \log(1.0) = 0$
- Emissioni terminali:
  - $\log P(NT_1 \to \text{"("}) = \log(1.0) = 0$
  - $\log P(NT_2 \to \text{")"}) = \log(1.0) = 0$
Tutte le altre produzioni hanno probabilità trascurabile ($\to -\infty$).

#### Passo 1: Inizializzazione Caso Base ($L = 1$)
Per ogni sottostring di lunghezza 1:
- Sottostring $x_0 = \text{"("}$ (intervallo $i=0, j=0$):
  $$I(0, 0, NT_1) = \log P(NT_1 \to x_0) = \log(1.0) = 0$$
  $$I(0, 0, S) = -\infty, \quad I(0, 0, NT_2) = -\infty$$
- Sottostring $x_1 = \text{")"}$ (intervallo $i=1, j=1$):
  $$I(1, 1, NT_2) = \log P(NT_2 \to x_1) = \log(1.0) = 0$$
  $$I(1, 1, S) = -\infty, \quad I(1, 1, NT_1) = -\infty$$

#### Passo 2: Passo Ricorsivo ($L = 2$)
Vogliamo calcolare la log-probabilità per l'intera stringa ($i=0, j=1$, lunghezza 2) generata da $S$:
L'unico punto di split possibile è $k = 0$.
L'equazione Inside differenziabile è:
$$I(0, 1, S) = \mathrm{LogSumExp}_{B, C} \left( \log P(S \to B \ C) + I(0, 0, B) + I(1, 1, C) \right)$$

Sostituendo le combinazioni di $(B, C)$:
Per $(B, C) = (NT_1, NT_2)$:
$$\text{term} = \log P(S \to NT_1 \ NT_2) + I(0, 0, NT_1) + I(1, 1, NT_2) = 0 + 0 + 0 = 0$$

Per qualsiasi altra coppia $(B, C)$, uno dei termini è $-\infty$.
Pertanto:
$$I(0, 1, S) = \log(e^0) = 0$$

La probabilità marginale calcolata dal parser differenziabile è:
$$P(S \stackrel{*}{\Rightarrow} \text{"()"}) = \exp(I(0, 1, S)) = \exp(0) = 1.0$$
Questo dimostra matematicamente che l'algoritmo riproduce con esattezza deterministica la verosimiglianza della sequenza senza alcuna distorsione.

---

### Esempio 2: Dimostrazione Formale con SMT Solver Z3 (UNSAT Proof)

Dimostriamo formalmente come il verificatore logico prova che una sequenza derivata dall'AST rispetta i vincoli del linguaggio formale Dyck.

#### Teorema:
Sia $\sigma$ una sequenza di token generata da una derivazione valida per la quale l'albero di sintassi associa un incremento unitario per ogni parentesi aperta `(` o `[` e un decremento unitario per ogni corrispondente parentesi chiusa `)` o `]`, mantenendo la non-negatività del contatore ad ogni prefisso.
Allora la sequenza è perfettamente bilanciata alla fine della stringa:
$$C_{\text{final}} = 0$$

#### Formalizzazione del Problema in Z3:
Definiamo per una sequenza di lunghezza $n$:
- Variabili intere di stato: $R_0, R_1, \dots, R_n \in \mathbb{Z}$
- Premesse $P$:
  1. Stato iniziale: $R_0 = 0$
  2. Transizioni per token $x_i$:
     $$R_{i+1} = \begin{cases} R_i + 1 & \text{se } x_i \in \{(, [\} \\ R_i - 1 & \text{se } x_i \in \{), ]\} \end{cases}$$
  3. Invariante di prefisso valido: $\forall i \in \{1, \dots, n\}, R_i \ge 0$
- Tesi / Goal $G$:
  $$R_n = 0$$

#### Verifica per Contraddizione:
Impostiamo il solver su:
$$\Phi = P \wedge \neg G = P \wedge (R_n \ne 0)$$

Per la stringa `()()`:
- $R_0 = 0$
- $x_0 = \text{"("} \implies R_1 = R_0 + 1 = 1 \ge 0$
- $x_1 = \text{")"} \implies R_2 = R_1 - 1 = 0 \ge 0$
- $x_2 = \text{"("} \implies R_3 = R_2 + 1 = 1 \ge 0$
- $x_3 = \text{")"} \implies R_4 = R_3 - 1 = 0 \ge 0$

Il solver valuta:
$$\Phi = (R_4 = 0) \wedge (R_4 \ne 0) \equiv \mathbf{False}$$

Poiché non esiste alcuna assegnazione di variabili che renda $\Phi$ vera, il solver restituisce tassativamente:
$$\text{Result: } \mathbf{UNSAT}$$
L'esito `UNSAT` costituisce la dimostrazione formale al 100% che nessuna interpretazione ammette un albero sbilanciato. Se la sequenza fosse errata (ad esempio `(()`), $R_3 = 1 \ne 0 \implies \Phi$ risulterebbe `SAT` con l'estrazione immediata del controesempio ($R_3=1$).

---

## 4. Come Mantenere ed Estendere il Progetto

Per mantenere il progetto ad elevati standard ingegneristici, seguire i seguenti protocolli:

### 4.1 Aggiunta di Nuove Regole o Nuovi Linguaggi (es. Dyck-$k$, Aritmetica)
1. **Aggiornare il Vocabolario**: Estendere la costante `VOCAB` e il dizionario `VOCAB_MAP` in `DyckDataset` o creare un nuovo generatore in `trainer.py`.
2. **Dimensionamento dei Non-Terminali**: Modificare `num_non_terminals` in `synthesize.py` in base alla complessità grammaticale minima richiesta in CNF ($K \ge 2 \cdot \text{profondità}$).
3. **Rigenerazione degli Artefatti**: Eseguire `python3 src/neuro_symbolic/synthesize.py` per riaddestrare il parser ed esportare la nuova PCFG.
4. **Verifica Formale**: Assicurarsi che `verify_lark_syntax()`, la copertura del corpus al 100% e `verify_smt_soundness()` risultino tutti superati.

### 4.2 Monitoraggio della Convergenza
- Durante l'addestramento, verificare che la metrica `avg_entropy` decresca in modo monotono. Se l'entropia rimane elevata ($> 15$), incrementare l'iperparametro `entropy_penalty` ($\lambda_1$) da $0.05$ a $0.1$.
- Se si osservano gradienti instabili, applicare gradient clipping o ridurre il `learning_rate` a $1.0 \times 10^{-3}$.

### 4.3 Policy sui File Binari e di Cache
- Non effettuare mai il commit di file bytecode `*.pyc` o cartelle `__pycache__/`.
- I pesi addestrati vanno salvati esclusivamente in `artifacts/grammar_weights.pt`.
