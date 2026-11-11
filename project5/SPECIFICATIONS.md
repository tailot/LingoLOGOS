# Specifiche Tecniche per la Generazione e Validazione dei Dataset Neuro-Simbolici (LingoLOGOS) - Full Pre-training Scale

Architettura di sintesi, quantificazione e validazione deterministica per dataset di addestramento e calibrazione neuro-simbolica calibrata esclusivamente per la scala di Full Pre-training.

---

## 1. Architettura Generale e Pipeline Deterministica

Le presenti specifiche definiscono l'infrastruttura di sintesi, quantificazione e validazione per 5 dataset ($\mathcal{D}_1 \dots \mathcal{D}_5$) calibrata esclusivamente per il Full Pre-training Scale con cardinalita $N = 1.000.000$ campioni per ciascun dataset (per un totale complessivo di 5.000.000 di campioni). L'obiettivo primario e la calibrazione e il pre-training dei modelli di linguaggio accoppiati a vincoli deterministici realizzati tramite Grammatical Framework (GF), soddisfacibilita di teorie logiche del primo ordine con Z3 (`ontology.smt2`) e mascheramento dinamico del vocabolario tramite `VolatileLogitsMasker`.

### Schema Architetturale della Pipeline

| Fase Pipeline | Componente | Input | Operazione / Output |
| :--- | :--- | :--- | :--- |
| **1. Sintesi Astratta** | Grammatical Framework (GF) | Grammatiche RGL | Generazione alberi AST ($\mathcal{T}_{\text{GF}}$) e stringhe NL |
| **2. Validazione Logica** | Z3 Solver (`ontology.smt2`) | Predicati Logici / SMT-LIB2 | Verifica consistenza SAT/UNSAT e vincoli |
| **3. Mascheramento** | `VolatileLogitsMasker` | Stato Automa (GF + Z3) | Calcolo bitset per mascheramento dinamico logit |
| **4. Export & Storage** | Pipeline Engine | Campioni Validati | Format Parquet / JSONL.zst deterministico (`--seed 42`) |

---

## 2. Definizione dei 5 Dataset ed Extracalibrazione Tecnica

### 2.1. $\mathcal{D}_1$: Derivazione Grammaticale e AST Mappings (GF RGL)

* **Target:** Istruire la testa del modello sulla proiezione univoca da stringhe in linguaggio naturale a nodi AST (Abstract Syntax Tree) validati dalla Resource Grammar Library (RGL) di Grammatical Framework.
* **Formato di Input/Output:** Schema JSON contenente la sequenza di token in linguaggio naturale $w_{1:T}$ e la serializzazione lineare dell'albero di sintassi astratta $\mathcal{T}_{\text{GF}}$.
* **Specifiche Esecutive & Parametriche ($N$):**
  * Equidistribuzione multilingue su 5 lingue target (`it`, `en`, `fr`, `de`, `es`).
  * Parsing ricorsivo con limite superiore di profondita astratta $d \le 6$ per prevenire ricorsioni infinite e comprimere la complessita combinatoria.
  * Preservazione delle riduzioni morfosintattiche, flessioni verbali complesse e dipendenze a lungo raggio.

### 2.2. $\mathcal{D}_2$: Consistenza Ontologica e Vincoli SMT (Z3 Integration)

* **Target:** Validare la capacita di ragionamento formale rispetto a predicati della logica del primo ordine e teorie SMT-LIB2 gestite dal solver Z3 (`ontology.smt2`).
* **Formato di Input/Output:** Tripla formale `(Query_Semantica, SMT_Assertions_Block, Expected_Satisfiability_Result)`.
* **Specifiche Esecutive & Parametriche ($N$):**
  * Bilanciamento rigoroso 50/50 tra istanze *SAT* (satisfiable) e *UNSAT* (unsatisfiable).
  * Iniezione di assiomi ontologici presi dal payload d'inizializzazione (`seed_payload.json`), inclusi vincoli temporali (intervalli di Allen), relazioni d'ordine parziale e disuguaglianze aritmetiche.
  * Stress-testing del solver tramite quantificazione di domini numerici finiti e bound sulle variabili di stato.

### 2.3. $\mathcal{D}_3$: Dataset Multidominio Orientato all'Esecuzione di Comandi (SFT & LLLayer)

* **Target:** Adattamento del modello alle chiamate strutturate per la piattaforma multi-agente (`agente_alfa`, `agente_beta`, `arbitro_logico`) e instradamento dei comandi tramite l'interfaccia CLI/orchestrator `LLLayer`.
* **Formato di Input/Output:** Prompts conversazionali multi-turn accoppiati al payload JSON strutturato ed esecutibile dal layer di sistema.
* **Specifiche Esecutive & Parametriche ($N$):**
  * Partizionamento uniforme del dataset sui 10 domini applicativi verticali (100.000 campioni per ciascun settore per un totale di $N = 1.000.000$: Agenda e Scheduling Temporale, Domotica e Smart Home Control, Finanza Operativa e Transattiva, Produttivita e Task Management, Salute e Parametri Biometrici, Logistica e Fleet/Supply Chain Management, Gestione Middleware e System Telemetry, Cybersecurity e IAM, E-Commerce e Inventory Routing, Customer Support e Ticket Resolution).
  * Inclusione di intenti ad alta arita (query multi-parametro con vincoli incrociati di budget, scheduling e dipendenze sequenziali).
  * Arricchimento dei token con metadati di contesto volatile iniettati al tempo di inferenza.

### 2.4. $\mathcal{D}_4$: Robustezza Sintattica e Controlli Negativi (Dyck-2 & Anti-Allucinazione)

* **Target:** Prevenire allucinazioni semantiche e forzare il rispetto rigido di strutture nidificate bilanciate tramite formalismi di tipo Dyck-2.
* **Formato di Input/Output:** Tripla `(Corrupted_Sequence, Error_Type_Tag, Corrected_Sequence)`.
* **Specifiche Esecutive & Parametriche ($N$):**
  * Iniezione controllata di rumore sintattico: token fuori vocabolario (OOV), sbilanciamento di delimitatori, violazioni di arita predicativa e incongruenze di tipo.
  * Inclusione di sequenze di controllo Dyck-2 con profondita di annidamento arbitraria per verificare le capacita di conteggio e tracciamento dello stack interno del modello ($N/2$ campioni dedicati).

### 2.5. $\mathcal{D}_5$: Proiezione e Mascheramento dei Logit (`VolatileLogitsMasker`)

* **Target:** Fornire la supervisione necessaria al modulo `VolatileLogitsMasker` per l'applicazione di maschere booleane passo-passo sui logit del vocabolario durante il decoding.
* **Formato di Input/Output:** Tripla `(Token_Prefix_Sequence, Active_Mask_Bitset, Next_Valid_Token_Set)`.
* **Specifiche Esecutive & Parametriche ($N$):**
  * Generazione dei vettori di bitset espliciti determinati dallo stato corrente dell'automa a stati finiti GF + Z3.
  * Mappatura bidirezionale fra gli ID dei token del vocabolario LLM e i simboli terminali validi della grammatica formale.

---

## 3. Matrice Comparativa dei Dataset

| Dataset | Focus Primario | Core Engine | Formato Output Key | Distribuzione / Vincoli Key |
| :--- | :--- | :--- | :--- | :--- |
| **$\mathcal{D}_1$** | Mapping NL-to-AST | GF RGL | `(w_{1:T}, T_GF)` | 5 lingue (it, en, fr, de, es), $d \le 6$ |
| **$\mathcal{D}_2$** | Consistenza Logica | Z3 Solver | `(Query, SMT_Block, SAT_Res)` | 50% SAT / 50% UNSAT, Intervalli Allen |
| **$\mathcal{D}_3$** | Execution & Multi-Agent | LLLayer / CLI | `(Prompt, Execution_JSON)` | 10 Domini ($N/10$), Intenti Alta Arita |
| **$\mathcal{D}_4$** | Robustezza & Anti-Noise | Dyck-2 Checker | `(Corrupted, Error, Corrected)` | $N/2$ Dyck-2, Iniezione OOV / Arity Error |
| **$\mathcal{D}_5$** | Dynamic Logit Masking | VolatileMasker | `(Prefix, Bitset, Valid_Tokens)` | Mapping State Automata -> Vocabulary IDs |

---

## 4. Requisiti di Esecuzione, Pipeline di Sintesi e Storage

* **Riproducibilita Deterministica e Scala:** L'architettura dei dati e calibrata esclusivamente per la scala di Full Pre-training (5.000.000 di campioni totali). Obbligo di fissare il seed pseudo-casuale per la pipeline di generazione (es. `--seed 42`).
* **Verifica a Due Fasi (Pre-Export Validation):** Nessun campione viene scritto su disco senza il superamento simultaneo del parsing GF e della verifica di consistenza SMT via Z3.
* **Serializzazione Dati e Volumi Target:** Output generato in formato Apache Parquet o JSON Lines (`.jsonl`) compresso in Zstandard. Il dataset completo compresso ammonta a circa 5,0 - 8,0 GB, corrispondenti a 30 - 45 GB di testo/JSON grezzo per un caricamento ad alte prestazioni nei framework di addestramento.
