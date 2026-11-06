# Istruzioni Obbligatorie per Agenti IA (AGENTS.md)

Questo file definisce le direttive tassative e i vincoli operativi per qualsiasi Agente IA o assistente automatico che operi su questo repository.

---

## 1. Vincolo di Aggiornamento Obbligatorio della Documentazione

Ad **ogni modifica strutturale, architetturale, algoritmica o di interfaccia**, l'Agente IA **DEVE**:
1. Aggiornare tempestivamente `doc/TECHNICAL_DOCUMENTATION.md` riflettendo i cambiamenti apportati.
2. Aggiornare `README.md` se la modifica impatta l'architettura, la matrice degli iperparametri, le metriche, le procedure di collaudo o le istruzioni di esecuzione.
3. Documentare le modifiche matematiche o le nuove equazioni/invarianti introdotte.

> **Regola Aurea**: Nessuna Pull Request o Commit che alteri codice sorgente in `src/` o artefatti in `artifacts/` può essere considerata completa senza il corrispondente aggiornamento della documentazione tecnica in `doc/`.

---

## 2. Standard di Igiene del Repository
- **Zero file binari di cache**: Non aggiungere o committare mai file `*.pyc` o cartelle `__pycache__/`.
- Tutte le nuove dipendenze devono essere documentate in `README.md` e in `doc/TECHNICAL_DOCUMENTATION.md`.
- Mantenere e rispettare sempre `.gitignore`.

---

## 3. Protocollo di Verifica e Collaudo Obbligatorio

Prima di finalizzare qualsiasi modifica o sottomissione, l'agente IA **DEVE** eseguire i seguenti controlli:

1. **Esecuzione dei Test Unitari e di Integrazione**:
   ```bash
   PYTHONPATH=. pytest -v tests/
   ```
   Tutti i test devono passare con esito `PASSED` (0 errori, 0 fallimenti).

2. **Verifica della Pipeline di Sintesi ed Esportazione**:
   Se sono stati modificati il parser, il trainer o la grammatica, rieseguire la sintesi e verificare la rigenerazione coerente degli artefatti:
   ```bash
   PYTHONPATH=. python3 src/neuro_symbolic/synthesize.py
   ```
   Verificare che:
   - `artifacts/grammar.json` sia valido e parsabile.
   - `artifacts/grammar.yaml` sia coerente.
   - `artifacts/DynamicGrammarIta.gf` sia generato correttamente.
   - La copertura del corpus risulti 100%.
   - La verifica formale Z3 SMT risulti **UNSAT** (Soundness 100%).
