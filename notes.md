# Guida alla Configurazione del Sistema di Correzione Automatica

## Modifiche Principali

Il sistema è stato aggiornato per utilizzare:
- **Mistral-7B-Instruct** tramite OpenRouter invece di OpenAI
- **HuggingFace Embeddings** (sentence-transformers) invece di OpenAI Embeddings

## Installazione

### 1. Installare le dipendenze

```bash
pip install -r requirements.txt
```

**Nota Importante**: 
- L'installazione di `torch` e `sentence-transformers` può richiedere alcuni minuti
- Il primo download del modello embeddings richiederà ~100-500MB
- Assicurati di avere almeno 2GB di spazio libero su disco

### 2. Verificare l'installazione

```bash
python test_setup.py
```

Questo script verificherà che tutto sia installato correttamente.

### 3. Configurare OpenRouter API Key

Ottieni una chiave API gratuita da [OpenRouter](https://openrouter.ai/keys):

**Su Linux/Mac:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

**Su Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

**Su Windows (CMD):**
```cmd
set OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Oppure crea un file `.env` nella root del progetto:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Il sistema ti chiederà la chiave all'avvio se non è configurata.

## Utilizzo

### 1. Creare il Vector Store (RAG)

Prima di tutto, popola il database vettoriale con le slide del corso:

```bash
python -m exam.rag --fill
```

Questo processo:
- Legge tutti i file `_index.md` dalla cartella `content/`
- Divide il contenuto in slide
- Genera embeddings usando sentence-transformers
- Salva tutto in `slides-rag.db`

**Tempo stimato**: 5-15 minuti (dipende dal numero di slide)

### 2. Testare il RAG

Verifica che il RAG funzioni correttamente:

```bash
python -m exam.rag
```

Ti permetterà di fare query interattive contro il vector store.

### 3. Generare le Checklist per le Domande

Genera le checklist di valutazione per tutte le domande:

```bash
python -m exam.solution
```

Oppure per domande specifiche:

```bash
python -m exam.solution SE-1 SE-2
```

Le checklist vengono salvate in `solutions/` come file YAML.

### 4. Valutare le Risposte degli Studenti

Struttura attesa per le risposte:

```
exam_submissions/
├── Q01 - SE-1/
│   ├── 12345 - Mario Rossi/
│   │   └── Attempt_1_textresponse
│   └── 67890 - Luigi Bianchi/
│       └── Attempt_1_textresponse
└── Q02 - SE-2/
    └── ...
```

Esegui la valutazione:

```bash
python -m exam.assess /path/to/exam_submissions
```

Oppure salvando l'output in un file:

```bash
OUTPUT_FILE=assessment_results.txt python -m exam.assess /path/to/exam_submissions
```

## Modelli Disponibili

### LLM (via OpenRouter)

Il sistema usa di default `mistralai/mistral-7b-instruct`, ma puoi specificare altri modelli:

```python
from exam.openai import AIOracle

# Mistral 7B (default, gratuito)
oracle = AIOracle("mistralai/mistral-7b-instruct")

# Altri modelli Mistral
oracle = AIOracle("mistralai/mixtral-8x7b-instruct")

# Modelli alternativi
oracle = AIOracle("meta-llama/llama-3-8b-instruct")
oracle = AIOracle("google/gemini-pro")
```

### Embeddings (HuggingFace)

Tre opzioni preconfigurate:

```python
from exam.rag import sqlite_vector_store

# Small - veloce, leggero (default)
store = sqlite_vector_store(model="small")

# Large - più accurato ma più lento
store = sqlite_vector_store(model="large")

# Multilingual - supporto multilingua (EN + IT)
store = sqlite_vector_store(model="multilingual")
```

## Workflow Completo

```bash
# 1. Setup iniziale (una volta sola)
export OPENROUTER_API_KEY="your-key"
pip install -r requirements.txt

# 2. Popola il RAG con le slide
python -m exam.rag --fill

# 3. Genera le checklist delle domande
python -m exam.solution

# 4. Valuta le risposte degli studenti
python -m exam.assess ./exam_submissions

# 5. (Opzionale) Genera varianti del test
python -m exam.test -w 9 -c 1 2 3
```

## Struttura dei File Generati

### Checklist (solutions/*.yaml)

```yaml
id: SE-1
question: "Descrivi il pattern MVC..."
model_name: mistralai/mistral-7b-instruct
should:
  - "Menziona Model, View, Controller"
  - "Spiega la separazione delle responsabilità"
should_not:
  - "Confonde MVC con MVVM"
examples:
  - "Framework come Django o Spring MVC"
see_also:
  - "Vantaggi della separazione UI/logica"
```

### Valutazioni (cache in submission folders)

```yaml
feature: "Menziona Model, View, Controller"
feature_type: SHOULD
satisfied: true
motivation: "Hai correttamente identificato i tre componenti principali del pattern MVC."
```

## Troubleshooting

### Errore: "Module not found"

```bash
pip install -r requirements.txt --upgrade
```

### Embeddings troppo lenti

Usa il modello "small" o abilita GPU:

```python
# In exam/rag/__init__.py, modifica:
model_kwargs={'device': 'cuda'}  # invece di 'cpu'
```

### OpenRouter rate limiting

OpenRouter ha limiti gratuiti. Se necessario, considera:
- Aumentare il delay tra richieste
- Usare un piano a pagamento
- Implementare caching più aggressivo

### Il RAG non trova contenuti rilevanti

Verifica che:
1. Le slide siano in formato corretto
2. Il delimitatore `---` o `+++` sia presente
3. Il contenuto sia in `content/**/_index.md`

```bash
# Debug: stampa le slide trovate
python -c "from exam.rag import all_slides; print(len(list(all_slides())))"
```

## Performance

### Tempi Stimati

- **RAG population**: 5-15 min (dipende da # slide)
- **Generazione checklist**: 30-60s per domanda
- **Valutazione risposta**: 10-30s per feature

### Ottimizzazioni

1. **Cache**: Tutte le operazioni LLM usano cache YAML
2. **Batch processing**: Le valutazioni sono elaborate sequenzialmente
3. **Embeddings locali**: Nessun costo API per il RAG

## Sicurezza

- Non committare mai `OPENROUTER_API_KEY` nel repository
- Usa `.env` file o variabili d'ambiente
- Le chiavi API sono richieste interattivamente se mancanti

## Limitazioni Attuali

1. **No browser storage**: Il sistema non usa localStorage/sessionStorage
2. **Single-threaded**: Le valutazioni sono sequenziali
3. **No GPU optimization**: Di default usa CPU per embeddings
4. **Italiano**: Ottimizzato per contenuti in italiano/inglese

## Prossimi Sviluppi (Agentic AI)

Per implementare il sistema agentico completo:

1. **Agent 1 - Checklist Generator**
   - Input: Domanda + RAG context
   - Output: Checklist strutturata
   - Tool: RAG search

2. **Agent 2 - Answer Assessor**
   - Input: Risposta studente + Checklist
   - Output: Valutazione + Feedback
   - Tool: Feature verification

3. **Tools Aggiuntivi**
   - Database reader (Excel/CSV)
   - Feedback generator
   - Score calculator
   - MCP gateway per integrazioni esterne

4. **Orchestration**
   - LangGraph per workflow multi-agent
   - State management
   - Error handling e retry logic