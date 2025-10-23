# Exam Corrector - AI-Powered Open Answer Assessment System

An intelligent system for automatically evaluating open-ended exam questions using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## Overview

This system uses AI to:
- Generate assessment checklists from course materials
- Evaluate student answers against predefined criteria
- Provide constructive feedback
- Calculate scores based on feature satisfaction

**Key Technologies:**
- **LangChain**: Orchestration framework
- **Groq API**: Fast LLM inference (Llama 3.3)
- **BGE Embeddings**: State-of-the-art semantic search
- **RAG**: Course material retrieval for context-aware assessment

---

## Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended for large embeddings)
- **Disk Space**: ~2GB for models and data

---

##  Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/exam_corrector.git
cd exam_corrector
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected time:** 2-3 minutes

### 4. Get Groq API Key (Free)

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up (free tier: 30 requests/min)
3. Create an API key
4. Copy the key (starts with `gsk_...`)

### 5. Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "gsk_your_key_here"
```

**macOS/Linux:**
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

**Permanent setup (recommended):**
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=gsk_your_key_here
```

### 6. Build RAG Knowledge Base

```bash
python -m exam.rag --fill --model bge-large
```

**What this does:**
- Downloads BGE-Large embedding model (~1GB, one-time)
- Processes course slides from `content/` directory
- Creates vector database for semantic search
- **Time:** 5-10 minutes (first run)


### 7. Generate Assessment Checklists

```bash
python -m exam.solution
```

**What this does:**
- Generates checklists (SHOULD/SHOULDN'T items) for each question
- Uses RAG to find relevant course content
- Saves results to `solutions/` directory
- **Time:** ~30 seconds per question

### 8. Run MCP client

```bash
python mcp_client.py
```

**What this does:**
- Evaluates a full exam



---

## 📁 Project Structure

```
exam_corrector/
├── exam/                      # Main package
│   ├── __init__.py           # Question/Answer data models
│   ├── assess/               # Assessment engine
│   │   ├── __init__.py       # Assessor class
│   │   └── prompt-template.txt
│   ├── solution/             # Checklist generation
│   │   ├── __init__.py       # SolutionProvider class
│   │   └── prompt-template.txt
│   ├── rag/                  # Retrieval-Augmented Generation
│   │   ├── __init__.py       # Vector store management
│   │   └── __main__.py       # CLI for RAG operations
│   ├── openai/               # LLM client (Groq)
│   │   └── __init__.py
│   └── mcp/                  # Model Context Protocol (optional)
│       └── __init__.py
├── static/
│   └── questions.csv         # Question bank
├── content/                  # Course materials (Markdown slides)
├── solutions/                # Generated checklists (YAML)
├── mock_exam_submissions/    # Example student answers
├── slides-rag.db            # Vector database (generated)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

