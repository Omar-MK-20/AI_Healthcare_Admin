# Architecture — AI Healthcare Administration Assistant

## Pipeline

```
User input
   │
   ▼
GUARDRAILS (emergency redirect, no-diagnosis, PII mask)   
   │
   ▼
ORCHESTRATOR (router — every route calls the LLM; offline mode disabled)
   ├── Booking ──► LLM extracts JSON → fuzzy patient match → validators/policy
   │                → Agent 5 Insurance (patients.csv) + Agent 4 Billing (billing.csv)
   │                → collision check → book → email + report (outbox/)
   ├── Policy Q&A ──► Agent 1 RAG (PDF → FAISS/MiniLM → context) ──► LLM answers
   ├── Summarize ──► Agent 3 LLM (3-5 bullets + disclaimer + mask_pii)
   ├── Billing/Insurance ──► LLM extracts service+patient → CSV lookups → LLM composes reply
   └── Registration ──► LLM extracts fields → validated CSV append
   │
   ▼
MEMORY (deque maxlen=6) + EVAL LOG (task, ok, latency) → Dashboard
```

## Models & optimization 

- LLM: `Qwen/Qwen3-1.7B` (thinking disabled, strict-JSON mode) —
  4-bit BitsAndBytes (`bnb_config`) on GPU, full weights on CPU (`device_map="cpu"`),
  so it runs with or without a GPU.
  `float16` compute — same as `Quantization.ipynb`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, FAISS retriever.
- Deployment: Gradio `share=True` OR FastAPI + Ngrok
  (`backend_api.py`, Bearer `secret123`, like `parser.py` backend).

## Data sources (Objective 1)

| Source | File | Agent |
|--------|------|-------|
| Hospital policy PDF | `data/hospital_policy.pdf` (built from `.txt`) | Agent 1 |
| Patient DB (→ Google Sheets in production) | `data/patients.csv` | Agent 5 |
| Price list (→ Sheets/DB) | `data/billing.csv` | Agent 4 |
| Appointments DB | `data/appointments_seed.csv` + in-memory list | Agent 2 |
| Discharge sample | `data/sample_discharge.txt` | Agent 3 |

Swap CSV reads for `gspread` / SQL / REST without changing agent interfaces.

## Guardrails 

1. Emergency keywords → refuse + redirect to ER.
2. Diagnosis/prescription/dosage patterns → refuse + offer admin help.
3. `valid_date` (future, YYYY-MM-DD), `valid_time` (HH:MM).
4. `validate_policy` (dept hours, Sunday closed).
5. Collision check per doctor+slot.
6. `mask_pii` on every output; insurance ID regex `^INS-\d+$`.

## Evaluation (Evaluation & Debugging)

- `tests/test_evaluation.py` (fast, no model): validators, policy,
  billing, insurance, guardrails, PII, JSON, seed collision.
- Runtime `eval_log`: success rate, avg latency in Dashboard tab.
