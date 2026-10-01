# Architecture — AI Healthcare Administration Assistant

## Pipeline

```
User input
   │
   ▼
GUARDRAILS (emergency redirect, no-diagnosis, PII mask)   [Session 10]
   │
   ▼
ORCHESTRATOR (router)
   ├── Booking ──► deterministic NL parser (dept/doctor/date/time/patient, no LLM needed)
   │                → fuzzy patient match → policy/billing/booking → natural reply
   │                → falls back to LLM JSON only when details are missing
   ├── Policy Q&A ──► Agent 1 RAG (PDF → FAISS/MiniLM → context) ──► answer + citation
   ├── Booking ──► Agent 1 context → LLM JSON → valid_date/time → validate_policy
   │                → Agent 5 Insurance (patients.csv) ┐ parallel
   │                → Agent 4 Billing (billing.csv)    ┘
   │                → collision check → book → email + report (outbox/)
   ├── Summarize ──► Agent 3 (3-5 bullets + disclaimer + mask_pii)
   └── Billing/Insurance only ──► Agents 4+5 direct lookup
   │
   ▼
MEMORY (deque maxlen=6) + EVAL LOG (task, ok, latency) → Dashboard
```

## Models & optimization (Session 10)

- LLM: `Qwen/Qwen3-1.7B` (thinking disabled, strict-JSON mode) —
  4-bit BitsAndBytes (`bnb_config`) on GPU, full weights on CPU (`device_map="cpu"`),
  so it runs with or without a GPU.
  `float16` compute — same as `Quantization.ipynb`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, FAISS retriever.
- Deployment: Gradio `share=True` (Session 9) OR FastAPI + Ngrok
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

## Guardrails (Session 10)

1. Emergency keywords → refuse + redirect to ER.
2. Diagnosis/prescription/dosage patterns → refuse + offer admin help.
3. `valid_date` (future, YYYY-MM-DD), `valid_time` (HH:MM).
4. `validate_policy` (dept hours, Sunday closed).
5. Collision check per doctor+slot.
6. `mask_pii` on every output; insurance ID regex `^INS-\d+$`.

## Evaluation (Session 9: Evaluation & Debugging)

- `tests/test_evaluation.py` (11 tests, offline-safe): validators, policy,
  billing, insurance, guardrails, PII, JSON, seed collision.
- Runtime `eval_log`: success rate, avg latency in Dashboard tab.
