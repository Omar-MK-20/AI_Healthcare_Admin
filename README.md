# 🏥 AI Healthcare Administration Assistant — Final Project

Multi-agent AI system (Week 6 Final Project). Fulfills all 6 objectives.

## 1. Quick start (YOU run these yourself)

```bash
cd AI_Healthcare_Admin
pip install -r requirements.txt
python create_sample_data.py   # builds data/hospital_policy.pdf
python app.py                  # Gradio UI (share link for demo)
```

Optional paths:
```bash
python backend_api.py          # FastAPI backend (then expose with ngrok)
streamlit run streamlit_app.py # Streamlit frontend like parser.py
python -m pytest tests/ -v     # Evaluation tests
```

Upload `healthcare_assistant.ipynb` to Colab + Kaggle for the GPU run
(Qwen3-1.7B runs 4-bit on GPU, full weights on CPU).

## 2. Objectives mapping

| # | Objective | Where |
|---|-----------|-------|
| 1 | Data Retrieval | `hospital_policy.pdf` via PyPDFLoader + FAISS RAG (Agent 1); `patients.csv`, `billing.csv`, `appointments_seed.csv` via pandas (Agents 4–5, stands in for Google Sheets/DB/API) |
| 2 | Processing & Summarization | Qwen3-1.7B-Instruct, thinking disabled (`generate()`), Summarizer Agent (Agent 3) |
| 3 | Automated Actions | `send_confirmation_email()` → `outbox/`, `generate_visit_report()`, Dashboard |
| 4 | Workflow Orchestration | `orchestrator()` routes RAG → Scheduler → Billing+Insurance → Email/Report (sequential + parallel lookups) |
| 5 | Presentation | `ARCHITECTURE.md` + `PRESENTATION_SCRIPT.md` (face-cam video guide) |
| 6 | Testing & Evaluation | `tests/test_evaluation.py`, `eval_log` + Dashboard tab, guardrails |

## 3. The 5 agents

1. **Policy RAG Agent** — answers from `hospital_policy.pdf` (FAISS + MiniLM).
2. **Appointment Scheduler Agent** — LLM → JSON → validators → collision check → books.
3. **Report Summarizer Agent** — 3–5 bullets, copies meds exactly, adds disclaimer.
4. **Billing Agent** — CSV price lookup + discount logic.
5. **Insurance Claim Agent** — CSV patient lookup + INS-XXXX validation.
+ **Orchestrator** — guardrails → routing → pipeline → email/report → eval log.
- Patient registration: ➕ Add Patient tab (or chat "Add patient …") appends to
  `patients.csv` with validation (email format, unique email/insurance ID,
  auto `P-XXXX` ID) — new patients are immediately usable for booking.

## 4. references used

- (`building web interface + RAG (improved).ipynb`): `load_pdf`, `build_retriever`,
  `generate`, `extract_json`, `valid_date/time`, `validate_policy`, `memory=deque`,
  `chat/clear_chat`, Gradio Blocks UI.
- `Projects/parser.py`: Streamlit frontend + FastAPI + Ngrok + Bearer key pattern
  → `streamlit_app.py` + `backend_api.py` + `ngrok_client_test.py`.
- `Projects/Summarizer.py`: chunk + summarize pattern → Agent 3.
- `Quantization.ipynb`: BitsAndBytes 4-bit loading → `bnb_config`.
- PPTs (`Guardrails & Optimization`, `Evaluation & Debugging`):
  `guardrail_check`, `mask_pii`, `eval_log`, Dashboard, `tests/`.

## 5. Demo script for video (5 min)

1. Show `ARCHITECTURE.md` diagram (30s).
2. Gradio chat: policy Q&A — "What are Cardiology hours?" (RAG cite).
3. Book: General for P-1002 (Basic, pays 0) → show email + report in `outbox/`.
4. Book: Specialist for P-1002 → blockedBilling note (Basic doesn't cover).
5. Book same slot twice → collision ❌.
6. Summarize `data/sample_discharge.txt` → bullets + disclaimer, ID masked.
7. Dashboard tab: success rate + latency. Mention 4-bit quantization + guardrails.

## 6. Files

```
AI_Healthcare_Admin/
  app.py  backend_api.py  streamlit_app.py  ngrok_client_test.py
  healthcare_assistant.ipynb  create_sample_data.py  requirements.txt
  data/  outbox/  tests/  README.md  ARCHITECTURE.md  PRESENTATION_SCRIPT.md
```
