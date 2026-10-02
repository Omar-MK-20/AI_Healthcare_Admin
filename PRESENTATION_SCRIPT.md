# 🎥 Presentation Script — Face-cam video (5–7 min)

## 0. Intro (30s, face-cam)
"Hi, I'm [name]. My final project is an AI Healthcare Administration Assistant —
a multi-agent system that books appointments, answers policy questions,
summarizes reports, and handles billing and insurance automatically."

## 1. Architecture (60s, screen: ARCHITECTURE.md)
- "Five agents plus an orchestrator, following our RAG + Gradio pattern
  and quantization, guardrails, and Ngrok deployment."
- Point at pipeline: guardrails → router → RAG/scheduler/summarizer/billing/insurance
  → email + report → evaluation dashboard.

## 2. Live demo (3–4 min, screen: Gradio app)
1. **Policy Q&A:** "What are Cardiology hours?" → answer from PDF.
2. **Booking success:** Book General for P-1002 on 2026-10-12 10:00 → ✅ + email + report.
   Open `outbox/` to prove automation (Objective 3).
3. **Policy block:** Specialist for Basic insurance → show "does NOT cover" note.
4. **Collision:** book same doctor/slot twice → ❌ already booked.
5. **Summarizer:** paste `data/sample_discharge.txt` → bullets + disclaimer, ID masked.
6. **Dashboard:** Refresh → success rate + latency (Objective 6).

## 3. Code walkthrough (60s)
- `app.py`: `bnb_config` 4-bit , `build_retriever`,
  `extract_json`, `validate_policy`, `orchestrator`, `send_confirmation_email`.
- `backend_api.py` + `streamlit_app.py`: parser.py-style deploy.

## 4. Testing & close (30s)
- "11 pytest tests cover validators, policy, billing, insurance, guardrails, PII."
- "Future work: real Google Sheets + SMTP + Arabic support."
- Thank + show repo structure. End.
