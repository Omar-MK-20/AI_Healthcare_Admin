# ============================================================
# AI Healthcare Administration Assistant
# Final Project - Multi-Agent System
# Based on: Session 9 (Gradio + RAG) + Session 10 (Quantization,
#           Guardrails, Evaluation, Deployment via Ngrok)
#
# 5 Agents + Orchestrator:
#   Agent 1 - Policy RAG Agent (PDF retrieval)
#   Agent 2 - Appointment Scheduler Agent (JSON + API + collision check)
#   Agent 3 - Medical Report Summarizer Agent
#   Agent 4 - Billing Agent (CSV / Google-Sheets-style lookup)
#   Agent 5 - Insurance Claim Agent (CSV lookup + rules)
#   Orchestrator - routes, runs pipeline, sends email/report, logs evaluation
#
# YOU run this yourself (testing is your responsibility):
#   pip install -r requirements.txt
#   python create_sample_data.py   # builds hospital_policy.pdf
#   python app.py
# ============================================================

# ---------------- Installations (run once in terminal, NOT here) ----
# pip install -q langchain langchain-community faiss-cpu pypdf
# pip install -q transformers accelerate bitsandbytes sentence-transformers gradio pandas fpdf2

# ---------------- Imports (Session 9 style) --------------------------
import torch
import json
import gradio as gr
import re
import csv
import time
from collections import deque
from datetime import datetime, date
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
OUTBOX = BASE / "outbox"
OUTBOX.mkdir(exist_ok=True)

# ---------------- Memory Initialization (Session 9 style) ------------
memory = deque(maxlen=6)

# Scheduled appointments = our "database" (also seeded from CSV)
scheduled_appointments = []
try:
    seed = pd.read_csv(DATA / "appointments_seed.csv")
    for _, r in seed.iterrows():
        scheduled_appointments.append({
            "appointment_id": r["appointment_id"],
            "patient_id": r["patient_id"],
            "date": r["date"],
            "time": r["time"],
            "doctor": r["doctor"],
            "department": r["department"],
            "status": r["status"],
        })
except Exception as e:
    print("Seed load skipped:", e)

# Evaluation log (Session 9: Evaluation & Debugging + Session 10)
eval_log = []  # list of dicts: {task, ok, latency_s, note}

DOCTORS = {
    "General": ["Dr. Ahmed Hassan"],
    "Cardiology": ["Dr. Sara Mahmoud"],
    "Dermatology": ["Dr. Omar Khaled"],
    "Orthopedics": ["Dr. Lina Adel"],
    "Lab": ["Lab Team"],
}

PRICES = {"General Checkup": 300, "Specialist Visit": 800, "Lab Test": 450}

# ---------------- Loading PDF (Session 9 style) ----------------------
def load_pdf(path):
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    print("PDF Loaded:", len(docs))
    return docs

# ---------------- Retriever + Embeddings (Session 9 style) -----------
def build_retriever(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

retriever = None  # built lazily on startup (needs hospital_policy.pdf)

def init_rag():
    global retriever
    pdf_path = DATA / "hospital_policy.pdf"
    txt_path = DATA / "hospital_policy.txt"
    if pdf_path.exists():
        docs = load_pdf(pdf_path)
    else:
        # Fallback: build docs from txt so app still runs before user
        # generates the PDF (you should run create_sample_data.py).
        from langchain.schema import Document
        docs = [Document(page_content=txt_path.read_text(encoding="utf-8"))]
        print("PDF not found, using TXT fallback. Run create_sample_data.py.")
    retriever = build_retriever(docs)
    return retriever

# ---------------- Model Loading (Session 10 Quantization) ------------
# 4-bit quantization with BitsAndBytes (same as Quantization.ipynb)
model_name = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

# ---------------- Generate Function (Session 9 style) ----------------
def generate(prompt, max_new_tokens=120):
    messages = [
        {"role": "system",
         "content": "You are a strict JSON generator. Return ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# A safe offline fallback if model is not loaded (lets UI + tests run)
def generate_safe(prompt, max_new_tokens=120):
    if model is None or tokenizer is None:
        return '{"action": "none"}'
    return generate(prompt, max_new_tokens)

# ---------------- Guardrails (Session 10: Guardrails) ----------------
BLOCKED_PATTERNS = [
    r"prescribe", r"dosage", r"how many mg", r"diagnose me",
    r"what disease do i have", r"emergency.*bleed", r"chest pain.*what should i do",
]
EMERGENCY_KEYWORDS = ["chest pain", "severe bleeding", "can't breathe", "unconscious"]

def guardrail_check(user_input):
    low = user_input.lower()
    # 1. Emergency redirect (never book, always redirect)
    if any(k in low for k in EMERGENCY_KEYWORDS):
        return False, ("🚨 This sounds like an emergency. This system books routine "
                       "appointments only. Please call emergency services or visit the ER immediately.")
    # 2. No diagnosis / prescription
    for p in BLOCKED_PATTERNS:
        if re.search(p, low):
            return False, ("⛔ I can't provide diagnosis or prescriptions/dosages. "
                           "I can book appointments, summarize reports administratively, "
                           "and check billing/insurance. Please consult a doctor.")
    return True, ""

def mask_pii(text):
    # Mask long digit strings (national IDs, phones) except last 4
    def _mask(m):
        s = m.group(0)
        return "*" * max(0, len(s) - 4) + s[-4:]
    return re.sub(r"\b\d{8,}\b", _mask, text)

# ---------------- Validators (Session 9 style) -----------------------
def valid_date(date_text):
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
        # no past dates
        if date_text < date.today().isoformat():
            return False
        return True
    except Exception:
        return False

def valid_time(time_text):
    if not re.match(r"^\d{2}:\d{2}$", time_text):
        return False
    try:
        datetime.strptime(time_text, "%H:%M")
        return True
    except Exception:
        return False

def appointment_exists(doctor, appt_date, appt_time):
    for a in scheduled_appointments:
        if a["doctor"] == doctor and a["date"] == appt_date and a["time"] == appt_time:
            return True
    return False

def validate_policy(data, context=""):
    """Healthcare policy rules (mirrors hospital_policy.txt)."""
    dept = data.get("department", "General")
    appt_date = data.get("date", "")
    appt_time = data.get("time", "")
    try:
        dt = datetime.strptime(appt_date, "%Y-%m-%d")
        weekday = dt.weekday()  # 0=Mon ... 6=Sun
    except Exception:
        return False, "Invalid date."
    if weekday == 6:
        return False, "No appointments on Sundays (policy)."
    hh = int(appt_time.split(":")[0])
    if dept == "General":
        if weekday > 4 or not (9 <= hh < 16):
            return False, "General Checkup must be Mon-Fri 09:00-16:00."
    elif dept in ("Cardiology", "Dermatology", "Orthopedics"):
        if weekday > 3 or not (10 <= hh < 15):
            return False, f"{dept} Specialist must be Mon-Thu 10:00-15:00 with referral."
    elif dept == "Lab":
        if not (8 <= hh < 18):
            return False, "Lab Test must be 08:00-18:00."
    else:
        return False, f"Unknown department: {dept}."
    return True, "OK"

# ---------------- Extract JSON Block (Session 9 style) ---------------
def extract_json(text):
    try:
        text = text.replace("```json", "").replace("```", "")
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        cleaned = text[start:end]
        print("\n===== CLEANED JSON =====")
        print(cleaned)
        return json.loads(cleaned)
    except Exception as e:
        print("\nJSON ERROR:", e)
        print("\nRAW OUTPUT:", text)
        return None

# ---------------- Agent 4: Billing Agent -----------------------------
def billing_agent(service_type, insurance_type):
    """Lookup data/billing.csv (stands in for Google Sheets / DB)."""
    try:
        df = pd.read_csv(DATA / "billing.csv")
    except Exception:
        df = None
    base = PRICES.get(service_type, 300)
    if insurance_type == "Premium" and service_type == "Specialist Visit":
        pay = int(base * 0.8)
        return {"service": service_type, "base": base, "patient_pays": pay,
                "note": "Premium 20% discount on Specialist."}
    if insurance_type == "Basic" and service_type in ("General Checkup", "Lab Test"):
        return {"service": service_type, "base": base, "patient_pays": 0,
                "note": "Basic covers in full."}
    if insurance_type == "Basic" and service_type == "Specialist Visit":
        return {"service": service_type, "base": base, "patient_pays": base,
                "note": "Basic does NOT cover Specialist. Full payment."}
    if insurance_type == "Premium":
        covered = 0 if service_type in ("General Checkup", "Lab Test") else int(base * 0.8)
        # Premium covers general/lab fully
        if service_type in ("General Checkup", "Lab Test"):
            covered = 0
        return {"service": service_type, "base": base, "patient_pays": covered,
                "note": "Premium coverage applied."}
    return {"service": service_type, "base": base, "patient_pays": base,
            "note": "No insurance. Full payment required before confirmation."}

# ---------------- Agent 5: Insurance Claim Agent ---------------------
def insurance_agent(patient_id_or_name):
    """Lookup data/patients.csv (stands in for Google Sheets / DB)."""
    try:
        df = pd.read_csv(DATA / "patients.csv", dtype=str).fillna("")
    except Exception as e:
        return {"ok": False, "reason": f"Patient DB unavailable: {e}"}
    q = str(patient_id_or_name).strip().lower()
    hit = df[(df["patient_id"].str.lower() == q) | (df["full_name"].str.lower() == q)]
    if hit.empty:
        return {"ok": False, "reason": "Patient not found. Treated as uninsured (None).",
                "insurance_type": "None"}
    row = hit.iloc[0].to_dict()
    ins = row.get("insurance_type", "None") or "None"
    ins_id = row.get("insurance_id", "")
    if ins != "None" and not re.match(r"^INS-\d+$", str(ins_id)):
        return {"ok": False, "reason": f"Invalid insurance ID '{ins_id}'. Treated as None.",
                "insurance_type": "None", "patient": row}
    return {"ok": True, "insurance_type": ins, "insurance_id": ins_id, "patient": row}

# ---------------- API Tool: schedule (Session 9 style) ---------------
def schedule_appointment_api(appt_date, appt_time, doctor, department, patient_id):
    if appointment_exists(doctor, appt_date, appt_time):
        return "❌ Time slot already booked for this doctor."
    appt_id = f"A-{len(scheduled_appointments)+1:04d}"
    data = {"appointment_id": appt_id, "patient_id": patient_id, "date": appt_date,
            "time": appt_time, "doctor": doctor, "department": department,
            "status": "confirmed"}
    scheduled_appointments.append(data)
    print("\n===== API CALLED =====")
    print(data)
    return f"✅ Appointment Booked!\n{data}"

# ---------------- Automated Action: email + report (Objective 3) -----
def send_confirmation_email(to_email, subject, body):
    """Mock email sender: writes to outbox/ + emails_log.csv (no real SMTP)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", to_email)
    path = OUTBOX / f"email_{ts}_{safe}.txt"
    path.write_text(f"To: {to_email}\nSubject: {subject}\n\n{body}", encoding="utf-8")
    log = OUTBOX / "emails_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "to", "subject", "file"])
        w.writerow([ts, to_email, subject, path.name])
    return str(path)

def generate_visit_report(appointment, billing, insurance):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTBOX / f"report_{appointment['appointment_id']}_{ts}.txt"
    body = (
        f"CITY GENERAL HOSPITAL - VISIT REPORT\n{appointment}\n\n"
        f"BILLING: {billing}\nINSURANCE: {insurance}\n\n"
        f"Disclaimer: administrative summary, not medical advice.\n"
    )
    path.write_text(mask_pii(body), encoding="utf-8")
    return str(path)

# ---------------- Memory + history (Session 9 style) -----------------
def get_memory_text():
    text = ""
    for item in memory:
        text += f"User: {item['user']}\nAssistant: {item['assistant']}\n\n"
    return text

def print_chat_history(user_input, context, data, response):
    print("\n========== RAG AGENT ==========\n")
    print("User Query:", user_input)
    print("\nRetrieved Context:", (context[:200] if context else "No context found."))
    print("\nGenerated JSON:", json.dumps(data, indent=2) if data else "No JSON generated.")
    print("\nAPI Result:", response)
    print("\n========== MEMORY ==========\n")
    if len(memory) == 0:
        print("Memory is empty.")
    else:
        print(f"Memory Size: {len(memory)}/{memory.maxlen}\n")
        for i, item in enumerate(memory, start=1):
            print(f"Conversation {i}\nUser      : {item['user']}\nAssistant : {item['assistant']}\n" + "-" * 50)

def print_scheduled_appointments():
    print("\n========== SCHEDULED APPOINTMENTS ==========\n")
    if not scheduled_appointments:
        print("No appointments scheduled.")
        return
    for i, a in enumerate(scheduled_appointments, start=1):
        print(f"Appointment {i}\n{a}\n" + "-" * 50)

# ---------------- Prompt Builder (Session 9 style) -------------------
def build_prompt(user_input, context):
    return f"""
You are an AI hospital scheduling agent.

Rules:
- Read the hospital policy carefully.
- Return ONLY valid JSON. No markdown. No explanations.

Departments: General, Cardiology, Dermatology, Orthopedics, Lab.
Doctors: Dr. Ahmed Hassan (General), Dr. Sara Mahmoud (Cardiology),
Dr. Omar Khaled (Dermatology), Dr. Lina Adel (Orthopedics), Lab Team (Lab).

Policy:
- General Checkup: Mon-Fri 09:00-16:00.
- Specialist (Cardiology/Dermatology/Orthopedics): Mon-Thu 10:00-15:00.
- Lab Test: Mon-Sat 08:00-18:00.
- Sunday: closed.

If user wants to book, ALWAYS return:
{{
  "action":"schedule",
  "date":"YYYY-MM-DD",
  "time":"HH:MM",
  "doctor":"string",
  "department":"string",
  "patient":"string (name or P-XXXX)"
}}
Only return {{"action":"none"}} for greetings / general questions.

Existing appointments:
{scheduled_appointments}

Context:
{context[:600]}

User:
{user_input}
"""

# ---------------- Main Orchestrator Agent ----------------------------
def orchestrator(user_input, patient_hint="P-1001", mode="auto"):
    t0 = time.time()
    ok, block_msg = guardrail_check(user_input)
    if not ok:
        eval_log.append({"task": "guardrail", "ok": False, "latency_s": 0, "note": "blocked"})
        memory.append({"user": user_input, "assistant": block_msg})
        return block_msg

    low = user_input.lower()
    # Route: summarizer
    if mode == "summarize" or any(k in low for k in ["summariz", "discharge", "lab report", "report:"]):
        report_text = user_input
        prompt = (f"Summarize this medical report in 3-5 short bullet points, plain language. "
                  f"Do NOT diagnose or prescribe. Copy medication exactly. "
                  f"End with: 'This is an administrative summary, not medical advice.'\n\n{report_text[:2000]}")
        raw = generate_safe(prompt, max_new_tokens=200)
        resp = mask_pii(raw if raw else "Summary unavailable (model not loaded).")
        eval_log.append({"task": "summarize", "ok": True,
                         "latency_s": round(time.time()-t0, 2), "note": "summarizer"})
        memory.append({"user": user_input, "assistant": resp})
        return resp

    # Route: billing / insurance quick lookup
    if mode in ("billing", "insurance") or any(k in low for k in ["price", "cost", "bill", "insurance", "claim", "cover"]):
        ins = insurance_agent(patient_hint)
        ins_type = ins.get("insurance_type", "None")
        svc = "Specialist Visit" if any(k in low for k in ["specialist", "cardio", "derma", "ortho"]) else (
            "Lab Test" if "lab" in low else "General Checkup")
        bill = billing_agent(svc, ins_type)
        resp = mask_pii(f"💳 Billing: {bill}\n🛡 Insurance: {ins}")
        eval_log.append({"task": mode, "ok": True,
                         "latency_s": round(time.time()-t0, 2), "note": svc})
        memory.append({"user": user_input, "assistant": resp})
        return resp

    # Default route: Agent 1 (RAG) -> Agent 2 (Scheduler) -> Agents 4+5 -> email/report
    docs = retriever.invoke(user_input) if retriever else []
    context = docs[0].page_content if docs else ""
    prompt = build_prompt(user_input, context)
    raw = generate_safe(prompt)
    data = extract_json(raw)

    if not data:
        response = "❌ Failed to extract JSON (is the model loaded?). Try the Booking tab with structured fields."
        eval_log.append({"task": "schedule", "ok": False,
                         "latency_s": round(time.time()-t0, 2), "note": "json fail"})
    elif data.get("action") == "none":
        # Pure policy Q&A: answer from retrieved context
        response = (context[:800] + "\n\n(This answer comes from the hospital policy PDF.)") if context else "No action needed."
        eval_log.append({"task": "qa", "ok": True,
                         "latency_s": round(time.time()-t0, 2), "note": "policy qa"})
    else:
        if not valid_date(data.get("date", "")):
            response = "❌ Invalid date (use YYYY-MM-DD, future date)."
        elif not valid_time(data.get("time", "")):
            response = "❌ Invalid time (use HH:MM)."
        else:
            okp, msg = validate_policy(data, context)
            if not okp:
                response = f"❌ Appointment violates policy: {msg}"
            else:
                # Insurance + billing pipeline
                ins = insurance_agent(data.get("patient", patient_hint))
                dept = data.get("department", "General")
                svc = "Specialist Visit" if dept in ("Cardiology", "Dermatology", "Orthopedics") else (
                    "Lab Test" if dept == "Lab" else "General Checkup")
                bill = billing_agent(svc, ins.get("insurance_type", "None"))
                response = schedule_appointment_api(
                    data["date"], data["time"], data.get("doctor", "Dr. Ahmed Hassan"),
                    dept, data.get("patient", patient_hint))
                if response.startswith("✅"):
                    appt = scheduled_appointments[-1]
                    email_to = ins.get("patient", {}).get("email", "patient@example.com")
                    email_body = (f"Dear {data.get('patient')},\nYour appointment is confirmed:\n"
                                  f"{appt}\nBilling: {bill}\nInsurance: {ins.get('insurance_type')}\n")
                    ep = send_confirmation_email(email_to, "Appointment Confirmed", email_body)
                    rp = generate_visit_report(appt, bill, ins.get("insurance_type"))
                    response += f"\n📧 Email saved: {ep}\n📝 Report saved: {rp}\n💳 Patient pays: {bill['patient_pays']} EGP"
                    eval_log.append({"task": "schedule", "ok": True,
                                     "latency_s": round(time.time()-t0, 2), "note": "full pipeline"})
                else:
                    eval_log.append({"task": "schedule", "ok": False,
                                     "latency_s": round(time.time()-t0, 2), "note": "collision"})

    memory.append({"user": user_input, "assistant": response})
    print_chat_history(user_input, context, data, response)
    print_scheduled_appointments()
    return mask_pii(response)

# ---------------- Gradio Functions (Session 9 style) -----------------
def chat(message, history):
    response = orchestrator(message)
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history

def clear_chat():
    memory.clear()
    return "", []

def book_structured(patient, department, doctor, appt_date, appt_time):
    msg = f"Book {department} with {doctor} for {patient} on {appt_date} at {appt_time}."
    return orchestrator(msg, patient_hint=patient)

def summarize_report(report_text):
    return orchestrator("Please summarize this discharge report: " + report_text, mode="summarize")

def check_billing(patient, question):
    return orchestrator(question, patient_hint=patient, mode="billing")

def dashboard():
    if not eval_log:
        return "No evaluations yet. Interact with the agents first."
    df = pd.DataFrame(eval_log)
    acc = (df["ok"].mean() * 100) if len(df) else 0
    avg_lat = df["latency_s"].mean() if len(df) else 0
    return (f"### 📊 Evaluation Dashboard\n- Total tasks: {len(df)}\n"
            f"- Success rate: {acc:.1f}%\n- Avg latency: {avg_lat:.2f}s\n\n"
            + df.to_markdown(index=False))

# ---------------- Gradio UI ------------------------------------------
def build_ui():
    with gr.Blocks(title="🏥 AI Healthcare Administration Assistant") as demo:
        gr.Markdown("# 🏥 AI Healthcare Administration Assistant\n"
                    "Multi-agent: Policy RAG • Scheduler • Summarizer • Billing • Insurance + Email/Report automation")
        with gr.Tab("💬 Assistant Chat (Orchestrator)"):
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(label="Ask or book (e.g. 'Book Cardiology with Dr. Sara Mahmoud for P-1001 next Monday 11:00')")
            msg.submit(chat, [msg, chatbot], [msg, chatbot])
            gr.Button("Clear").click(clear_chat, outputs=[msg, chatbot])
        with gr.Tab("📅 Book Appointment"):
            patient = gr.Textbox(label="Patient (name or P-XXXX)", value="P-1001")
            dept = gr.Dropdown(["General", "Cardiology", "Dermatology", "Orthopedics", "Lab"], value="General")
            doc = gr.Dropdown(["Dr. Ahmed Hassan", "Dr. Sara Mahmoud", "Dr. Omar Khaled",
                               "Dr. Lina Adel", "Lab Team"], value="Dr. Ahmed Hassan")
            d = gr.Textbox(label="Date YYYY-MM-DD", value="2026-10-12")
            t = gr.Textbox(label="Time HH:MM", value="10:00")
            out = gr.Textbox(label="Result")
            gr.Button("Book").click(book_structured, [patient, dept, doc, d, t], out)
        with gr.Tab("📝 Summarize Report"):
            rep = gr.Textbox(label="Paste discharge / lab report", lines=8)
            sout = gr.Textbox(label="Summary")
            gr.Button("Summarize").click(summarize_report, rep, sout)
        with gr.Tab("💳 Billing & Insurance"):
            pat2 = gr.Textbox(label="Patient", value="P-1002")
            q = gr.Textbox(label="Question", value="How much for Specialist Visit with my insurance?")
            bout = gr.Textbox(label="Result")
            gr.Button("Check").click(check_billing, [pat2, q], bout)
        with gr.Tab("📊 Evaluation Dashboard"):
            dout = gr.Markdown()
            gr.Button("Refresh").click(dashboard, outputs=dout)
    return demo

if __name__ == "__main__":
    init_rag()
    try:
        load_model()
    except Exception as e:
        print("Model load skipped (offline mode still works for structured booking):", e)
    demo = build_ui()
    demo.launch(share=True)
