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
from datetime import datetime, date, timedelta
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
        try:
            from langchain_core.documents import Document
        except ImportError:  # older langchain versions
            from langchain.docstore.document import Document  # type: ignore
        docs = [Document(page_content=txt_path.read_text(encoding="utf-8"))]
        print("PDF not found, using TXT fallback. Run create_sample_data.py.")
    retriever = build_retriever(docs)
    return retriever

# ---------------- Model Loading (Session 10 Quantization) ------------
# Qwen3-4B-Instruct: 4-bit BitsAndBytes on GPU (same as Quantization.ipynb),
# full-precision on CPU (device_map="cpu") so it runs WITHOUT a GPU.
# Override with env var:  set HEALTHCARE_MODEL=Qwen/other-model
import os as _os

model_name = _os.environ.get("HEALTHCARE_MODEL", "Qwen/Qwen3-1.7B")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = None  # set by load_model(); None = offline mode
model = None  # set by load_model(); None = offline mode
model_device = "cpu"  # updated by load_model()

def load_model():
    global tokenizer, model, model_device
    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_cuda:
        # GPU path: 4-bit quantization (Session 10 style)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model_device = "cuda"
    else:
        # CPU path: full weights on CPU (Qwen3-1.7B safetensors are bfloat16,
        # ~3.5GB RAM). Never skipped — fast enough for chat on CPU.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model_device = "cpu"
    model.eval()
    print(f"Model loaded successfully! ({model_name} on {model_device})")

# ---------------- Generate Function (Session 9 style) ----------------
def build_chat_text(messages):
    # Qwen3 is a hybrid thinking model: disable thinking so it returns
    # ONLY the answer (strict JSON), no <think> chains.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:  # older tokenizer template without the flag
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

def generate(prompt, max_new_tokens=120):
    messages = [
        {"role": "system",
         "content": "You are a strict JSON generator. Return ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]
    text = build_chat_text(messages)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(  # type: ignore[attr-defined]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,  # Qwen3 recommended sampling for non-thinking mode
            top_p=0.8,
            top_k=20,
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
# Intent-based: only blocks REQUESTS for medical advice, not words merely
# mentioned inside a pasted report (e.g. "as prescribed by Dr…").
BLOCKED_PATTERNS = [
    r"prescrib\w*\s+(me|us|for me|for us)", r"prescription\s+for\s+me",
    r"give\s+me\s+a\s+prescription", r"write\s+me\s+a\s+prescription",
    r"what\s+dosage", r"which\s+dosage", r"correct\s+dosage",
    r"dosage\s+(should|for|do)\b", r"how\s+many\s+mg\s+should",
    r"how\s+much\s+should\s+i\s+take", r"diagnose\s+me", r"diagnose\s+my",
    r"what\s+disease\s+do\s+i\s+have", r"do\s+i\s+have\s+\w+",
    r"what(\'s| is)\s+wrong\s+with\s+me", r"emergency.*bleed",
    r"chest pain.*what should i do",
]
EMERGENCY_KEYWORDS = ["chest pain", "severe bleeding", "can't breathe", "unconscious"]

SUMMARY_KEYWORDS = ["summariz", "discharge", "lab report", "report:"]

def is_summary_request(user_input, mode="auto"):
    if mode == "summarize":
        return True
    return any(k in user_input.lower() for k in SUMMARY_KEYWORDS)

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

# ---------------- Natural Language Understanding -------------------
# Deterministic parsers so chat feels natural and works fast — even with a
# small model or no model at all. The LLM is only a fallback, never a blocker.
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}

DEPT_KEYWORDS = {"cardiology": "Cardiology", "dermatology": "Dermatology",
                 "orthopedics": "Orthopedics", "orthopedic": "Orthopedics",
                 "check-up": "General", "checkup": "General", "check up": "General",
                 "general": "General", "lab test": "Lab", "lab": "Lab"}

_MENTION_STOPWORDS = {"me", "us", "today", "tomorrow", "checkup", "check-up",
                      "general", "lab", "cardiology", "dermatology", "orthopedics"}

def parse_natural_date(text):
    low = text.lower()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", low)
    if m:
        return m.group(1)
    today = date.today()
    if re.search(r"\btomorrow\b", low):
        return (today + timedelta(days=1)).isoformat()
    if re.search(r"\btoday\b", low):
        return today.isoformat()
    m = re.search(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low)
    if m:
        target = WEEKDAYS[m.group(1)]
        days = (target - today.weekday()) % 7 + 7
        return (today + timedelta(days=days)).isoformat()
    m = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low)
    if m:
        target = WEEKDAYS[m.group(1)]
        days = (target - today.weekday()) % 7 or 7
        return (today + timedelta(days=days)).isoformat()
    return None

def parse_natural_time(text):
    low = text.lower()
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", low)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return f"{h:02d}:{mi:02d}"
        return None
    m = re.search(r"\bat\s+(\d{1,2})\s*(am|pm)\b", low)
    if m:
        h, ap = int(m.group(1)), m.group(2)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 23:
            return f"{h:02d}:00"
        return None
    m = re.search(r"\bat\s+(\d{1,2})\b", low)
    if m and 0 <= int(m.group(1)) <= 23:
        return f"{int(m.group(1)):02d}:00"
    return None

def find_doctor(text):
    low = text.lower()
    all_docs = [(dept, d) for dept, ds in DOCTORS.items()
                for d in ds if d != "Lab Team"]
    for dept, d in all_docs:  # full name mentioned
        if d.lower() in low:
            return d, dept
    for dept, d in all_docs:  # "Dr. Mahmoud" style
        last = d.split()[-1].lower()
        if re.search(r"\bdr\.?\s+" + re.escape(last) + r"\b", low):
            return d, dept
    for dept, d in all_docs:  # first name mentioned
        first = d.split()[1].lower()
        if re.search(r"\b" + re.escape(first) + r"\b", low):
            return d, dept
    return None, None

def find_department(text):
    low = text.lower()
    for kw in sorted(DEPT_KEYWORDS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(kw) + r"\b", low):
            return DEPT_KEYWORDS[kw]
    return None

def extract_patient_mention(text):
    m = re.search(r"\bP-\d+\b", text)
    if m:
        return m.group(0)
    m = re.search(r"\bfor\s+([A-Za-z][A-Za-z .'\-]*?)(?=\s+on\b|\s+at\b|\s*,|\s*\d{4}-\d{2}-\d{2}|\s*$)",
                  text, re.IGNORECASE)
    if m:
        mention = m.group(1).strip().strip(".,")
        if mention.lower() not in _MENTION_STOPWORDS:
            return mention
    return None

def parse_booking_request(text):
    dept = find_department(text)
    doc, doc_dept = find_doctor(text)
    if dept is None and doc_dept:
        dept = doc_dept  # infer department from the doctor
    if doc is None and dept in DOCTORS and len(DOCTORS[dept]) == 1 and dept != "Lab":
        doc = DOCTORS[dept][0]  # infer doctor from a unique department
    if doc is None and dept == "Lab":
        doc = "Lab Team"
    return {"date": parse_natural_date(text),
            "time": parse_natural_time(text),
            "doctor": doc,
            "department": dept,
            "patient_mention": extract_patient_mention(text)}

def booking_signals(parsed):
    return sum(1 for v in parsed.values() if v)

def resolve_patient(mention, patient_hint="P-1001"):
    """Match a name/ID to the patient DB. Exact first, then fuzzy substring.

    Returns (status, patient_id, display_name, guidance_note).
    status: 'ok' | 'ambiguous' | 'notfound'.
    """
    if not mention:
        ins = insurance_agent(patient_hint)
        if ins.get("ok"):
            row = ins["patient"]
            return "ok", row["patient_id"], row["full_name"], ""
        return ("notfound", None, None,
                "❌ I don't know who to book for. Tell me a name or P-XXXX, "
                "or register them first in the ➕ Add Patient tab.")
    ins = insurance_agent(mention)
    if ins.get("ok"):
        row = ins["patient"]
        return "ok", row["patient_id"], row["full_name"], ""
    df, _ = _read_patients()
    q = str(mention).strip().lower()
    cands = []
    for _, r in df.iterrows():
        nm = str(r.get("full_name", "")).lower()
        if not nm:
            continue
        if q in nm or any(len(w) > 2 and w == q for w in nm.split()):
            cands.append({"patient_id": r.get("patient_id", ""),
                          "full_name": r.get("full_name", "")})
    seen, uniq = set(), []
    for c in cands:
        if c["patient_id"] not in seen:
            seen.add(c["patient_id"])
            uniq.append(c)
    if len(uniq) == 1:
        return "ok", uniq[0]["patient_id"], uniq[0]["full_name"], ""
    if len(uniq) > 1:
        opts = ", ".join(f"{c['full_name']} ({c['patient_id']})" for c in uniq)
        return ("ambiguous", None, None,
                f"❓ \"{mention}\" matches several patients: {opts}. "
                "Which one (full name or P-XXXX)?")
    return ("notfound", None, None,
            f"❌ I couldn't find \"{mention}\" in the patient records. "
            "Register them in the ➕ Add Patient tab, then I'll book right away.")

def complete_booking(appt_date, appt_time, doctor, department, patient_id, patient_name):
    """Shared booking pipeline with natural-language replies."""
    if not valid_date(appt_date):
        return ("❌ That date doesn't work — I need a future date like YYYY-MM-DD "
                "(e.g. 2026-10-12). Which day suits you?")
    if not valid_time(appt_time):
        return "❌ I need a valid time like 10:00 or 2:30pm. What time suits you?"
    okp, reason = validate_policy({"department": department, "date": appt_date,
                                   "time": appt_time})
    if not okp:
        return (f"❌ Can't book that slot: {reason}\n"
                "Hours — General: Mon–Fri 09:00–16:00 · Specialists: Mon–Thu "
                "10:00–15:00 · Lab: Mon–Sat 08:00–18:00. Closed Sundays.\n"
                "Give me another date/time and I'll book it right away.")
    ins = insurance_agent(patient_id)
    svc = ("Specialist Visit"
           if department in ("Cardiology", "Dermatology", "Orthopedics") else
           "Lab Test" if department == "Lab" else "General Checkup")
    bill = billing_agent(svc, ins.get("insurance_type", "None"))
    booked = schedule_appointment_api(appt_date, appt_time, doctor, department, patient_id)
    if not booked.startswith("✅"):
        return (f"❌ {doctor} is already booked on {appt_date} at {appt_time}. "
                "Pick another time and I'll book it right away.")
    appt = scheduled_appointments[-1]
    email_to = ins.get("patient", {}).get("email", "patient@example.com")
    email_body = (f"Dear {patient_name},\nYour appointment is confirmed:\n"
                  f"{appt}\nBilling: {bill}\nInsurance: {ins.get('insurance_type')}\n")
    send_confirmation_email(email_to, "Appointment Confirmed", email_body)
    generate_visit_report(appt, bill, ins.get("insurance_type"))
    pays = bill["patient_pays"]
    pay_text = (f"Covered in full by {ins.get('insurance_type')} insurance — you pay 0 EGP."
                if pays == 0 else
                f"You pay {pays} EGP. {bill.get('note', '')}".strip())
    return (f"✅ You're booked, {patient_name}!\n"
            f"🩺 {doctor} — {department}\n"
            f"📅 {appt_date} at {appt_time} · ID {appt['appointment_id']}\n"
            f"💳 {pay_text}\n"
            "📧 Confirmation email + visit report saved.")

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

# ---------------- Patient Registration (new patient onboarding) ------
PATIENT_CSV = DATA / "patients.csv"
PATIENT_COLUMNS = ["patient_id", "full_name", "email", "insurance_type", "insurance_id", "phone"]

def _read_patients(csv_path=None):
    path = Path(csv_path) if csv_path else PATIENT_CSV
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except FileNotFoundError:
        df = pd.DataFrame(columns=PATIENT_COLUMNS)
    return df, path

def _next_patient_id(df):
    nums = []
    for pid in df.get("patient_id", []):
        m = re.match(r"^P-(\d+)$", str(pid).strip())
        if m:
            nums.append(int(m.group(1)))
    return f"P-{(max(nums) if nums else 1000) + 1}"

def add_patient(full_name, email, insurance_type="None", insurance_id="", phone="", csv_path=None):
    """Register a new patient in patients.csv (stands in for Google Sheets / DB).

    Returns (ok: bool, message: str). The message contains the new patient ID.
    """
    name = str(full_name or "").strip()
    mail = str(email or "").strip()
    ins = str(insurance_type or "None").strip().capitalize()
    ins_id = str(insurance_id or "").strip()
    ph = str(phone or "").strip()

    if not name:
        return False, "❌ Full name is required."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", mail):
        return False, "❌ Invalid email format."
    if ins not in ("Basic", "Premium", "None"):
        return False, "❌ Insurance type must be Basic, Premium, or None."
    if ins == "None":
        ins_id = ""
    elif not re.match(r"^INS-\d+$", ins_id):
        return False, "❌ Insurance ID must look like INS-1001 (or choose None)."

    df, path = _read_patients(csv_path)
    low_mail = mail.lower()
    if any(str(e).strip().lower() == low_mail for e in df.get("email", [])):
        return False, f"❌ A patient with email {mail} already exists."
    if ins_id and any(str(i).strip() == ins_id for i in df.get("insurance_id", [])):
        return False, f"❌ Insurance ID {ins_id} is already registered."

    new_id = _next_patient_id(df)
    row = {"patient_id": new_id, "full_name": name, "email": mail,
           "insurance_type": ins, "insurance_id": ins_id, "phone": ph}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    print("\n===== PATIENT REGISTERED =====")
    print(row)
    return True, (f"✅ Patient registered! ID: {new_id}\n{name} | {mail} | "
                  f"Insurance: {ins}{f' ({ins_id})' if ins_id else ''}")

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
    low = user_input.lower()
    # Summarization requests carry a pasted document: scan only for a genuine
    # request elsewhere — never block on words inside the document itself.
    if not is_summary_request(user_input, mode):
        ok, block_msg = guardrail_check(user_input)
        if not ok:
            eval_log.append({"task": "guardrail", "ok": False, "latency_s": 0, "note": "blocked"})
            memory.append({"user": user_input, "assistant": block_msg})
            return block_msg

    # Route: patient registration
    if any(k in low for k in ["add patient", "register patient", "new patient",
                              "create patient", "add a patient"]):
        prompt = (
            "Extract new-patient details from the request below. Return ONLY valid JSON, "
            "no markdown, no explanations.\n"
            'If details are present return: {"full_name": "string", "email": "string", '
            '"insurance_type": "Basic|Premium|None", "insurance_id": "string (may be empty)", '
            '"phone": "string (may be empty)"}.\n'
            'If name or email is missing return exactly: {"action": "need_details"}\n\n'
            f"Request:\n{user_input}"
        )
        data = extract_json(generate_safe(prompt))
        if not data or data.get("action") == "need_details" or not data.get("full_name") or not data.get("email"):
            resp = ("📝 To register a patient I need at least **full name and email**.\n"
                    "Example: 'Add patient Laila Hassan, email laila.h@example.com, "
                    "Premium insurance INS-1006, phone 01005556666'\n"
                    "Or use the ➕ Add Patient tab for a form.")
            ok_flag = False
        else:
            ok_flag, resp = add_patient(data.get("full_name"), data.get("email"),
                                        data.get("insurance_type", "None"),
                                        data.get("insurance_id", ""),
                                        data.get("phone", ""))
        eval_log.append({"task": "add_patient", "ok": ok_flag,
                         "latency_s": round(time.time() - t0, 2), "note": "registration"})
        memory.append({"user": user_input, "assistant": resp})
        return mask_pii(resp)

    # Route: summarizer FIRST — a pasted report contains dept/doctor/date
    # words that would otherwise look like a booking request.
    if is_summary_request(user_input, mode):
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

    # Route: natural booking (deterministic parser — fast, works with or without LLM).
    # Falls through to the LLM path below only when details are missing AND a model is loaded.
    parsed = parse_booking_request(user_input)
    if (any(k in low for k in ["book", "appointment", "schedule", "reserve"])
            or booking_signals(parsed) >= 2):
        missing = [k for k in ("doctor", "department", "date", "time") if not parsed[k]]
        status, pid, pname, note = resolve_patient(parsed["patient_mention"], patient_hint)
        resp, ok_flag = None, None
        if status != "ok":
            resp, ok_flag = note, False
        elif missing and model is None:
            resp = ("Almost there! I still need: " + ", ".join(missing) + ".\n"
                    "Example: 'Book General Checkup with Dr. Ahmed Hassan for P-1002 "
                    "on 2026-10-12 at 10:00'.")
            ok_flag = False
        elif not missing:
            resp = complete_booking(parsed["date"], parsed["time"], parsed["doctor"],
                                    parsed["department"], pid, pname)
            ok_flag = resp.startswith("✅")
        if resp is not None:
            eval_log.append({"task": "schedule", "ok": ok_flag,
                             "latency_s": round(time.time() - t0, 2), "note": "natural"})
            memory.append({"user": user_input, "assistant": resp})
            return mask_pii(resp)

    # Route: billing / insurance quick lookup
    if mode in ("billing", "insurance") or any(k in low for k in ["price", "cost", "bill", "insurance", "claim", "cover", "how much", "pay", "fee", "charge"]):
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
        # LLM-extracted booking -> same natural pipeline (fuzzy patient + validation)
        status, pid, pname, note = resolve_patient(data.get("patient", patient_hint),
                                                   patient_hint)
        if status != "ok":
            response = note
            eval_log.append({"task": "schedule", "ok": False,
                             "latency_s": round(time.time()-t0, 2), "note": "patient"})
        else:
            response = complete_booking(data.get("date", ""), data.get("time", ""),
                                        data.get("doctor") or "Dr. Ahmed Hassan",
                                        data.get("department") or "General",
                                        pid, pname)
            eval_log.append({"task": "schedule", "ok": response.startswith("✅"),
                             "latency_s": round(time.time()-t0, 2), "note": "llm pipeline"})

    memory.append({"user": user_input, "assistant": response})
    print_chat_history(user_input, context, data, response)
    print_scheduled_appointments()
    return mask_pii(response)

# ---------------- Gradio Functions (Session 9 style) -----------------
def chat(message, history, patient="P-1001"):
    response = orchestrator(message, patient_hint=(patient or "P-1001"))
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

def register_patient_structured(full_name, email, insurance_type, insurance_id, phone):
    ok, msg = add_patient(full_name, email, insurance_type, insurance_id, phone)
    eval_log.append({"task": "add_patient", "ok": ok, "latency_s": 0, "note": "tab"})
    memory.append({"user": f"Register patient {full_name} ({email})", "assistant": msg})
    return mask_pii(msg)

def dashboard():
    if not eval_log:
        return "No evaluations yet. Interact with the agents first."
    df = pd.DataFrame(eval_log)
    acc = (df["ok"].mean() * 100) if len(df) else 0
    avg_lat = df["latency_s"].mean() if len(df) else 0
    try:
        table = df.to_markdown(index=False)
    except ImportError:  # tabulate not installed -> plain text fallback
        table = "```\n" + df.to_string(index=False) + "\n```"
    return (f"### 📊 Evaluation Dashboard\n- Total tasks: {len(df)}\n"
            f"- Success rate: {acc:.1f}%\n- Avg latency: {avg_lat:.2f}s\n\n"
            + table)

# ---------------- Gradio UI ------------------------------------------
def build_ui():
    with gr.Blocks(title="🏥 AI Healthcare Administration Assistant") as demo:
        gr.Markdown("# 🏥 AI Healthcare Administration Assistant\n"
                    "Multi-agent: Policy RAG • Scheduler • Summarizer • Billing • Insurance + Email/Report automation")
        with gr.Tab("💬 General Chat (does everything)"):
            gr.Markdown("One chat for **all agents** — just type naturally:\n"
                        "- 📅 Book: *Book Cardiology with Dr. Sara Mahmoud for P-1001 on 2026-10-13 at 11:00*\n"
                        "- 📖 Policy: *What are the Cardiology hours?*\n"
                        "- 💳 Billing/Insurance: *How much for a Specialist Visit?*\n"
                        "- 📝 Summarize: *Summarize this discharge report: …*\n"
                        "- ➕ Register: *Add patient Laila Hassan, email laila.h@example.com, Premium INS-1006*")
            who = gr.Textbox(label="Acting as patient (name or P-XXXX)", value="P-1001")
            chatbot = gr.Chatbot()  # NOTE: no type= arg for Gradio < 5 compat
            msg = gr.Textbox(label="Message")
            msg.submit(chat, [msg, chatbot, who], [msg, chatbot])
            gr.Examples(
                ["What are the Cardiology hours?",
                 "Book General Checkup with Dr. Ahmed Hassan for P-1002 on 2026-10-12 at 10:00",
                 "How much for a Specialist Visit?",
                 "Add patient Laila Hassan, email laila.h@example.com, Premium INS-1006",
                 "Summarize this discharge report: patient had mild chest discomfort, ECG normal, Atorvastatin 20mg once daily."],
                inputs=msg, label="Try one click",
            )
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
        with gr.Tab("➕ Add Patient"):
            np_name = gr.Textbox(label="Full name")
            np_email = gr.Textbox(label="Email")
            np_ins = gr.Dropdown(["Basic", "Premium", "None"], value="None", label="Insurance type")
            np_ins_id = gr.Textbox(label="Insurance ID (e.g. INS-1006, empty if None)")
            np_phone = gr.Textbox(label="Phone (optional)")
            np_out = gr.Textbox(label="Result")
            gr.Button("Register").click(register_patient_structured,
                                        [np_name, np_email, np_ins, np_ins_id, np_phone], np_out)
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
        load_model()  # GPU (4-bit) or CPU (full weights) — always attempted, never skipped
    except Exception as e:
        print("Model load failed (offline mode: billing/insurance/guardrails still work):", e)
    demo = build_ui()
    demo.launch(share=True)
