# ============================================================
# tests/test_evaluation.py - Session 9 (Evaluation & Debugging)
# YOU run this yourself:  python -m pytest tests/ -v
# Tests run WITHOUT the LLM (offline-safe) using structured paths.
# ============================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    valid_date, valid_time, validate_policy, appointment_exists,
    billing_agent, insurance_agent, guardrail_check, mask_pii,
    extract_json,
)

def test_validators():
    assert valid_date("2026-10-12") is True
    assert valid_date("12-10-2026") is False
    assert valid_time("10:00") is True
    assert valid_time("25:00") is False

def test_policy_general_ok():
    ok, _ = validate_policy({"department": "General", "date": "2026-10-12", "time": "10:00"})
    assert ok is True  # Monday

def test_policy_sunday_blocked():
    ok, msg = validate_policy({"department": "General", "date": "2026-10-11", "time": "10:00"})
    assert ok is False  # 2026-10-11 is a Sunday

def test_policy_specialist_hours():
    ok, _ = validate_policy({"department": "Cardiology", "date": "2026-10-13", "time": "18:00"})
    assert ok is False  # after 15:00

def test_billing_rules():
    b = billing_agent("Specialist Visit", "Basic")
    assert b["patient_pays"] == 800  # Basic does not cover specialist
    b2 = billing_agent("General Checkup", "Basic")
    assert b2["patient_pays"] == 0

def test_insurance_lookup():
    ins = insurance_agent("P-1001")
    assert ins["ok"] is True and ins["insurance_type"] == "Premium"
    missing = insurance_agent("NO-ONE")
    assert missing["insurance_type"] == "None"

def test_guardrails_block_diagnosis():
    ok, _ = guardrail_check("Can you prescribe me dosage for my chest pain?")
    assert ok is False

def test_guardrails_emergency():
    ok, msg = guardrail_check("I have severe bleeding, what should I do?")
    assert ok is False and "emergency" in msg.lower()

def test_pii_masking():
    assert "29801011234567" not in mask_pii("ID 29801011234567")

def test_extract_json():
    d = extract_json('Here: {"action": "none"} done')
    assert d == {"action": "none"}

def test_no_double_booking_seed():
    assert appointment_exists("Dr. Ahmed Hassan", "2026-10-05", "10:00") is True
