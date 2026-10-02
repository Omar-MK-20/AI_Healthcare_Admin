# ============================================================
# tests/test_evaluation.py - (Evaluation & Debugging)
# YOU run this yourself:  python -m pytest tests/ -v
# Tests run WITHOUT the LLM (offline-safe) using structured paths.
# ============================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    valid_date, valid_time, validate_policy, appointment_exists,
    billing_agent, insurance_agent, guardrail_check, mask_pii,
    extract_json, add_patient, parse_booking_request, resolve_patient,
    complete_booking, orchestrator, parse_natural_date, parse_natural_time,
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

def _seed_csv(tmp_path):
    p = tmp_path / "patients.csv"
    p.write_text(
        "patient_id,full_name,email,insurance_type,insurance_id,phone\n"
        "P-1001,John Samir,john.samir@example.com,Premium,INS-1001,01001234567\n"
    )
    return str(p)

def test_add_patient_ok(tmp_path):
    csv_path = _seed_csv(tmp_path)
    ok, msg = add_patient("Laila Hassan", "laila.h@example.com",
                          "Premium", "INS-1006", "01005556666", csv_path=csv_path)
    assert ok is True and "P-1002" in msg
    # new patient is immediately lookable (insurance agent reads the same CSV)
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str)
    assert len(df) == 2

def test_add_patient_duplicate_email(tmp_path):
    csv_path = _seed_csv(tmp_path)
    ok, msg = add_patient("Someone Else", "JOHN.samir@example.com",
                          "None", "", "", csv_path=csv_path)
    assert ok is False and "already exists" in msg

def test_add_patient_bad_insurance_id(tmp_path):
    csv_path = _seed_csv(tmp_path)
    ok, _ = add_patient("Bad Ins", "bad@example.com", "Basic", "XYZ", "", csv_path=csv_path)
    assert ok is False
    ok, _ = add_patient("Bad Mail", "not-an-email", "None", "", "", csv_path=csv_path)
    assert ok is False

def test_parse_booking_full_sentence():
    p = parse_booking_request(
        "Book Cardiology with Dr. Sara Mahmoud for Omar on 2026-10-15 at 9:00")
    assert p["department"] == "Cardiology"
    assert p["doctor"] == "Dr. Sara Mahmoud"
    assert p["date"] == "2026-10-15"
    assert p["time"] == "09:00"
    assert p["patient_mention"] == "Omar"

def test_parse_relative_datetime():
    from datetime import date as _d, timedelta as _td
    assert parse_natural_date("book for tomorrow") == (_d.today() + _td(days=1)).isoformat()
    assert parse_natural_time("at 2:30pm") == "14:30"
    assert parse_natural_time("at 9") == "09:00"

def test_resolve_patient_fuzzy():
    status, pid, name, _ = resolve_patient("Farouk")
    assert status == "ok" and pid == "P-1003" and "Omar" in name

def test_resolve_patient_ambiguous():
    # "Omar" matches more than one record -> must ask, never guess
    status, _, _, note = resolve_patient("Omar")
    assert status == "ambiguous" and "Which one" in note

def test_resolve_patient_unknown():
    status, _, _, note = resolve_patient("Nobody Here")
    assert status == "notfound" and "Add Patient" in note

def test_booking_example_gets_natural_reply_not_policy_dump():
    # Regression test: the exact reported case. 9:00 is outside Cardiology
    # hours, so it must refuse NATURALLY (with hours + alternative) —
    # never dump the policy PDF. No LLM needed (deterministic route).
    r = orchestrator("Book Cardiology with Dr. Sara Mahmoud for Omar Farouk on 2026-10-15 at 9:00")
    assert "policy PDF" not in r
    assert "Can't book that slot" in r and "10:00" in r

def test_report_words_dont_trigger_guardrail():
    # "as prescribed by Dr…" inside a document is a mention, not a request.
    ok, _ = guardrail_check(
        "Current medication: Atorvastatin 20mg once daily (as prescribed by Dr. Sara Mahmoud).")
    assert ok is True

def test_genuine_prescription_requests_still_blocked():
    ok, _ = guardrail_check("Can you prescribe me something for my headache?")
    assert ok is False
    ok, _ = guardrail_check("What dosage should I take?")
    assert ok is False

def test_summarize_report_not_blocked():
    # The exact reported failure: summary request + full real report file.
    # Must reach the SUMMARIZER (offline stub echoes JSON), never the
    # guardrail block nor the booking route ("Almost there").
    report = (Path(__file__).parent.parent / "data" / "sample_discharge.txt").read_text()
    r = orchestrator("Summarize this discharge report: " + report)
    assert "⛔" not in r and "🚨" not in r and "Almost there" not in r

def test_billing_result_is_formatted_markdown():
    from app import format_billing_insurance
    bill = billing_agent("Specialist Visit", "Basic")
    ins = insurance_agent("P-1002")
    out = format_billing_insurance(bill, ins)
    assert "{'service'" not in out and "{'ok'" not in out  # no raw dicts
    assert "### 💳 Billing" in out and "### 🛡 Insurance" in out
    assert "800 EGP" in out and "Mariam Adel (P-1002)" in out
    assert "Basic (INS-1002)" in out

def test_patients_table_masks_phones():
    from app import patients_table
    df = patients_table()
    assert len(df) >= 7 and list(df.columns) == [
        "patient_id", "full_name", "email", "insurance_type", "insurance_id", "phone"]
    assert "01001234567" not in " ".join(df["phone"].tolist())  # masked
    assert "P-1001" in df["patient_id"].tolist()

def test_appointments_table_shows_seeds_and_status():
    from app import appointments_table
    df = appointments_table()
    assert list(df.columns) == ["appointment_id", "patient_id", "date", "time",
                                "doctor", "department", "status"]
    assert "A-0001" in df["appointment_id"].tolist()
    assert set(df["status"].tolist()) <= {"confirmed"}
