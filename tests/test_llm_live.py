# ============================================================
# tests/test_llm_live.py - needs the LLM loaded (slow, CPU ok).
# Session-scoped: model + RAG load once; whole file skips if unavailable.
# YOU run this yourself:  python -m pytest tests/test_llm_live.py -v
# ============================================================
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app


@pytest.fixture(scope="session")
def live():
    try:
        app.init_rag()
        app.load_model()
    except Exception as e:
        pytest.skip(f"model unavailable: {e}")
    assert app.require_model()
    return app


def test_llm_booking_policy_refusal_is_natural(live):
    # Reported case, end to end through the LLM: 9:00 is outside
    # Cardiology hours -> natural refusal, never a policy dump.
    r = live.orchestrator(
        "Book Cardiology with Dr. Sara Mahmoud for Omar Farouk on 2026-10-15 at 9:00")
    assert "policy PDF" not in r
    assert "Can't book that slot" in r


def test_llm_billing_chat_answers_from_facts(live):
    r = live.orchestrator("How much for a Specialist Visit?", patient_hint="P-1002")
    assert "800" in r and "⛔" not in r


def test_llm_summarizes_real_report(live):
    report = (Path(__file__).parent.parent / "data" / "sample_discharge.txt").read_text()
    r = live.orchestrator("Summarize this discharge report: " + report)
    assert "⛔" not in r and "🚨" not in r
    assert "administrative summary, not medical advice" in r


def test_llm_policy_qa_grounded(live):
    r = live.orchestrator("What are the Cardiology hours?")
    assert "10:00" in r and "15:00" in r


def test_llm_registration_chat_then_cleanup(live):
    mail = "live.test.patient@example.com"
    r = live.orchestrator(
        f"Add patient Live Test, email {mail}, Basic insurance INS-1999, phone 01000000001")
    assert "P-" in r and "registered" in r.lower()
    # Cleanup: remove the test row so the real CSV stays untouched.
    import pandas as pd
    path = live.PATIENT_CSV
    df = pd.read_csv(path, dtype=str)
    df = df[df["email"].str.lower() != mail.lower()]
    df.to_csv(path, index=False)
