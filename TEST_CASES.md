# ✅ TEST CASES — AI Healthcare Administration Assistant
Use these in the **💬 General Chat** tab (`python app.py`).
Set **"Acting as patient"** to the value in each case (or leave P-1001).

> Run each case in a **fresh session** (`Clear` button) unless it says "run right after".
> New bookings get IDs `A-0003, A-0004…` (A-0001/A-0002 are pre-booked seeds).

## 1. Natural booking — valid slot (the happy path)
**Acting as:** `P-1003`
**Input:**
`Book Cardiology with Dr. Sara Mahmoud for Omar Farouk on 2026-10-15 at 11:00`
**Expected:** ✅ confirmation, NOT a policy dump:
`✅ You're booked, Omar Farouk! / 🩺 Dr. Sara Mahmoud — Cardiology / 📅 2026-10-15 at 11:00 · ID A-0003 / 💳 You pay 800 EGP… / 📧 Confirmation email + visit report saved.`
(Omar Farouk has no insurance → pays full 800. Check `outbox/` for the email + report files.)

## 2. Booking outside policy hours (your reported example)
**Acting as:** `P-1003`
**Input:**
`Book Cardiology with Dr. Sara Mahmoud for Omar Farouk on 2026-10-15 at 9:00`
**Expected:** ❌ natural refusal WITH hours and an alternative — never a PDF dump:
`❌ Can't book that slot: Cardiology Specialist must be Mon-Thu 10:00-15:00… / Hours — General… / Give me another date/time…`
(9:00 is outside Cardiology hours, so refusing is correct. Change to `at 11:00` → books.)

## 3. Ambiguous patient name
**Input:**
`Book General Checkup with Dr. Ahmed Hassan for Omar on 2026-10-12 at 10:00`
**Expected:** ❓ clarification, never a guess:
`"Omar" matches several patients: Omar Farouk (P-1003), Omar Mohamed (P-1006). Which one?`

## 4. Double-booking collision (run right after case 1)
**Input:** same as case 1 again.
**Expected:** `❌ Dr. Sara Mahmoud is already booked on 2026-10-15 at 11:00. Pick another time…`

## 5. Past date + Sunday (validation)
**Input:** `Book General Checkup with Dr. Ahmed Hassan for P-1002 on 2020-01-01 at 10:00`
**Expected:** `❌ That date doesn't work — I need a future date…`
**Input:** `Book General Checkup with Dr. Ahmed Hassan for P-1002 on 2026-10-11 at 10:00`
**Expected:** `❌ Can't book that slot: No appointments on Sundays…` (2026-10-11 is a Sunday)

## 6. Billing by insurance type (rendered as Markdown, no raw dicts)
**Acting as:** `P-1002` (Basic) — **Input:** `How much for a Specialist Visit?`
**Expected:** `### 💳 Billing` with `You pay: 800 EGP` + `Basic does NOT cover Specialist`,
and `### 🛡 Insurance` with `✅ verified`, `Basic (INS-1002)`, `Mariam Adel (P-1002)`.
**Acting as:** `P-1001` (Premium) — same input.
**Expected:** `You pay: 640 EGP` + `Premium 20% discount`.

## 7. Policy Q&A (needs model loaded)
**Input:** `What are the Cardiology hours?`
**Expected:** answer from the policy PDF (Mon–Thu 10:00–15:00) ending with `(This answer comes from the hospital policy PDF.)`

## 8. Report summarizer (needs model loaded)
**Input:** `Summarize this discharge report: ` + contents of `data/sample_discharge.txt`
**Expected:** 3–5 short bullets, medication copied exactly, ends with `This is an administrative summary, not medical advice.` National ID masked (`**********4567`).

## 9. Register a new patient (➕ Add Patient tab — no model needed)
Fill: name `Nour El-Din`, email `nour.e@example.com`, insurance `Basic`, ID `INS-1010`, phone `01009998888` → Register.
**Expected:** `✅ Patient registered! ID: P-1008 …` (next free ID).
Then in chat: `How much for a Lab Test?` as `P-1008` → `patient_pays: 0`, Basic covers in full.
**Duplicate check:** register the same email again → `❌ A patient with email nour.e@example.com already exists.`

## 10. Guardrails (no model needed)
**Input:** `Can you prescribe me a dosage for my headache?`
**Expected:** `⛔ I can't provide diagnosis or prescriptions/dosages…`
**Input:** `I have severe bleeding, what should I do?`
**Expected:** `🚨 …emergency… call emergency services or visit the ER immediately.`

## 11. Evaluation dashboard
Open **📊 Evaluation Dashboard** → Refresh.
**Expected:** total tasks count, success rate %, avg latency, per-task table.

## Automated suite (same logic, no GUI)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
**Expected:** `20 passed` — includes a regression test of case 2 above.
