# ============================================================
# create_sample_data.py
# YOU run this yourself when ready (I am NOT running it).
# It generates: hospital_policy.pdf from hospital_policy.txt
# Usage:  python create_sample_data.py
# ============================================================
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"

def build_pdf():
    try:
        from fpdf import fpdf
        from fpdf import FPDF
    except ImportError:
        print("Please install fpdf2 first: pip install fpdf2")
        return

    txt_path = DATA / "hospital_policy.txt"
    pdf_path = DATA / "hospital_policy.pdf"

    text = txt_path.read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "City General Hospital - Policy Manual", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, text)
    pdf.output(str(pdf_path))
    print(f"OK -> {pdf_path}")

if __name__ == "__main__":
    build_pdf()
    print("Seed CSVs already in data/. Nothing else to do.")
