# ============================================================
# streamlit_app.py - Projects style (like parser.py)
# Lightweight frontend that talks to backend_api.py via Ngrok.
# YOU run this yourself:  streamlit run streamlit_app.py
# ============================================================
import streamlit as st
import requests

st.set_page_config(page_title="Healthcare Admin", page_icon="🏥", layout="centered")

st.markdown("# 🏥 AI Healthcare Administration Assistant")
st.caption("Streamlit frontend → FastAPI backend (Ngrok)")

API_URL = st.text_input("Backend /book URL", "http://localhost:8000/book")
API_KEY = st.text_input("API Key", "secret123", type="password")

tab1, tab2 = st.tabs(["📅 Book", "📝 Summarize"])

with tab1:
    msg = st.text_area("Request",
        "Book Cardiology with Dr. Sara Mahmoud for P-1001 on 2026-10-13 at 11:00")
    if st.button("Book Appointment"):
        with st.spinner("Booking..."):
            try:
                r = requests.post(API_URL, timeout=300,
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={"message": msg, "patient": "P-1001"})
                r.raise_for_status()
                st.success(r.json()["response"])
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    rep = st.text_area("Paste report", height=200)
    if st.button("Summarize"):
        base = API_URL.rsplit("/", 1)[0] + "/summarize"
        with st.spinner("Summarizing..."):
            try:
                r = requests.post(base, timeout=300,
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={"report": rep})
                r.raise_for_status()
                st.info(r.json()["response"])
            except Exception as e:
                st.error(f"Error: {e}")
