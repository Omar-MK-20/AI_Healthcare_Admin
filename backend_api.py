# ============================================================
# backend_api.py - style: FastAPI backend + Ngrok deploy
# YOU run this yourself:
#   pip install fastapi uvicorn pyngrok
#   python backend_api.py
# Then test with ngrok_client_test.py (set your NGROK URL + key).
# ============================================================
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app import init_rag, orchestrator  # reuse all 5 agents + orchestrator

API_KEY = "secret123"
app = FastAPI(title="Healthcare Admin API")

class BookRequest(BaseModel):
    message: str
    patient: Optional[str] = "P-1001"

class SummarizeRequest(BaseModel):
    report: str

@app.on_event("startup")
def startup():
    init_rag()
    # NOTE: model loading is lazy here so the API boots fast.
    # Call /load_model on a GPU machine (Kaggle/Colab) before /book with LLM.
    print("RAG ready. Call /load_model to enable LLM.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/load_model")
def load_model_route(authorization: str = Header("")):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "bad key")
    from app import load_model
    load_model()
    return {"loaded": True}

@app.post("/book")
def book(req: BookRequest, authorization: str = Header("")):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "bad key")
    return {"response": orchestrator(req.message, patient_hint=req.patient)}

@app.post("/summarize")
def summarize(req: SummarizeRequest, authorization: str = Header("")):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "bad key")
    return {"response": orchestrator("Summarize: " + req.report, mode="summarize")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
