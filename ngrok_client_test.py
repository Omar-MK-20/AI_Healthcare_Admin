# Run LOCAL against your Ngrok backend (NGROK_Send_Prompt_Local)
# YOU run this yourself after starting backend_api.py + ngrok.
import requests

URL = "ADD_NGROK_LINK_HERE/book"   # e.g. https://xxxx.ngrok-free.dev/book
headers = {"Authorization": "Bearer secret123"}

payload = {
    "message": "Book General Checkup with Dr. Ahmed Hassan for P-1001 on 2026-10-12 at 10:00",
    "patient": "P-1001",
}

res = requests.post(URL, headers=headers, json=payload, timeout=300)
print(res.json()["response"])
