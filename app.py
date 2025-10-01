from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid, time

API_TOKEN = os.getenv("API_TOKEN", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
PROVIDER = os.getenv("PROVIDER", "openai")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluateRequest(BaseModel):
    reportId: str
    language: str
    answers: dict

class EvaluateResponse(BaseModel):
    traceId: str
    evaluation: dict

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest, request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer ") or auth.split(" ")[1] != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    evaluation = {
        "chief_complaint": req.answers.get("reason", "Unknown"),
        "history_summary": f"Patient {req.answers.get('patient_name', '?')} reports {req.answers.get('reason', '')}",
        "risk_flags": ["Demo only"],
        "recommended_followups": ["Demo follow-up"],
        "differential_considerations": ["Demo differential"],
        "patient_friendly_summary": "This is just a demo response until AI is connected.",
        "emergency_guidance": ""
    }

    return {"traceId": str(uuid.uuid4()), "evaluation": evaluation}

