from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel
import os, uuid, time
from openai import OpenAI

# --- Config ---
API_TOKEN = os.getenv("API_TOKEN", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI(
    title="HART Evaluation API",
    description="API to evaluate patient intake forms using AI.",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)) -> APIKey:
    if not api_key_header or not api_key_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = api_key_header.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return token

# --- Models ---
class EvaluateRequest(BaseModel):
    reportId: str
    language: str
    answers: dict

class EvaluateResponse(BaseModel):
    traceId: str
    evaluation: dict

# --- Routes ---
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest, api_key: APIKey = Depends(get_api_key)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Create a structured prompt
    prompt = f"""
    You are a medical AI assistant helping interpret structured intake data.
    Summarize the case clearly and return JSON with these fields:
    - chief_complaint
    - history_summary
    - risk_flags
    - recommended_followups
    - differential_considerations
    - patient_friendly_summary
    - emergency_guidance

    Patient intake data:
    {req.answers}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful medical evaluation assistant."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        evaluation = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {"traceId": str(uuid.uuid4()), "evaluation": evaluation}
