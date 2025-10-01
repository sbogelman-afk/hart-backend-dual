from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import os, uuid, time, json
from openai import OpenAI
from fastapi.openapi.utils import get_openapi

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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = api_key.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return token

# Request/Response Models
class EvaluateRequest(BaseModel):
    reportId: str
    language: str
    answers: dict

class EvaluateResponse(BaseModel):
    traceId: str
    evaluation: dict

# Routes
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest, api_key: str = Depends(get_api_key)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    client = OpenAI(api_key=OPENAI_API_KEY)

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
            messages=[
                {"role": "system", "content": "You are a helpful medical evaluation assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # Raw AI response (string)
        raw = resp.choices[0].message.content

        # Convert JSON string → dict
        evaluation = json.loads(raw)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {"traceId": str(uuid.uuid4()), "evaluation": evaluation}

# Force OpenAPI to show Authorize button
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"APIKeyAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
