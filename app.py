import os
import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
from openai import OpenAI

# Initialize app
app = FastAPI(
    title="HART Evaluation API",
    description="Backend service for evaluating patient intake forms with AI",
    version="1.0.0",
    swagger_ui_init_oauth={"usePkceWithAuthorizationCodeGrant": True}
)

# Security: Bearer scheme for Swagger Authorize button
bearer_scheme = HTTPBearer()

# CORS (frontend <-> backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security token
API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Not authenticated")
    return True

# Pydantic models
class IntakeForm(BaseModel):
    name: str
    age: Union[int, str]
    gender: Optional[str] = None
    symptoms: List[str]
    history: Optional[str] = None
    medications: Optional[str] = None

class EvaluationResult(BaseModel):
    chief_complaint: str
    history_summary: str
    risk_flags: Dict[str, str]
    recommended_followups: List[str]
    differential_considerations: List[str]
    patient_friendly_summary: str
    emergency_guidance: str

# OpenAI client
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
client = OpenAI(api_key=OPENAI_KEY)

@app.post("/evaluate", response_model=EvaluationResult, dependencies=[Depends(verify_token)])
async def evaluate_patient(data: IntakeForm):
    """
    Evaluate patient intake form using OpenAI GPT
    """
    try:
        prompt = f"""
        You are a medical AI assistant. Analyze the following intake:

        Name: {data.name}
        Age: {data.age}
        Gender: {data.gender}
        Symptoms: {", ".join(data.symptoms)}
        History: {data.history}
        Medications: {data.medications}

        Provide a structured analysis in JSON with keys:
        - chief_complaint
        - history_summary
        - risk_flags (dict, but keep values as strings not booleans)
        - recommended_followups (list)
        - differential_considerations (list)
        - patient_friendly_summary
        - emergency_guidance
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        ai_content = response.choices[0].message.content
        evaluation = json.loads(ai_content)

        # --- Normalization step ---
        def normalize(val):
            if isinstance(val, bool):
                return "Yes" if val else "No"
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, dict):
                return json.dumps(val)
            if isinstance(val, list):
                return ", ".join(map(str, val))
            return str(val)

        evaluation["history_summary"] = str(evaluation.get("history_summary", ""))

        if "risk_flags" in evaluation:
            rf = evaluation["risk_flags"]
            if isinstance(rf, dict):
                evaluation["risk_flags"] = {k: normalize(v) for k, v in rf.items()}
            elif isinstance(rf, list):
                evaluation["risk_flags"] = {f"flag_{i+1}": normalize(v) for i, v in enumerate(rf)}
            else:
                evaluation["risk_flags"] = {"note": normalize(rf)}

        return evaluation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
