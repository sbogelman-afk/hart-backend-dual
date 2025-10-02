import os
import json
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Union
from openai import OpenAI

# Initialize app with explicit OpenAPI security scheme
app = FastAPI(
    title="HART Evaluation API",
    description="Backend service for evaluating patient intake forms with AI",
    version="1.0.0",
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True
    }
)

# Define Bearer security for Swagger Authorize button
bearer_scheme = HTTPBearer()

# CORS (so frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security: simple bearer token
API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Not authenticated")
    return True

# Pydantic models
class IntakeForm(BaseModel):
    name: str
    age: Union[int, str]  # accepts number or string
    gender: Optional[str] = None
    symptoms: List[str]
    history: Optional[str] = None
    medications: Optional[str] = None

class EvaluationResult(BaseModel):
    chief_complaint: str
    history_summary: str
    risk_flags: dict
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
        - risk_flags (dict)
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

        return evaluation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
