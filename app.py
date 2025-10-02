import os
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import openai

# --- FastAPI app ---
app = FastAPI()

# --- CORS (allow frontend on Netlify to call backend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your Netlify domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication (Bearer token) ---
API_TOKEN = os.getenv("API_TOKEN")

def authenticate(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return token

# --- IntakeForm schema ---
class IntakeForm(BaseModel):
    chief_complaint: Optional[str] = None
    history: Optional[str] = None
    medications: Optional[str] = None
    allergies: Optional[str] = None
    chest_pain: Optional[str] = None
    palpitations: Optional[str] = None
    shortness_breath: Optional[str] = None
    fainting: Optional[str] = None

    # New fields
    risk_flags: Optional[str] = None         # emergency red flags
    general_symptoms: Optional[str] = None   # general complaints

# --- OpenAI config ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Routes ---
@app.get("/")
def home():
    return {"message": "HART Backend Running"}

@app.post("/evaluate")
async def evaluate(data: IntakeForm, token: str = Depends(authenticate)):
    answers = data.dict()
    # remove None values
    clean_answers = {k: v for k, v in answers.items() if v is not None}

    # Build a simple evaluation prompt
    prompt = f"""
    You are a medical AI assistant. Evaluate this intake form data and provide:
    - Chief complaint
    - History summary
    - Notable risk flags
    - Recommended follow-ups
    - Possible differential considerations
    - A patient-friendly summary
    - Emergency guidance if needed

    Intake data:
    {clean_answers}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # can change to gpt-4o or gpt-4.1 if available
            messages=[
                {"role": "system", "content": "You are a careful medical assistant helping to interpret intake forms."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        evaluation = response.choices[0].message.content.strip()

        return {
            "evaluation": evaluation,
            "submitted_data": clean_answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
