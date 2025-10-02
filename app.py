from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import openai
import traceback

app = FastAPI()

# Security: simple token check
API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")

def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Not authenticated")
    return True

# Flexible IntakeForm: all fields optional
class IntakeForm(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    symptoms: Optional[List[str]] = []
    medications: Optional[str] = None
    history: Optional[str] = None

@app.post("/evaluate")
async def evaluate(data: IntakeForm, authorized: bool = Depends(verify_token)):
    """
    Take patient intake data, send to OpenAI, return structured evaluation.
    """
    try:
        prompt = f"""
        Patient intake information:
        Name: {data.name}
        Age: {data.age}
        Gender: {data.gender}
        Symptoms: {", ".join(data.symptoms) if data.symptoms else "None"}
        Medications: {data.medications}
        History: {data.history}

        Please provide:
        - Chief complaint
        - Summary
        - Risk flags
        - Recommended followups
        - Differential considerations
        - Patient-friendly explanation
        - Emergency guidance
        """

        # Debug log: show we are about to call OpenAI
        print("DEBUG: Sending prompt to OpenAI...")

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        # Debug log: show raw response keys
        print("DEBUG: OpenAI response keys:", response.__dict__.keys())

        evaluation = response.choices[0].message.content.strip()

        # Debug log: show partial evaluation
        print("DEBUG: Evaluation preview:", evaluation[:200])

        return {"evaluation": evaluation}

    except Exception as e:
        print("ERROR: Exception during evaluation")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
