from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import openai

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
        # Build prompt for AI
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

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        evaluation = response.choices[0].message.content.strip()

        return {"evaluation": evaluation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
