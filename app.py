from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from openai import OpenAI
import traceback

app = FastAPI()

API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")

def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Not authenticated")
    return True

class IntakeForm(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    symptoms: Optional[List[str]] = []
    medications: Optional[str] = None
    history: Optional[str] = None

@app.post("/evaluate")
async def evaluate(data: IntakeForm, authorized: bool = Depends(verify_token)):
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

        # ✅ Fixed: no "proxies"
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return {"evaluation": response.choices[0].message.content.strip()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
