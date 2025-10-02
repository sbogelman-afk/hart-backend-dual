from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI(title="HART Evaluation API")

# --- Security ---
security = HTTPBearer()  
API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")  # fallback if env not set

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Not authenticated")
    return credentials

# --- Data Model ---
class IntakeForm(BaseModel):
    name: str
    age: int
    gender: str
    symptoms: list[str] = []
    medical_history: str | None = None

# --- Routes ---
@app.get("/")
def root():
    return {"message": "HART backend running. Use /docs for API documentation."}

@app.post("/evaluate")
def evaluate(form: IntakeForm, creds: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant evaluating intake forms."},
                {"role": "user", "content": f"Evaluate this patient form: {form.model_dump_json()}"}
            ],
            max_tokens=300
        )
        return {"evaluation": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
