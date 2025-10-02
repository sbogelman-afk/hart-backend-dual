import os
import traceback
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from openai import OpenAI

# ---------- App ----------
app = FastAPI(
    title="HART Evaluation API",
    description="Backend for patient intake + AI evaluation",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},  # keep auth active in UI
)

# CORS (tighten later to your Netlify domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://hartintake.netlify.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Security (Bearer) ----------
bearer = HTTPBearer(auto_error=True)
API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not credentials or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated"
        )
    return True

# Add Bearer security to OpenAPI so the *Authorize* button appears
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
    }
    # Apply globally so the padlock shows at the top
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ---------- Model (flexible for testing) ----------
class IntakeForm(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    symptoms: Optional[List[str]] = []   # checkboxes -> array
    medications: Optional[str] = None
    history: Optional[str] = None

# ---------- OpenAI client (new SDK) ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Routes ----------
@app.get("/")
def health():
    return {"status": "ok", "service": "HART backend"}

@app.post("/evaluate", dependencies=[Depends(validate_token)], tags=["Evaluation"])
async def evaluate(form: IntakeForm):
    """
    Take patient intake data, send to OpenAI, return structured evaluation.
    """
    try:
        prompt = f"""
        Patient intake information:
        Name: {form.name}
        Age: {form.age}
        Gender: {form.gender}
        Symptoms: {", ".join(form.symptoms) if form.symptoms else "None"}
        Medications: {form.medications}
        History: {form.history}

        Please provide a structured analysis with:
        - Chief complaint
        - Summary
        - Risk flags
        - Recommended followups
        - Differential considerations
        - Patient-friendly explanation
        - Emergency guidance (if needed)
        """

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        evaluation = resp.choices[0].message.content.strip()
        return {"evaluation": evaluation}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
