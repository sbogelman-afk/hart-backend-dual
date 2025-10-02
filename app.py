from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import openai

# Initialize app
app = FastAPI(title="HART Evaluation API")

# Security: require Bearer token
security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN")

# Restrict CORS to Netlify frontend only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hartintake.netlify.app"],  # ✅ restrict here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request model
class IntakeData(BaseModel):
    reportId: str
    language: str = "en"
    answers: dict

# Protect all routes with API token
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Root (just a sanity check, not exposed in Swagger)
@app.get("/")
def root():
    return {"message": "HART backend is running"}

# Evaluation route
@app.post("/evaluate")
async def evaluate(data: IntakeData, creds: HTTPAuthorizationCredentials = Depends(verify_token)):
    try:
        # Generate structured evaluation with OpenAI
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        user_text = f"""
        Patient Report ID: {data.reportId}
        Language: {data.language}
        Intake Answers: {data.answers}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical assistant that summarizes patient intake forms into structured evaluations."},
                {"role": "user", "content": user_text}
            ]
        )

        ai_text = response.choices[0].message.content

        # Return structured response
        return {
            "evaluation": {
                "chief_complaint": data.answers.get("reason", "N/A"),
                "history_summary": ai_text,
                "risk_flags": {
                    "age": data.answers.get("age", "N/A"),
                    "symptoms": data.answers.get("symptoms", []),
                },
                "recommended_followups": [
                    "Schedule primary care follow-up",
                    "Run diagnostic labs if indicated"
                ],
                "patient_friendly_summary": "Your intake has been reviewed. Follow the doctor’s instructions and complete recommended tests.",
                "emergency_guidance": "If you experience chest pain, fainting, or severe shortness of breath, call 911 immediately."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
