import os
import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
from openai import OpenAI

# Initialize app with OpenAPI security scheme
app = FastAPI(
    title="HART Evaluation API",
    description="Backend service for evaluating patient intake forms with AI",
    version="1.0.0"
)

# Define Bearer security for Swagger Authorize button
bearer_scheme = HTTPBearer()

# CORS (so frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to frontend domain later
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
    lifestyle: Optional[Dict[str, str]] = None  # smoking, alcohol

class EvaluationResult(BaseModel):
    chief_complaint: str
    history_summary: str
    risk_flags: Dict[str, str]
    recommended_followups: List[str]
    differential_considerations: List[str]
    patient_friendly_summary: str
    emergency_guidance: str
    formatted_report: str

# OpenAI client
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
client = OpenAI(api_key=OPENAI_KEY)

# Formatter for polished report
def format_report(evaluation: dict, patient: IntakeForm) -> str:
    """Format evaluation JSON into a polished report string"""
    report = f"""
    ======================================
              Patient Evaluation Report
    ======================================

    Patient: {patient.name}
    Age: {patient.age}
    Gender: {patient.gender or "Not specified"}

    --------------------------------------
    Chief Complaint
    --------------------------------------
    {evaluation.get('chief_complaint', 'N/A')}

    --------------------------------------
    History Summary
    --------------------------------------
    {evaluation.get('history_summary', 'N/A')}

    --------------------------------------
    Risk Flags
    --------------------------------------
    """
    for key, value in evaluation.get("risk_flags", {}).items():
        report += f"- {key}: {value}\n"

    # FIXED here: no backslashes in f-string braces
    report += "\n\n--------------------------------------\nRecommended Follow-ups\n--------------------------------------\n"
    report += "".join([f"- {item}\n" for item in evaluation.get("recommended_followups", [])])

    report += "\n\n--------------------------------------\nDifferential Considerations\n--------------------------------------\n"
    report += "".join([f"- {item}\n" for item in evaluation.get("differential_considerations", [])])

    report += f"""

    --------------------------------------
    Patient-Friendly Summary
    --------------------------------------
    {evaluation.get('patient_friendly_summary', 'N/A')}

    --------------------------------------
    Emergency Guidance
    --------------------------------------
    🚨 {evaluation.get('emergency_guidance', 'N/A')} 🚨
    """

    return report.strip()


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
        Lifestyle: {data.lifestyle}

        Provide a structured analysis in JSON with keys:
        - chief_complaint (string)
        - history_summary (string)
        - risk_flags (dictionary with string values only)
        - recommended_followups (list of strings)
        - differential_considerations (list of strings)
        - patient_friendly_summary (string)
        - emergency_guidance (string)
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        ai_content = response.choices[0].message.content
        evaluation = json.loads(ai_content)

        # Ensure risk_flags are strings
        evaluation["risk_flags"] = {
            k: str(v) for k, v in evaluation.get("risk_flags", {}).items()
        }

        # Format report
        evaluation["formatted_report"] = format_report(evaluation, data)

        return evaluation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

@app.post("/export-pdf", dependencies=[Depends(verify_token)])
async def export_pdf(data: EvaluationResult):
    """
    Export the evaluation result as a formatted PDF.
    """
    try:
        # Create a temporary file for PDF
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(tmp_file.name, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []

        # Title
        flowables.append(Paragraph("❤️ HART Patient Evaluation Report", styles["Title"]))
        flowables.append(Spacer(1, 12))

        # Sections
        flowables.append(Paragraph("<b>Chief Complaint</b>", styles["Heading2"]))
        flowables.append(Paragraph(data.chief_complaint, styles["Normal"]))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("<b>History Summary</b>", styles["Heading2"]))
        flowables.append(Paragraph(data.history_summary, styles["Normal"]))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("<b>Risk Flags</b>", styles["Heading2"]))
        for k, v in data.risk_flags.items():
            flowables.append(Paragraph(f"- {k}: {v}", styles["Normal"]))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("<b>Recommended Follow-ups</b>", styles["Heading2"]))
        for item in data.recommended_followups:
            flowables.append(Paragraph(f"- {item}", styles["Normal"]))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("<b>Differential Considerations</b>", styles["Heading2"]))
        for item in data.differential_considerations:
            flowables.append(Paragraph(f"- {item}", styles["Normal"]))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("<b>Patient-Friendly Summary</b>", styles["Heading2"]))
        flowables.append(Paragraph(data.patient_friendly_summary, styles["Normal"]))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("<b>Emergency Guidance</b>", styles["Heading2"]))
        flowables.append(Paragraph(f"🚨 {data.emergency_guidance} 🚨", styles["Normal"]))

        # Build PDF
        doc.build(flowables)

        return FileResponse(tmp_file.name, media_type="application/pdf", filename="HART_Report.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")
