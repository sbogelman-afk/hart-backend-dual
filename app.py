from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import os

app = FastAPI(
    title="HART Evaluation API",
    description="API backend for patient intake and AI evaluation",
    version="1.0.0",
)

# Read token from environment
API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")

# Security scheme
bearer_scheme = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials is None or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated"
        )
    return True

# Intake schema
class IntakeForm(BaseModel):
    name: str
    age: str
    gender: str
    symptoms: list[str]
    medications: str
    history: str

@app.post("/evaluate", tags=["Evaluation"])
async def evaluate(form: IntakeForm, authorized: bool = Depends(validate_token)):
    return {"message": f"Evaluation received for {form.name}"}

# Custom OpenAPI to prefill Swagger with token
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    # Default security applied
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
