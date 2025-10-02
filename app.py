from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import os

# Security scheme
security = HTTPBearer()

app = FastAPI(
    title="HART Evaluation API",
    description="API backend for patient intake and AI evaluation",
    version="1.0.0",
    openapi_tags=[{"name": "Evaluation", "description": "Endpoints for AI evaluation"}],
)

# Token from Heroku Config Vars
API_TOKEN = os.getenv("API_TOKEN", "hart-backend-secret-2025")

# Validation
def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )
    return credentials

# Intake form schema
class IntakeForm(BaseModel):
    name: str
    age: str
    gender: str
    symptoms: list[str]
    medications: str
    history: str

# Endpoint
@app.post("/evaluate", dependencies=[Depends(validate_token)], tags=["Evaluation"])
async def evaluate(form: IntakeForm):
    return {"message": f"Evaluation received for {form.name}"}


# ---- Custom OpenAPI to pre-fill Swagger Authorize with token ----
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

    # Global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]

    # Inject default value so Swagger pre-fills it
    openapi_schema["components"]["securitySchemes"]["BearerAuth"]["x-tokenDefault"] = API_TOKEN

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
