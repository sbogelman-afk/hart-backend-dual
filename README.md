# HART Evaluation API (Dual Provider)

This repo provides a FastAPI backend for evaluating structured patient intake forms using either:

- **Google Vertex AI (Gemini)** — HIPAA-ready path
- **OpenAI GPT-4o-mini** — quick prototyping, not HIPAA-compliant for PHI

## Files
- `app.py` — FastAPI app with `/evaluate`
- `requirements.txt` — dependencies
- `README.md` — this file

## Run locally

```bash
pip install -r requirements.txt
export API_TOKEN="yoursecrettoken"
export ALLOWED_ORIGINS="http://localhost:3000"

# OpenAI mode
export PROVIDER=openai
export OPENAI_API_KEY=sk-...
uvicorn app:app --reload

# Vertex mode
export PROVIDER=vertex
export GCP_PROJECT=your-project-id
export GCP_LOCATION=us-central1
export MODEL_NAME=gemini-1.5-flash
uvicorn app:app --reload
```

## Deploy to Google Cloud Run

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud run deploy hart-eval --source . --region us-central1   --set-env-vars "PROVIDER=vertex,GCP_PROJECT=YOUR_PROJECT_ID,GCP_LOCATION=us-central1,MODEL_NAME=gemini-1.5-flash,API_TOKEN=yoursecrettoken,ALLOWED_ORIGINS=https://your-netlify-site.netlify.app"
```

## Deploy to other PaaS (for OpenAI)
- Zip this repo and upload to **Railway**, **Render**, or **Heroku**.
- Set env vars (API_TOKEN, ALLOWED_ORIGINS, PROVIDER=openai, OPENAI_API_KEY).
