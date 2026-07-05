# Deployment

The system is two independently deployable services: the FastAPI recommender
(stateless, serves from pre-trained artifacts) and the Streamlit storefront
demo (thin client that calls the API).

## 1. API (FastAPI + Docker)

Build and run locally:

```bash
docker build -t shelfsense-api .
docker run -p 8000:8000 shelfsense-api
curl http://localhost:8000/health
```

Deploy the image to any container host — Render, Fly.io, and Railway all
support "deploy from Dockerfile" with a free tier:

- **Render**: New -> Web Service -> connect the repo -> it detects the
  Dockerfile automatically. Set the port to 8000.
- **Fly.io**: `fly launch` from the project root (it finds the Dockerfile),
  then `fly deploy`.

The `artifacts/` and `data/processed/` directories must exist before building
the image — run `python -m shelfsense.build_features` then
`python -m shelfsense.train` locally first (see main README), commit the
small artifact files, or add a build step that runs training in CI.

## 2. Frontend (Streamlit)

Same pattern as `Portfolio_Risk_Platform`: push `frontend/streamlit_app.py`
to Streamlit Community Cloud. Set the `API_URL` secret/env var to the
deployed API's public URL (e.g. `https://shelfsense-api.onrender.com`).

## Environment variables

| Variable | Used by | Default |
|---|---|---|
| `API_URL` | Streamlit frontend | `http://localhost:8000` |
