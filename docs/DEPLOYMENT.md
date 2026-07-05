# Deployment

The system is two independently deployable services: the FastAPI recommender
(stateless, serves from pre-trained artifacts) and the Streamlit storefront
demo (thin client that calls the API).

## 1. API (Render, or any Dockerfile host)

The `artifacts/` and `data/processed/{articles,transactions_train}.parquet`
files needed at serving time are committed to the repo (see `.gitignore`),
so the image builds straight from a fresh clone — no local training run
required first.

Build and run locally to sanity-check before deploying:

```bash
docker build -t shelfsense-api .
docker run -p 8000:8000 shelfsense-api
curl http://localhost:8000/health
```

**Render** (used for the live deploy):
1. New → Web Service → connect the GitHub repo → Render auto-detects the
   Dockerfile. Leave build/start commands blank.
2. Instance type: Free.
3. Create Web Service. Render assigns a URL immediately (e.g.
   `https://shelfsense-<random>.onrender.com`) even before the first build
   finishes — copy it, you'll need it for the frontend.
4. Free-tier instances spin down after inactivity; the first request after
   idle can take ~50 seconds to wake back up. Not a bug.

Fly.io and Railway also support "deploy from Dockerfile" with a free tier if
preferred — same Dockerfile, no changes needed.

**Known build failure and fix**: `implicit` (the ALS library) has no
prebuilt Linux wheel for this Python version and compiles from source at
install time. The `python:3.12-slim` base image has no C compiler, so the
first deploy failed with `Failed to build installable wheels for implicit`.
Fixed by installing `build-essential` before `pip install` (see
`Dockerfile`) — verified with a full local `docker build` + `docker run` +
live API calls against the container before trusting it on Render.

## 2. Frontend (Streamlit Community Cloud)

1. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub →
   New app.
2. Repo: this one. Branch: `main`. Main file path:
   `frontend/streamlit_app.py`.
3. **Advanced settings** → Python version: set to **3.12** (matches what
   this project is built and tested on — a newer default like 3.14 risks
   the same "no prebuilt wheel yet" failure the API hit on Render).
4. **Advanced settings** → Secrets — paste exactly:
   ```toml
   API_URL = "https://<your-render-url>"
   ```
   (the placeholder text already in that box — `DB_USERNAME`, `some_key`,
   etc. — is Streamlit's generic example; delete it and replace with the
   line above.)
5. Deploy.

## Environment variables

| Variable | Used by | Default |
|---|---|---|
| `API_URL` | Streamlit frontend | `http://localhost:8000` |

## Verifying a live deployment

```bash
curl https://<render-url>/health
curl "https://<render-url>/recommend/<any-customer-id>?model=hybrid&k=3"
```
Then open the Streamlit URL and confirm the customer dropdown populates and
recommendations render — that round-trips through the deployed API, not
just confirms the frontend loaded.
