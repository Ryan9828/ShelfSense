FROM python:3.12-slim

WORKDIR /srv

# implicit (ALS) has no prebuilt Linux wheel for this Python version and compiles
# from source — needs a C/C++ compiler, which the slim base image doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
COPY data/processed/ data/processed/

ENV PYTHONPATH=/srv/src
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
