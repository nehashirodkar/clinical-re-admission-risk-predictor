#!/usr/bin/env bash
set -e

# FastAPI on an internal port; Streamlit (public) calls it via API_URL.
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

exec streamlit run dashboard/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
