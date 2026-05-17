# Hugging Face Spaces (Docker SDK). Public app must listen on 7860.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_URL=http://127.0.0.1:8000

# libgomp1 is required by xgboost (OpenMP runtime)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as non-root uid 1000
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY dashboard ./dashboard
COPY models ./models
COPY .streamlit ./.streamlit
COPY start.sh .
RUN chmod +x start.sh && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860
CMD ["./start.sh"]
