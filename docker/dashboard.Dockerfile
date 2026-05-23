FROM python:3.11-slim AS builder

WORKDIR /app
RUN pip install --no-cache-dir streamlit requests pandas pyarrow

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY dashboard/ dashboard/

ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/models
ENV API_URL=http://api:8000

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
