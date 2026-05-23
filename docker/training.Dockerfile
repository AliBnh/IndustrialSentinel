FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/
COPY train.py .

ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/models
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "train.py"]
