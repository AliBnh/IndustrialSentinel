"""
FastAPI application for IndustrialSentinel predictive maintenance API.
"""
import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import RequestLoggingMiddleware
from api.routers import health, predict, results, drift

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = FastAPI(
    title="IndustrialSentinel API",
    description="Predictive maintenance intelligence for industrial rotating equipment",
    version="1.0.0",
)

# Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    logging.getLogger("api").warning("prometheus-fastapi-instrumentator not installed, /metrics disabled")

# Middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(results.router, tags=["Results"])
app.include_router(drift.router, tags=["Drift"])
