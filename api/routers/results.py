"""
Results and metadata endpoints for IndustrialSentinel API.
"""
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pathlib import Path

from src.config import MODEL_DIR, METADATA_FILE

router = APIRouter()


@router.get("/results")
def get_results() -> dict:
    """
    Return the full metrics dictionary from metadata.json.

    Returns:
        API response with all training metrics.
    """
    metadata_path = MODEL_DIR / METADATA_FILE
    if not metadata_path.exists():
        raise HTTPException(status_code=500, detail="Metadata file not found. Run training first.")

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid metadata JSON: {e}")

    return {
        "status": "ok",
        "data": metadata,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metadata")
def get_metadata() -> dict:
    """
    Return model metadata including artifact paths and hyperparameters.

    Returns:
        API response with metadata.
    """
    return get_results()
