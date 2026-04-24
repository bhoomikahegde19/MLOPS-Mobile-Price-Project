from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from pipeline.config import REPORTS_DIR


router = APIRouter()


@router.get("/stream")
def stream_metrics():
    summary_path = REPORTS_DIR / "automl_summary.json"

    def event_stream():
        payload = {"message": "No training summary available yet."}
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        yield json.dumps(payload)

    return StreamingResponse(event_stream(), media_type="application/json")
