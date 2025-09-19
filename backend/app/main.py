from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import time
import uvicorn
import os

from .inference import run_inference

# --- Config ---
API_KEY = os.getenv("API_KEY", "mysecretkey")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(title="DeepShield API", version="1.0.0")

# --- Auth dependency ---
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# --- Request/Response Models ---
class Metadata(BaseModel):
    source: Optional[str] = None
    user_id: Optional[str] = None

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    inference_time_ms: int

# --- Routes ---
@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(...), metadata: Optional[str] = Form(None)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        meta = Metadata.parse_raw(metadata) if metadata else Metadata()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    contents = await file.read()
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    start = time.time()
    label, confidence, model_version = run_inference(tmp_path)
    elapsed = int((time.time() - start) * 1000)

    return PredictionResponse(
        label=label,
        confidence=confidence,
        model_version=model_version,
        inference_time_ms=elapsed,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": "v1.0.0"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
