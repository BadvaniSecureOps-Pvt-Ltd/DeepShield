# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import os
import shutil
import uuid
import uvicorn

from app.models.database import SessionLocal, ScanResult
from .inference import run_inference
from PIL import Image
import io

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY", "mysecretkey")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(title="DeepShield API", version="1.0.0")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- AUTH ----------------
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# ---------------- SCHEMAS ----------------
class Metadata(BaseModel):
    source: Optional[str] = None
    user_id: Optional[str] = None

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    inference_time_ms: int
    saved_filename: str
    explanation_path: Optional[str] = None

# ---------------- HELPER ----------------
async def save_upload_file(file: UploadFile) -> str:
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    tmp_path = os.path.join(upload_dir, unique_filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return tmp_path, unique_filename

# ---------------- ROUTES ----------------
@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(...), metadata: Optional[str] = Form(None)):

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".pdf", ".heic")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        meta = Metadata.parse_raw(metadata) if metadata else Metadata()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    try:
        tmp_path, unique_filename = await save_upload_file(file)
        start = time.time()
        label, confidence, model_version, explanation_path = run_inference(tmp_path)
        elapsed = int((time.time() - start) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Save results to DB
    try:
        db = SessionLocal()
        db_result = ScanResult(
            filename=unique_filename,
            label=label,
            confidence=confidence,
            model_version=model_version,
            source=meta.source,
            user_id=meta.user_id,
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        db.close()
    except Exception:
        pass

    return PredictionResponse(
        label=label,
        confidence=confidence,
        model_version=model_version,
        inference_time_ms=elapsed,
        saved_filename=unique_filename,
        explanation_path=explanation_path,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": "v1.0.0"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
