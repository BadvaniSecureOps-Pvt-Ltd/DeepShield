# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import time
import os
import shutil
import uuid
import uvicorn
import socket

from app.models.database import SessionLocal, ScanResult
from .inference import run_inference_with_explain
from PIL import Image, ImageDraw, ImageFont

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

# --- Serve explanation images ---
os.makedirs("explanations", exist_ok=True)
app.mount("/explanations", StaticFiles(directory="explanations"), name="explanations")

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

def get_local_ip():
    """Return the LAN IP of the current machine (eth0)."""
    try:
        # Connect to an external host; the IP used will be the LAN IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def draw_label_on_image(image_path: str, label: str) -> str:
    """Draw label text on Grad-CAM snapshot and save."""
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font_size = max(20, img.size[0] // 20)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        text = f"Prediction: {label}"
        draw.rectangle([5, 5, 5+len(text)*font_size//2, 5+font_size+5], fill=(0,0,0,128))
        draw.text((10, 10), text, fill="white", font=font)
        img.save(image_path)
    except Exception as e:
        print("Failed to draw label on image:", e)
    return image_path

# ---------------- ROUTES ----------------
@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(...), metadata: Optional[str] = Form(None)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        meta = Metadata.parse_raw(metadata) if metadata else Metadata()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    try:
        tmp_path, unique_filename = await save_upload_file(file)
        start = time.time()

        # Grad-CAM inference
        label, confidence, model_version, explanation_path = run_inference_with_explain(tmp_path)
        elapsed = int((time.time() - start) * 1000)

        if explanation_path:
            # Ensure single .jpg
            explanation_path = os.path.join("explanations", os.path.basename(explanation_path))
            # Draw label on image
            explanation_path = draw_label_on_image(explanation_path, label)
            # Convert to frontend-friendly URL
            host_ip = get_local_ip()
            explanation_path = f"http://{host_ip}:8000/explanations/{os.path.basename(explanation_path)}"

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

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_version": "v1.0.0"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
