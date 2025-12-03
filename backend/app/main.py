# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
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
from enum import Enum

from app.models.database import SessionLocal, ScanResult
from .inference import run_inference_with_explain
from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener

register_heif_opener()

# --- NEW: File Type Enumeration ---
class FileType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY", "mysecretkey")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
DEFAULT_PORT = os.getenv("PORT", "7000")

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

# --- NEW: Frame cleanup directory ---
os.makedirs("video_frames", exist_ok=True)

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
async def save_upload_file(file: UploadFile) -> tuple[str, str]:
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    tmp_path = os.path.join(upload_dir, unique_filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return tmp_path, unique_filename

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def draw_label_on_image(image_path: str, label: str) -> str:
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font_size = max(20, img.size[0] // 20)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        text = f"Prediction: {label}"
        draw.rectangle([5, 5, 5 + len(text) * font_size // 2, 5 + font_size + 5], fill=(0, 0, 0, 128))
        draw.text((10, 10), text, fill="white", font=font)
        img.save(image_path)
    except Exception as e:
        print("Failed to draw label:", e)
    return image_path

def get_file_type(filename: str) -> Optional[FileType]:
    if not filename or "." not in filename:
        return None
    ext = filename.lower().split(".")[-1]
    if ext in ["jpg", "jpeg", "png", "heic", "heif", "hif"]:
        return FileType.IMAGE
    elif ext in ["mp4", "avi", "mov", "mkv"]:
        return FileType.VIDEO
    return None

def cleanup_files(temp_file_path: str, unique_filename: str, frames_dir: Optional[str] = None):
    try:
        print(f"Cleaning up: {temp_file_path}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        jpg_conversion = temp_file_path + ".jpg"
        if os.path.exists(jpg_conversion):
            os.remove(jpg_conversion)

        if frames_dir and os.path.exists(frames_dir):
            print(f"Cleaning frames: {frames_dir}")
            shutil.rmtree(frames_dir)
    except Exception as e:
        print("Cleanup error:", e)

def normalize_inference_output(out):
    if isinstance(out, dict):
        return (
            out.get("label", "unknown"),
            float(out.get("confidence", 0.0)),
            out.get("model_version", "unknown"),
            out.get("explanation") or out.get("explanation_path") or out.get("explain")
        )

    elif isinstance(out, (tuple, list)):
        padded = list(out) + [None] * (4 - len(out))
        label, conf, mv, exp = padded
        try:
            conf = float(conf)
        except:
            conf = 0.0
        return label, conf, mv or "unknown", exp

    return "unknown", 0.0, "unknown", None

# ---------------- ROUTES ----------------
@app.post("/scan", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def scan(background_tasks: BackgroundTasks, file: UploadFile = File(...), metadata: Optional[str] = Form(None)):
    ftype = get_file_type(file.filename)
    if ftype == FileType.IMAGE:
        return await predict(background_tasks, file=file, metadata=metadata)
    elif ftype == FileType.VIDEO:
        return await predict_video(background_tasks, file=file, metadata=metadata)
    else:
        raise HTTPException(400, "Unsupported file type")

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...), metadata: Optional[str] = Form(None)):

    try:
        meta = Metadata.parse_raw(metadata) if metadata else Metadata()
    except:
        raise HTTPException(400, "Invalid metadata JSON")

    tmp_path = None
    unique_filename = None
    try:
        tmp_path, unique_filename = await save_upload_file(file)

        # HEIC → JPG conversion
        ext = file.filename.lower().split(".")[-1]
        if ext in ["heic", "heif", "hif"]:
            img = Image.open(tmp_path)
            jpg_path = tmp_path + ".jpg"
            img.save(jpg_path, "JPEG")
            tmp_path = jpg_path

        start = time.time()

        raw_out = run_inference_with_explain(tmp_path)
        label, confidence, model_version, explanation_path = normalize_inference_output(raw_out)

        elapsed = int((time.time() - start) * 1000)

        if explanation_path:
            explanation_path = os.path.join("explanations", os.path.basename(str(explanation_path)))
            try:
                explanation_path = draw_label_on_image(explanation_path, label)
            except:
                pass
            host_ip = get_local_ip()
            explanation_path = f"http://{host_ip}:{DEFAULT_PORT}/explanations/{os.path.basename(explanation_path)}"
        else:
            explanation_path = None

        # Save to DB
        try:
            db = SessionLocal()
            row = ScanResult(
                filename=unique_filename,
                label=label,
                confidence=confidence,
                model_version=model_version,
                source=meta.source,
                user_id=meta.user_id,
            )
            db.add(row)
            db.commit()
            db.close()
        except Exception as e:
            print("DB warning:", e)

        return PredictionResponse(
            label=label,
            confidence=confidence,
            model_version=model_version,
            inference_time_ms=elapsed,
            saved_filename=unique_filename,
            explanation_path=explanation_path,
        )

    finally:
        if tmp_path and unique_filename:
            background_tasks.add_task(cleanup_files, tmp_path, unique_filename)

@app.post("/predict_video", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), metadata: Optional[str] = Form(None)):

    try:
        meta = Metadata.parse_raw(metadata) if metadata else Metadata()
    except:
        raise HTTPException(400, "Invalid metadata JSON")

    tmp_path = None
    frames_dir_path = None
    unique_filename = None

    try:
        import cv2

        tmp_path, unique_filename = await save_upload_file(file)

        frames_dir = f"video_frames/{uuid.uuid4()}"
        frames_dir_path = frames_dir
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(500, "Could not open video file")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
        frame_interval = fps

        frame_count = 0
        results = []

        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(frames_dir, f"{uuid.uuid4()}.jpg")
                cv2.imwrite(frame_path, frame)

                try:
                    raw_out = run_inference_with_explain(frame_path)
                    label, confidence, model_version, explanation = normalize_inference_output(raw_out)

                    # store ALL FOUR values (this was the bug)
                    results.append((label, confidence, model_version, explanation))
                except Exception as e:
                    print("Frame error:", e)

            frame_count += 1

        cap.release()

        if not results:
            raise HTTPException(500, "No valid frames extracted")

        # majority vote
        from collections import Counter
        final_label = Counter([r[0] for r in results]).most_common(1)[0][0]
        avg_conf = sum([r[1] for r in results]) / len(results)
        final_model_version = results[0][2]  # take the model version from first frame

        explanation_path = results[len(results)//2][3]

        elapsed = int((time.time() - start) * 1000)

        if explanation_path:
            explanation_path = os.path.join("explanations", os.path.basename(str(explanation_path)))
            try:
                explanation_path = draw_label_on_image(explanation_path, final_label)
            except:
                pass
            host_ip = get_local_ip()
            explanation_path = f"http://{host_ip}:{DEFAULT_PORT}/explanations/{os.path.basename(explanation_path)}"

        # save DB
        try:
            db = SessionLocal()
            row = ScanResult(
                filename=unique_filename,
                label=final_label,
                confidence=avg_conf,
                model_version=final_model_version,
                source=meta.source,
                user_id=meta.user_id,
            )
            db.add(row)
            db.commit()
            db.close()
        except Exception as e:
            print("DB warning:", e)

        return PredictionResponse(
            label=final_label,
            confidence=avg_conf,
            model_version=final_model_version,
            inference_time_ms=elapsed,
            saved_filename=unique_filename,
            explanation_path=explanation_path,
        )

    finally:
        if tmp_path and unique_filename:
            background_tasks.add_task(cleanup_files, tmp_path, unique_filename, frames_dir_path)

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_version": "v1.0.0"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(DEFAULT_PORT), reload=True)
