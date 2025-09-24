DeepShield — Backend README 
Project Overview
DeepShield Backend is the AI-powered detection service for identifying deepfake or fake content. It provides REST APIs for the mobile app to upload images/videos and returns detection results with confidence scores and explainability.
Core responsibilities:
•	Handle file uploads (images, videos, PDFs → extracted frames).
•	Run inference using ML models (app/models/model.py).
•	Store results in SQLite (results.db) for auditability.
•	Provide APIs to frontend/mobile app.
________________________________________
 Setup Instructions
1. Clone the Repository
git clone https://github.com/bittubadwani/DeepShield.git
cd DeepShield/backend
2. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
3. Install Requirements
pip install -r requirements.txt
4. Environment Variables
Create a file .env in /backend:
SECRET_KEY=########
DB_URL=sqlite:///results.db
MODEL_PATH=app/models/weights/deepfake_detecer.pth
5. Run the Backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Server will be live at → http://127.0.0.1:8000
________________________________________
API Usage
Health Check
curl http://127.0.0.1:8000/health
Response:
{"status": "ok"}
Predict Endpoint
curl -X POST http://127.0.0.1:8000/predict \
     -F "file=@sample_real.jpg"
Response:
{
  "result": "real",
  "confidence": 0.92,
  "explanation": "subtle artifacts checked"
}
Postman Collection
Import postman_collection.json in Postman → Run /predict with sample images.
________________________________________
 Git Workflow
Create Backend Branch
git checkout -b backend-api
Commit & Push
git add .
git commit -m "feat(backend): add predict endpoint"
git push origin backend-api
Pull Updates
git pull origin backend-api

