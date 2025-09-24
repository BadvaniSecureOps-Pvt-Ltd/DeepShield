DeepShield

Mobile-First Deepfake Detection** | SIH 2025 Project

About the Project

DeepShield helps users **detect deepfakes and fake content** by analyzing uploaded **images/videos** using AI/ML models.
The solution is:

Mobile-first → Works on Android/iOS using Flutter.
Lightweight → FastAPI backend optimized for hackathon deployment.
Reliable → Provides detection + explainability with confidence score.
Secure → Supports API key authentication and minimal data collection.

*********************************

Problem Statement

> With the rise of manipulated media, misinformation spreads quickly. Detecting deepfakes in a user-friendly way is critical for digital trust.

DeepShield addresses this by offering a **simple mobile app** + **scalable backend** that can be used by individuals, journalists, or institutions.

*********************************

Architecture

```
[ Mobile App (Flutter) ] ⇆ [ Backend API (FastAPI) ] ⇆ [ ML Model (PyTorch/TensorFlow) ]
```

App (Flutter):Capture/Upload → Send to backend → Show result.
Backend (FastAPI): Handle requests → Run inference → Return JSON.
Model: Detect deepfakes → Return label + confidence + explanation.

---

## Repository Structure

```
DeepShield/
│── app/                 # Flutter mobile app
│   └── deepfake_app/    # App source
│── backend/             # FastAPI backend
│── README.md            # Main project documentation
```

---

Getting Started

 1. Clone Repository

```bash
git clone https://github.com/<org_or_user>/DeepShield.git
cd DeepShield
```

2. Setup Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at:
`http://127.0.0.1:8000`

 3. Setup Mobile App

```bash
cd app/deepfake_app
flutter pub get
flutter run
```

---

Git Workflow

We follow a **branch-based workflow**:

* `main` → Stable, production-ready code.
* `backend-api` → Backend development branch.
* `flutter-app` → Flutter app development branch.

### Common Git Commands

```bash
# Clone repo
git clone https://github.com/BadvaniSecureOps-Pvt-Ltd/DeepShield

# Switch branch
git checkout flutter-app  # or backend-api

# Pull latest changes
git pull origin flutter-app

# Add and commit changes
git add .
git commit -m "feat: added upload API integration"

# Push changes
git push origin flutter-app
```

---

##  Demo Flow (Hackathon Thin-Slice)

1. User uploads **image/video**.
2. App compresses & sends file → backend.
3. Backend runs model → returns JSON.
4. App shows result: ✅ *Real* / ❌ *Deepfake* with confidence %.

---

##  Impact & Benefits

* **Misinformation Control** → Stops fake media early.
* **Accessibility** → Works on mobile, no heavy infra needed.
* **Scalable** → Extendable for government/media use.

---

##  Team DeepShield (IIIT Kottayam)

* **Shyam (Leader)** → Coordination, integration, presentation.
* **ML Engineer** → Model training & inference.
* **Backend Developer** → API design & deployment.
* **App Developer (Flutter)** → UI/UX + mobile integration.
* **Research & Docs** → Market study, PPT, Q\&A prep.
* **QA & Support** → Testing, validation, feedback.

---

## References & Research

* SIH Problem Statement 2025
* Papers on deepfake detection (XceptionNet, MesoNet)
* FastAPI docs, Flutter docs

---

## 📌 Roadmap

* Hackathon prototype (thin slice working demo)
* Model fine-tuning & dataset expansion
* Cloud deployment (AWS/GCP)
* Security & privacy compliance

---

##  License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

 *DeepShield — Building trust in the digital age.*
