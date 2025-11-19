DeepShield Mobile App
DeepShield is a mobile-first deepfake detection app built using Flutter.
Users can upload images or videos, and the app communicates with the backend API to return predictions with confidence scores.
________________________________________
Setup & Installation
1. Clone Repository
git clone https://github.com/BadvaniSecureOps-Pvt-Ltd/DeepShield.git
cd DeepShield/app/deepfake_app
2. Install Dependencies
flutter pub get
3. Configure API URL
Update the API endpoint inside lib/config.dart (create this file if not present):
const String API_URL = "http://192.168.xxx.xxx:8000/predict"; // Replace with backend IP
 For Android Emulator use:
const String API_URL = "http://10.0.2.2:8000/predict";
4. Run the App
flutter run
________________________________________
 Features
•	Upload image/video from gallery or camera.
•	Compresses media before upload.
•	 Sends request to backend (/predict).
•	Displays prediction with confidence score.
•	API key support for extra security.
________________________________________
Git Workflow (For App Developers)
1. Switch to App Branch
git checkout flutter-app
2. Pull Latest Changes
git pull origin flutter-app
3. Make Changes & Commit
git add .
git commit -m "feat: added upload button UI"
4. Push Changes
git push origin flutter-app
5. Create a Pull Request
•	Go to GitHub → Compare & Pull Request.
•	Request review from team lead.
•	Merge into main after approval.
