# model.py - ML teammate should implement real logic here

import time

# Example: load model once (replace with actual model code)
def load_model():
    # TODO: Load your real ML/DL model here
    print("Dummy model loaded")
    return "dummy-model"

# Example: predict function
def predict(file_bytes: bytes):
    # TODO: preprocess input, run inference, postprocess
    time.sleep(0.1)  # simulate work
    return {
        "label": "deepfake",   # or "real"
        "confidence": 0.83,
        "model_version": "v1.0.0"
    }

# Global model (loaded at import time)
model = load_model()
