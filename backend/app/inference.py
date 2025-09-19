import random

def run_inference(file_path: str):
    """
    Dummy inference function – replace with real ML model later.
    """
    label = random.choice(["real", "deepfake"])
    confidence = round(random.uniform(0.7, 0.99), 2)
    model_version = "v1.0.0"
    return label, confidence, model_version
