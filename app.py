import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image

app = FastAPI(title="Rice Leaf Disease Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

MODEL_PATH = "mobilenetv2_frozen.keras"

model = None

def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Match exactly the training preprocessing pipeline."""
    # Decode bytes -> numpy array (same as cv2.imread)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # reads as BGR

    if img is None:
        raise ValueError("Could not decode the uploaded image.")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # BGR → RGB
    img = img.astype("float32") / 255.0              # normalize [0, 1]
    img = np.expand_dims(img, axis=0)                # (1, 224, 224, 3)
    return img


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/")
def root():
    return {"status": "Rice Leaf Disease Classifier is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        img = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    m = get_model()
    preds = m.predict(img, verbose=0)[0]           # shape: (3,)

    pred_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(preds[pred_index])

    probabilities = {
        CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=round(confidence * 100, 2),
        probabilities=probabilities,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
