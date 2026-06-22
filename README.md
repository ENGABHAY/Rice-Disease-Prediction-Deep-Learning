# 🌾 Rice Leaf Disease Classifier

**Deep learning system that detects three major rice leaf diseases from a single photo — built with a custom CNN, benchmarked against four transfer-learning backbones, and shipped as a FastAPI + Streamlit web app.**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Best%20Model-MobileNetV2-blue" />
  <img src="https://img.shields.io/badge/Test%20Accuracy-95.83%25-success" />
</p>

---

## 📌 Overview

Rice is one of the world's most important staple crops, and leaf disease is one of the biggest threats to its yield. Catching an infection early — and identifying *which* disease it is — makes the difference between a quick, targeted treatment and a damaged harvest.

This project builds an end-to-end image classification pipeline that takes a photo of a rice leaf and predicts which of **three common diseases** it shows:

| Disease | Cause | Typical Symptom |
|---|---|---|
| 🦠 **Bacterial Leaf Blight** | *Xanthomonas oryzae* | Water-soaked streaks that turn yellow-white along leaf margins |
| 🟤 **Brown Spot** | *Bipolaris oryzae* | Circular brown/tan lesions scattered across the blade |
| ⚫ **Leaf Smut** | *Entyloma oryzae* | Small black raised spots (sori) on both leaf surfaces |

Thirteen different model configurations — spanning a CNN built from scratch and four ImageNet-pretrained backbones — were trained and benchmarked on the same dataset and split. The best-performing model (**MobileNetV2, frozen base**) is served through a **FastAPI** inference backend and a polished **Streamlit** front end that returns a diagnosis with confidence scores and a practical management tip.

---

## ✨ Key Highlights

- 🧠 **13 model configurations evaluated** — a from-scratch CNN (baseline, tuned, and augmented variants) benchmarked against MobileNetV2, MobileNet, VGG19, and EfficientNetB0
- 🎯 **95.83% test accuracy** and **0.96 weighted F1-score** with the best model (MobileNetV2, frozen base)
- 🔁 **5-fold stratified cross-validation** used to sanity-check results given the small dataset (119 images)
- 🛠️ **Keras Tuner hyperparameter search** to systematically improve the custom CNN
- 🌱 **Data augmentation pipeline** (flip, rotation, zoom, contrast) to combat overfitting on a small dataset
- 🌐 **Production-style deployment** — FastAPI REST API + Streamlit UI, fully decoupled
- 📊 **Full data-analysis & model-comparison report** documented inside the Jupyter notebook (EDA → preprocessing → 13 trained models → comparison → final recommendation)

---

## 🖥️ Application Preview

### Landing Page
A clean drag-and-drop interface — upload a leaf photo and get a diagnosis in seconds.

![Landing Page](Assets/Landing%20Page.png)

### Upload & Live Inference
The uploaded leaf is previewed instantly while the model runs inference in the background.

![Uploading Overview](Assets/Uploading%20Overview.png)

### Prediction Result
The result card shows the predicted disease, a confidence score, a probability breakdown across all three classes, and a short agronomic management tip — plus a raw JSON view of the API response for developers.

![Prediction Page](Assets/Prediction%20Page.png)

> 🎥 A full screen-recorded walkthrough of the app (`Live Demo.mp4`) is included in the repository.

---

## 📊 Model Performance

All 13 trained configurations were evaluated on an identical held-out test split for a fair, apples-to-apples comparison.

### Test Accuracy — All Models

![Model Accuracy Comparison](Assets/Models%20Accuracy%20Comparision.png)

### Test Loss — All Models (lower is better)

![Model Loss Comparison](Assets/Models%20Loss%20Comparision.png)

### 🏆 Final Model Leaderboard

| Rank | Model | Test Accuracy | Test Loss |
|---|---|---|---|
| 🥇 | **MobileNetV2 (Frozen)** | **95.83%** | **0.1263** |
| 🥇 | MobileNetV2 + Augmentation | 95.83% | 0.2003 |
| 🥈 | MobileNet (Frozen) + Augmentation | 91.67% | 0.2298 |
| 🥈 | Custom CNN + Augmentation + Tuned HP | 91.67% | 0.4627 |
| 🥉 | Custom CNN + Data Augmentation | 87.50% | 0.3934 |
| 🥉 | VGG19 (Frozen) + Augmentation | 87.50% | 0.5640 |
| — | MobileNetV2 + Augmentation + 5-Fold CV | 86.96% (avg) | 0.3062 |
| — | Custom CNN + Keras Tuner (best trial) | 83.33% | 0.6947 |
| — | Custom CNN (Augmented, final run) | 83.33% | 0.4578 |
| — | Baseline Custom CNN | 79.17% | 0.5868 |
| — | MobileNetV2 (fine-tuned, last 30 layers) | 54.17% | 1.2585 |
| — | EfficientNetB0 (Frozen) | 33.33% | 1.1016 |
| — | EfficientNetB0 + Augmentation | 33.33% | 1.0974 |

### ✅ Why MobileNetV2 (Frozen) Was Selected

- Highest test accuracy (**95.83%**) and lowest test loss (**0.1263**) of all 13 models
- Correctly classified **23 of 24** held-out test images
- Weighted **F1-score of 0.96** with strong precision/recall across all three classes
- Reuses rich, general-purpose visual features pretrained on ImageNet, which compensates for the very limited training data (only 95 training images after the split)
- Lightweight enough (frozen backbone, small trainable head) to run fast inference in a web app without a GPU

> ⚠️ **Interesting failure case:** EfficientNetB0 failed to learn at all (33% accuracy ≈ random guessing for 3 classes) regardless of augmentation. This is a known quirk of EfficientNet's expected `[0, 255]`-range preprocessing — the notebook documents this as a key debugging/learning takeaway rather than hiding it.

---

## 🧪 Methodology

The full experimental process is documented step-by-step inside [`Rice Leaf Disease Prediction.ipynb`](./Rice%20Leaf%20Disease%20Prediction.ipynb):

1. **Dataset Overview** — 119 images across 3 near-perfectly balanced classes (40 / 40 / 39), loaded and resized to 224×224×3.
2. **Exploratory Data Analysis** — class balance checks, sample image grids per class, pixel-intensity histograms, and image-dimension audits.
3. **Preprocessing** — label encoding + one-hot encoding, pixel normalization to `[0, 1]`, and a consistent train/test split reused across every model for fair comparison.
4. **Model Building**
   - **Custom CNN (from scratch)** — baseline → Keras Tuner hyperparameter search → data augmentation → combined (tuned + augmented)
   - **Transfer Learning** — EfficientNetB0, MobileNetV2 (frozen + fine-tuned), MobileNet, VGG19, each with a frozen ImageNet backbone and a custom classification head
   - **Robustness check** — 5-fold stratified cross-validation on the strongest transfer-learning configuration to confirm results weren't a lucky split
5. **Model Comparison** — consolidated accuracy/loss bar charts, precision/recall/F1 comparison, and per-class confusion-matrix error analysis between the best CNN and best transfer-learning model.
6. **Final Recommendation** — MobileNetV2 (frozen) selected and justified with full metrics.

### Key Techniques Explored
`Data Augmentation` · `Keras Tuner (RandomSearch)` · `Transfer Learning` · `Layer Fine-Tuning` · `K-Fold Cross-Validation` · `EarlyStopping`

### Challenges Addressed in the Notebook
- **Very small dataset (119 images total)** — the dominant constraint throughout the project, addressed via augmentation, transfer learning, and cross-validation to obtain more reliable estimates.
- **High metric sensitivity** — with only 24 test images, a single misclassification swings test accuracy by ~4.17%, so results are interpreted with appropriate caution rather than over-claiming.
- **Backbone-specific preprocessing quirks** — EfficientNetB0's failure to converge under `[0, 1]` normalization versus MobileNetV2's success under the same pipeline is documented as a concrete lesson on architecture-specific input requirements.

---

## 🏗️ Tech Stack & Architecture

| Layer | Technology |
|---|---|
| Modeling | TensorFlow / Keras, Keras Tuner |
| Computer Vision | OpenCV, Pillow |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Data / Plotting | NumPy, Pandas, Matplotlib, Scikit-learn |

The app follows a simple **client–server architecture**: the Streamlit UI never touches the model directly — it sends the uploaded image to a FastAPI inference endpoint and renders whatever JSON comes back. This keeps the model swappable and makes the API reusable by any other client (mobile app, curl, Postman, etc.).

```
┌─────────────────────┐        multipart/form-data        ┌──────────────────────────┐
│   Streamlit Frontend │ ────────────────────────────────► │      FastAPI Backend     │
│  (streamlit_app.py)  │                                   │        (app.py)         │
│                      │ ◄──────────────────────────────── │                          │
│  • Upload widget     │        JSON prediction             │  • Loads .keras model    │
│  • Result UI / bars  │                                   │  • Preprocess (cv2)      │
└─────────────────────┘                                   │  • Predict + softmax     │
                                                            └──────────────────────────┘
                                                                       │
                                                                       ▼
                                                          mobilenetv2_frozen.keras
```

**Inference pipeline** (mirrors training preprocessing exactly):
1. Decode uploaded bytes → image array (OpenCV)
2. Resize to `224 × 224`
3. Convert `BGR → RGB`
4. Normalize pixel values to `[0, 1]`
5. Run through MobileNetV2 → softmax over 3 classes
6. Return predicted class, confidence %, and full per-class probability breakdown

---

## 📁 Repository Structure

```
Rice-Leaf-Disease-Prediction/
│
├── Assets/                                  # Screenshots used in this README
│   ├── Landing Page.png
│   ├── Uploading Overview.png
│   ├── Prediction Page.png
│   ├── Models Accuracy Comparision.png
│   └── Models Loss Comparision.png
│
├── Rice Leaf Disease Datasets/              # 119 labeled leaf images (3 classes)
│   ├── Bacterial leaf blight/
│   ├── Brown spot/
│   └── Leaf smut/
│
├── Rice Leaf Disease Prediction.ipynb       # Full EDA → modeling → comparison notebook
├── PRCP-1001-RiceLeaf.docx                  # Original problem statement
├── app.py                                   # FastAPI inference backend
├── streamlit_app.py                         # Streamlit frontend
├── mobilenetv2_frozen.keras                 # Final recommended model (95.83% accuracy)
├── MobileNetV2_aug_model.keras              # Augmented MobileNetV2 variant
├── Live Demo.mp4                            # Screen-recorded app walkthrough
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Rice-Leaf-Disease-Prediction.git
cd Rice-Leaf-Disease-Prediction
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the FastAPI backend
```bash
uvicorn app:app --reload --port 8000
```
The API will be live at `http://localhost:8000`. Interactive docs are auto-generated at `http://localhost:8000/docs`.

### 4. Start the Streamlit frontend
In a **second terminal**:
```bash
streamlit run streamlit_app.py
```
The app will open at `http://localhost:8501`.

### 5. Try it out
Upload any rice leaf image (JPG, JPEG, PNG, or WEBP) and get an instant prediction with confidence scores.

---

## 🔌 API Reference

### `POST /predict`
Classifies an uploaded rice leaf image.

**Request:** `multipart/form-data` with a single `file` field (image).

**Response:**
```json
{
  "predicted_class": "Brown spot",
  "confidence": 99.05,
  "probabilities": {
    "Bacterial leaf blight": 0.0,
    "Brown spot": 99.05,
    "Leaf smut": 0.95
  }
}
```

### `GET /health`
Returns service and model-load status — useful for uptime checks.

### `GET /`
Basic root health message confirming the API is running.


## 📚 Dataset & Acknowledgements

- **Dataset:** 119 rice leaf images across 3 disease classes (Bacterial Leaf Blight, Brown Spot, Leaf Smut), sourced from the original capstone project brief (`PRCP-1001-RiceLeaf`).
- **Pretrained backbones:** MobileNetV2, MobileNet, VGG19, and EfficientNetB0 — all ImageNet-pretrained weights via `tensorflow.keras.applications`.

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE). Feel free to use, modify, and build on it.

---

<p align="center">Built with 🌱 for smarter, faster rice crop diagnosis.</p>
