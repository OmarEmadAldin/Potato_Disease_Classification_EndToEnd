# Potato Disease Classification — End to End

A deep learning project that classifies potato leaf diseases from images using a custom CNN built with TensorFlow/Keras. Includes a FastAPI backend and a custom HTML frontend, packaged in Docker for deployment anywhere.

---

## Demo

Upload a potato leaf photo → get instant diagnosis with confidence scores.

**Supported classes:**

| Class | Cause | Severity |
|---|---|---|
| Early Blight | *Alternaria solani* (fungus) | Moderate |
| Late Blight | *Phytophthora infestans* | Severe |
| Healthy | — | — |

---

## Project Structure

```
├── api.py                       # FastAPI backend — serves the model and the UI
├── ui.html                      # Frontend — drag & drop image classifier
├── potato_disease_model.keras   # Trained CNN model
├── model.tflite                 # TFLite version (for mobile/edge)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container config — run anywhere
└── train.ipynb                  # Model training notebook
```

---

## How It Works

1. User uploads a potato leaf image via the HTML UI
2. The image is sent to the FastAPI `/predict` endpoint
3. The model preprocesses the image (resize to 256×256, normalize) and runs inference
4. The API returns the predicted class, confidence score, and all class probabilities
5. The UI displays the result with color-coded diagnosis and advice

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the HTML frontend |
| `GET` | `/ping` | Health check |
| `POST` | `/predict` | Accepts an image file, returns prediction |

### Example `/predict` response

```json
{
  "class": "Early_blight",
  "confidence": 0.97,
  "all_scores": {
    "Early_blight": 0.97,
    "Late_blight": 0.02,
    "Healthy": 0.01
  },
  "model": "potato_disease_model",
  "image_size": 256
}
```

---

## Run with Docker

```bash
# Build the image
docker build -t potato-disease-app .

# Run the container
docker run -p 8000:8000 potato-disease-app
```

Then open [http://localhost:8000](http://localhost:8000) — the UI loads directly.

---

## Run Locally (without Docker)

```bash
pip install -r requirements.txt
python api.py
```

---

## Model

- **Architecture:** Custom CNN (Convolutional Neural Network)
- **Framework:** TensorFlow / Keras
- **Input size:** 256 × 256 RGB
- **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) — potato subset (2,152 images)
- **Output:** Softmax over 3 classes

The training notebook (`train.ipynb`) covers data loading, augmentation, model definition, training, and evaluation.

A TFLite version (`model.tflite`) is also included for lightweight/mobile inference.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Image processing | Pillow |
| Deployment | Docker |

---

## Author

**Omar Emad Aldin** — [GitHub](https://github.com/OmarEmadAldin)
