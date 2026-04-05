from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import cv2
import tensorflow as tf

MODEL_PATH = "./model/potato_disease_model.keras"
CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']
IMAGE_SIZE = 256   # change if your model uses different size
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(
    title="Potato Disease Classification API",
    version="1.0"
)

@app.get("/")
def health():
    return {"status": "API running"}

@app.get("/ping")
def ping():
    return {"message": "API alive"}


def read_file_as_image(data: bytes) -> np.ndarray:

    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        image = read_file_as_image(await file.read())

        image_batch = np.expand_dims(image, axis=0)

        predictions = model.predict(image_batch)

        predicted_index = int(np.argmax(predictions[0]))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=1234)