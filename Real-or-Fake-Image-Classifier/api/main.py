from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from FaceNest import load_model, process_image
import torch

app = FastAPI()

# Load the model
model = load_model("../model/best_model.pth")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process the image and make predictions
        processed_image = process_image(image)
        with torch.no_grad():
            outputs = model(processed_image)
            _, predicted = torch.max(outputs, 1)
            label = "Real" if predicted.item() == 0 else "AI-Generated"
        return {"label": label}
    except Exception as e:
        return {"error": str(e)}
