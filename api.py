from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
from inference import load_model, VAL_TF, device

app = FastAPI()
model, idx2disease = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = VAL_TF(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(image, lambda_=0.0)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = probs.topk(5)

    results = [
        {"class": idx2disease[i.item()], "prob": float(p.item())}
        for p, i in zip(top5.values, top5.indices)
    ]

    return {"predictions": results}