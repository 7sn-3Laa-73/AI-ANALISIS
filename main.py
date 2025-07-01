from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# استخدم الموديل بشكل مباشر بدون الحاجة لـ torch
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"  # أو جرب تحذفها تماماً
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    result = classifier(data.text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4)
    }
