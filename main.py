import requests
from fastapi import FastAPI, Query
from fastapi.responses import Response
import os

app = FastAPI()

# Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.get("/")
def root():
    return {"message": "AI Image Generator Backend is running 🚀"}

@app.get("/generate")
def generate(prompt: str = Query(..., description="Your text prompt for image generation")):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        return {"error": response.text}
    
    return Response(content=response.content, media_type="image/png")
