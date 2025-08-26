import requests
from fastapi import FastAPI, Query
from fastapi.responses import Response
import os

app = FastAPI()

# Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# Default model
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Optional: Map themes to specific models (uncomment if you want multiple models)
# model_map = {
#     "anime": "hakurei/waifu-diffusion",
#     "realistic": "stabilityai/stable-diffusion-xl-base-1.0",
#     "futuristic": "stabilityai/stable-diffusion-xl-refiner-1.0"
# }

@app.get("/")
def root():
    return {"message": "AI Image Generator Backend is running 🚀"}

@app.get("/generate")
def generate(
    prompt: str = Query(..., description="Your text prompt for image generation"),
    theme: str = Query("realistic", description="Theme/style of the image (anime, realistic, futuristic)"),
    width: int = Query(512, description="Image width in pixels"),
    height: int = Query(512, description="Image height in pixels")
):
    # If using model_map for themes, you can switch models here:
    # api_url = model_map.get(theme, API_URL)
    api_url = API_URL

    # Append theme to prompt
    full_prompt = f"{prompt}, {theme} style"

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "width": width,
            "height": height
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    return Response(content=response.content, media_type="image/png")