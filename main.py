import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import logging
import uuid
import threading
from typing import Optional

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Generator", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN not set!")

BASE_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Simple in-memory job store
jobs = {}

# Model map (you can expand later)
MODEL_MAP = {
    "realistic": "stabilityai/stable-diffusion-xl-base-1.0",
    "anime": "stabilityai/stable-diffusion-xl-base-1.0",
    "artistic": "stabilityai/stable-diffusion-xl-base-1.0",
    "digital_art": "stabilityai/stable-diffusion-xl-base-1.0",
}

# Quality presets
QUALITY_PRESETS = {
    "draft": {"width": 512, "height": 512},
    "standard": {"width": 768, "height": 768},
    "high": {"width": 1024, "height": 1024},
}


def run_generation(job_id, model_name, prompt, quality_settings, negative_prompt=None):
    """Background worker that calls Hugging Face and saves the result."""
    api_url = f"{BASE_URL}{model_name}"
    payload = {
        "inputs": prompt,
        "parameters": quality_settings
    }
    if negative_prompt:
        payload["parameters"]["negative_prompt"] = negative_prompt

    try:
        logger.info(f"[{job_id}] Starting generation…")
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["image"] = response.content
            logger.info(f"[{job_id}] Finished successfully")
        else:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = response.text
            logger.error(f"[{job_id}] Failed: {response.text}")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        logger.error(f"[{job_id}] Exception: {e}")


@app.get("/")
def root():
    return {
        "message": "AI Image Generator Backend is running 🚀",
        "available_themes": list(MODEL_MAP.keys()),
        "quality_presets": list(QUALITY_PRESETS.keys())
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/models")
def get_available_models():
    return {
        "themes": MODEL_MAP,
        "quality_presets": QUALITY_PRESETS
    }


@app.get("/generate")
def generate(
    prompt: str = Query(..., description="Your text prompt for image generation"),
    theme: str = Query("realistic", description="Theme/style of the image"),
    quality: str = Query("draft", description="Quality preset (draft, standard, high)"),
    negative_prompt: Optional[str] = Query(None, description="Things to avoid")
):
    if theme not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid theme. Available: {list(MODEL_MAP.keys())}")

    if quality not in QUALITY_PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid quality. Available: {list(QUALITY_PRESETS.keys())}")

    model_name = MODEL_MAP[theme]
    quality_settings = QUALITY_PRESETS[quality].copy()

    # Enhance prompt automatically
    enhancers = {
        "realistic": "highly detailed, professional photography, sharp focus, realistic, 8k uhd",
        "anime": "anime style, vibrant colors, manga illustration",
        "artistic": "artistic, masterpiece, oil painting style",
        "digital_art": "digital art, concept art, trending on artstation"
    }
    enhancement = enhancers.get(theme, "high quality, detailed")
    full_prompt = f"{prompt}, {enhancement}"

    # Create a job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "created": time.time()}

    # Run in background thread
    thread = threading.Thread(
        target=run_generation,
        args=(job_id, model_name, full_prompt, quality_settings, negative_prompt)
    )
    thread.start()

    return {"job_id": job_id, "status": "pending"}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "done":
        return Response(content=job["image"], media_type="image/png")

    elif job["status"] == "error":
        return {"status": "error", "error": job.get("error")}

    return {"status": job["status"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
