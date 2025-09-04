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
import json

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

# Alternative models (try different ones if one fails)
MODEL_MAP = {
    "realistic": [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4"
    ],
    "anime": [
        "runwayml/stable-diffusion-v1-5",
        "hakurei/waifu-diffusion"
    ],
    "artistic": [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1"
    ],
    "digital_art": [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1"
    ],
}

# Quality presets
QUALITY_PRESETS = {
    "draft": {"width": 512, "height": 512, "num_inference_steps": 20},
    "standard": {"width": 512, "height": 512, "num_inference_steps": 30},
    "high": {"width": 512, "height": 512, "num_inference_steps": 50},
}


def run_generation(job_id, models_to_try, prompt, quality_settings, negative_prompt=None):
    """Background worker that calls Hugging Face and saves the result."""
    
    for model_name in models_to_try:
        api_url = f"{BASE_URL}{model_name}"
        payload = {
            "inputs": prompt,
            "parameters": quality_settings
        }
        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        try:
            logger.info(f"[{job_id}] Trying model: {model_name}")
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                # Check if response is actually an image
                if response.headers.get('content-type', '').startswith('image/'):
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["image"] = response.content
                    jobs[job_id]["model_used"] = model_name
                    logger.info(f"[{job_id}] Finished successfully with {model_name}")
                    return
                else:
                    logger.warning(f"[{job_id}] {model_name} returned non-image content")
                    continue
            
            elif response.status_code == 503:
                # Model is loading, wait and try next
                logger.warning(f"[{job_id}] {model_name} is loading, trying next...")
                continue
                
            elif response.status_code == 504:
                # Gateway timeout, try next model
                logger.warning(f"[{job_id}] {model_name} timeout, trying next...")
                continue
                
            else:
                logger.error(f"[{job_id}] {model_name} failed: {response.status_code} - {response.text[:200]}")
                continue

        except requests.exceptions.Timeout:
            logger.error(f"[{job_id}] {model_name} request timeout")
            continue
            
        except Exception as e:
            logger.error(f"[{job_id}] {model_name} exception: {e}")
            continue

    # If we get here, all models failed
    jobs[job_id]["status"] = "error"
    jobs[job_id]["error"] = "All models failed or are currently unavailable. Please try again later."
    logger.error(f"[{job_id}] All models failed")


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
        "themes": {k: v[0] for k, v in MODEL_MAP.items()},  # Show primary model
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

    models_to_try = MODEL_MAP[theme]
    quality_settings = QUALITY_PRESETS[quality].copy()

    # Enhance prompt automatically
    enhancers = {
        "realistic": "highly detailed, professional photography, sharp focus, realistic",
        "anime": "anime style, vibrant colors, manga illustration",
        "artistic": "artistic, masterpiece, oil painting style",
        "digital_art": "digital art, concept art, trending on artstation"
    }
    enhancement = enhancers.get(theme, "high quality, detailed")
    full_prompt = f"{prompt}, {enhancement}"

    # Create a job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending", 
        "created": time.time(),
        "prompt": full_prompt,
        "theme": theme,
        "quality": quality
    }

    # Run in background thread
    thread = threading.Thread(
        target=run_generation,
        args=(job_id, models_to_try, full_prompt, quality_settings, negative_prompt)
    )
    thread.daemon = True  # Dies when main thread dies
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
        return {
            "status": "error", 
            "error": job.get("error", "Unknown error occurred"),
            "job_id": job_id
        }

    elif job["status"] == "pending":
        # Check if job is too old (over 5 minutes)
        if time.time() - job["created"] > 300:
            job["status"] = "error"
            job["error"] = "Job timed out"
            return {"status": "error", "error": "Job timed out"}

    return {
        "status": job["status"],
        "job_id": job_id,
        "created": job["created"]
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Get just the status without downloading the image"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "created": job["created"],
        "model_used": job.get("model_used"),
        "error": job.get("error") if job["status"] == "error" else None
    }


# Clean up old jobs periodically
@app.on_event("startup")
async def startup_event():
    import asyncio
    
    async def cleanup_old_jobs():
        while True:
            current_time = time.time()
            old_jobs = [
                job_id for job_id, job in jobs.items() 
                if current_time - job["created"] > 3600  # 1 hour
            ]
            for job_id in old_jobs:
                del jobs[job_id]
            
            await asyncio.sleep(300)  # Clean every 5 minutes
    
    asyncio.create_task(cleanup_old_jobs())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)