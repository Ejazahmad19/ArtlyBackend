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
        # Premium models (try first)
        "black-forest-labs/FLUX.1-schnell",  # Fast, high quality
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
        "Lykon/DreamShaper",
        # Free fallback models
        "dreamlike-art/dreamlike-diffusion-1.0",
        "prompthero/openjourney",
        "wavymulder/Analog-Diffusion",
        "nitrosocke/Ghibli-Diffusion",
    ],
    "anime": [
        # Premium models (try first)
        "cagliostrolab/animagine-xl-3.1",
        "Linaqruf/anything-v3.0",
        "black-forest-labs/FLUX.1-schnell",
        # Free fallback models
        "hakurei/waifu-diffusion",
        "nitrosocke/Ghibli-Diffusion",
        "dreamlike-art/dreamlike-anime-1.0",
        "gsdf/Counterfeit-V2.5",
    ],
    "artistic": [
        # Premium models (try first)
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "Lykon/DreamShaper",
        # Free fallback models
        "dreamlike-art/dreamlike-diffusion-1.0",
        "nitrosocke/Arcane-Diffusion",
        "prompthero/openjourney",
        "wavymulder/Analog-Diffusion",
        "22h/vintedois-diffusion-v0-1",
    ],
    "digital_art": [
        # Premium models (try first)
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "Lykon/DreamShaper",
        # Free fallback models
        "prompthero/openjourney",
        "dreamlike-art/dreamlike-diffusion-1.0",
        "nitrosocke/Arcane-Diffusion",
        "wavymulder/Analog-Diffusion",
        "22h/vintedois-diffusion-v0-1",
    ],
}

# Quality presets
QUALITY_PRESETS = {
    "draft": {"width": 512, "height": 512, "num_inference_steps": 4},
    "standard": {"width": 512, "height": 512, "num_inference_steps": 8},
    "high": {"width": 768, "height": 768, "num_inference_steps": 12},
}

# Model-specific adjustments
def adjust_quality_for_model(model_name, quality_settings):
    """Adjust quality settings based on model capabilities"""
    free_models = [
        "dreamlike-art", "prompthero", "wavymulder", "nitrosocke", 
        "hakurei", "gsdf", "22h", "Linaqruf/anything-v3.0"
    ]
    
    # Check if this is a free model (lower capability)
    is_free_model = any(free_model in model_name for free_model in free_models)
    
    if is_free_model:
        # Reduce resolution for free models to ensure they work
        if quality_settings.get("width", 512) > 512:
            quality_settings = quality_settings.copy()
            quality_settings["width"] = 512
            quality_settings["height"] = 512
            # Increase steps slightly to compensate for lower resolution
            quality_settings["num_inference_steps"] = min(20, quality_settings.get("num_inference_steps", 10) + 5)
    
    return quality_settings


def run_generation(job_id, models_to_try, prompt, quality_settings, negative_prompt=None):
    """Background worker that calls Hugging Face and saves the result."""
    jobs[job_id]["status"] = "processing"

    
    for model_name in models_to_try:
        api_url = f"{BASE_URL}{model_name}"
        
        # Adjust quality settings based on model capabilities
        adjusted_settings = adjust_quality_for_model(model_name, quality_settings)
        
        payload = {
            "inputs": prompt,
            "parameters": adjusted_settings
        }
        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        try:
            logger.info(f"[{job_id}] Trying model: {model_name} (settings: {adjusted_settings})")
            jobs[job_id]["current_model"] = model_name 
            response = requests.post(api_url, headers=headers, json=payload, timeout=180)
            logger.info(f"[{job_id}] Response status: {response.status_code}")
            logger.info(f"[{job_id}] Content-Type: {response.headers.get('content-type')}")

            if response.status_code == 200:
                # Check if response is actually an image
                if response.headers.get('content-type', '').startswith('image/'):
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["image"] = response.content
                    jobs[job_id]["model_used"] = model_name
                    logger.info(f"[{job_id}] ✅ SUCCESS with {model_name}")
                    return
                else:
                    logger.warning(f"[{job_id}] {model_name} returned non-image content")
                    continue
            
            elif response.status_code == 402:
                # Payment required - this is a premium model, try next (could be free)
                logger.warning(f"[{job_id}] 💳 {model_name} requires payment (premium model), trying next...")
                continue
            
            elif response.status_code == 503:
                # Model is loading, wait a bit and try next
                logger.warning(f"[{job_id}] ⏳ {model_name} is loading, waiting 10s then trying next...")
                time.sleep(10)
                continue
                
            elif response.status_code == 504:
                # Gateway timeout, try next model
                logger.warning(f"[{job_id}] ⏰ {model_name} timeout, trying next...")
                continue
                
            else:
                logger.error(f"[{job_id}] ❌ {model_name} failed: {response.status_code} - {response.text[:200]}")
                continue

        except requests.exceptions.Timeout:
            logger.error(f"[{job_id}] {model_name} request timeout")
            continue
            
        except Exception as e:
            logger.error(f"[{job_id}] {model_name} exception: {e}")
            continue

    # If we get here, all models failed
    jobs[job_id]["status"] = "error"
    jobs[job_id]["error"] = "All models failed or are currently unavailable. Tried both premium and free models."
    logger.error(f"[{job_id}] ❌ All models failed (both premium and free)")


@app.get("/")
def root():
    return {
        "message": "AI Image Generator Backend is running 🚀",
        "available_themes": list(MODEL_MAP.keys()),
        "quality_presets": list(QUALITY_PRESETS.keys()),
        "strategy": "Premium models first, fallback to free community models"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/models")
def get_available_models():
    return {
        "themes": {k: v[0] for k, v in MODEL_MAP.items()},  # Show primary model
        "quality_presets": QUALITY_PRESETS,
        "note": "Premium models tried first, with free fallbacks"
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

    # Enhanced prompts for better results
    enhancers = {
        "realistic": "highly detailed, professional photography, sharp focus, realistic, best quality",
        "anime": "anime style, vibrant colors, manga illustration, high quality, detailed",
        "artistic": "artistic, masterpiece, beautiful art, high quality, detailed",
        "digital_art": "digital art, concept art, high quality, detailed, beautiful"
    }
    enhancement = enhancers.get(theme, "high quality, detailed, beautiful")
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
        # Check if job is too old (over 10 minutes)
        if time.time() - job["created"] > 600:
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
        "current_model": job.get("current_model"),
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


@app.get("/debug/{job_id}")
def debug_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "current_model": job.get("current_model"),
        "created": job.get("created"),
        "error": job.get("error")
    }

@app.get("/debug-token")
def debug_token():
    """Test if HF token is working with both premium and free models"""
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set"}
    
    results = []
    
    # Test premium model first
    premium_url = f"{BASE_URL}black-forest-labs/FLUX.1-schnell"
    premium_payload = {
        "inputs": "a simple test image",
        "parameters": {"width": 512, "height": 512, "num_inference_steps": 4}
    }
    
    try:
        response = requests.post(premium_url, headers=headers, json=premium_payload, timeout=30)
        results.append({
            "model": "black-forest-labs/FLUX.1-schnell",
            "type": "premium",
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type'),
            "response_text": response.text[:100] if response.status_code != 200 else "OK"
        })
    except Exception as e:
        results.append({
            "model": "black-forest-labs/FLUX.1-schnell",
            "type": "premium",
            "error": str(e)
        })
    
    # Test free model
    free_url = f"{BASE_URL}dreamlike-art/dreamlike-diffusion-1.0"
    free_payload = {
        "inputs": "a simple test image",
        "parameters": {"width": 512, "height": 512, "num_inference_steps": 10}
    }
    
    try:
        response = requests.post(free_url, headers=headers, json=free_payload, timeout=30)
        results.append({
            "model": "dreamlike-art/dreamlike-diffusion-1.0",
            "type": "free",
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type'),
            "response_text": response.text[:100] if response.status_code != 200 else "OK"
        })
    except Exception as e:
        results.append({
            "model": "dreamlike-art/dreamlike-diffusion-1.0",
            "type": "free",
            "error": str(e)
        })
    
    return {
        "token_set": True,
        "test_results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)