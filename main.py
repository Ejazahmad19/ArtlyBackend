from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import logging
import uuid
import asyncio
from typing import Optional
import json
import gc
from concurrent.futures import ThreadPoolExecutor
import weakref
from contextlib import asynccontextmanager

# Diffusers imports
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import numpy as np
from PIL import Image

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration - Optimized for Render
MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1"))
MAX_JOBS_IN_MEMORY = int(os.getenv("MAX_JOBS_IN_MEMORY", "10"))
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))  # 5 minutes timeout
IMAGE_CLEANUP_INTERVAL = int(os.getenv("IMAGE_CLEANUP_INTERVAL", "120"))  # 2 minutes
MAX_THREAD_WORKERS = int(os.getenv("MAX_THREAD_WORKERS", "1"))

# Get port from environment (Render requirement)
PORT = int(os.getenv("PORT", "8000"))

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS)

# Enhanced job storage with cleanup
class JobManager:
    def __init__(self):
        self.jobs = {}
        self.active_generations = 0
        self.last_cleanup = time.time()

    def add_job(self, job_id: str, job_data: dict):
        self._cleanup_if_needed()
        self.jobs[job_id] = job_data

    def get_job(self, job_id: str):
        self._cleanup_if_needed()
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, updates: dict):
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)

    def _cleanup_if_needed(self):
        now = time.time()
        if now - self.last_cleanup > IMAGE_CLEANUP_INTERVAL:
            self._cleanup_old_jobs()
            self.last_cleanup = now

    def _cleanup_old_jobs(self):
        now = time.time()
        expired_jobs = []

        for job_id, job in list(self.jobs.items()):
            age = now - job.get('created', now)
            if age > JOB_TIMEOUT or (len(self.jobs) > MAX_JOBS_IN_MEMORY and job.get('status') == 'done'):
                expired_jobs.append(job_id)

        for job_id in expired_jobs:
            if job_id in self.jobs:
                del self.jobs[job_id]

        if expired_jobs:
            logger.info(f"🧹 Cleaned up {len(expired_jobs)} old jobs")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

job_manager = JobManager()

# Model management with better memory handling for Render
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        # Optimized models for CPU/limited GPU environments
        self.model_map = {
            "realistic": "runwayml/stable-diffusion-v1-5",
            "anime": "hakurei/waifu-diffusion", 
            "artistic": "dreamlike-art/dreamlike-diffusion-1.0",
            "digital_art": "prompthero/openjourney-v4"
        }

    def get_pipeline(self, model_name: str):
        """Load model only when needed, unload others to save memory"""
        if self.current_model_name == model_name and self.current_model is not None:
            return self.current_model

        # Clear previous model from memory
        if self.current_model is not None:
            logger.info(f"🗑️ Unloading {self.current_model_name}")
            del self.current_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load new model
        logger.info(f"⬇️ Loading model: {model_name}")
        try:
            # Check if we should use CPU or GPU
            device = "cpu"  # Default to CPU for Render free tier
            dtype = torch.float32
            
            # Only use GPU if explicitly available and we have enough memory
            if torch.cuda.is_available():
                try:
                    # Check GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if gpu_memory > 6e9:  # Only use GPU if > 6GB VRAM
                        device = "cuda"
                        dtype = torch.float16
                        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        logger.info(f"GPU memory insufficient ({gpu_memory/1e9:.1f}GB), using CPU")
                except:
                    logger.info("GPU available but defaulting to CPU for stability")
            
            # Conservative loading parameters for Render
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=False,
                resume_download=True
            )

            # Use better scheduler
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

            # Move to device
            pipe.to(device)
            
            # Enable optimizations based on device
            if device == "cuda":
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_memory_efficient_attention()
                    pipe.enable_model_cpu_offload()
                except Exception as e:
                    logger.warning(f"Could not enable GPU optimizations: {e}")
            else:
                # CPU optimizations
                try:
                    pipe.enable_attention_slicing()
                except Exception as e:
                    logger.warning(f"Could not enable CPU optimizations: {e}")

            self.current_model = pipe
            self.current_model_name = model_name
            logger.info(f"✅ Model loaded successfully on {device}")
            return pipe

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

model_manager = ModelManager()

# Quality presets optimized for Render constraints
QUALITY_PRESETS = {
    "draft": {"width": 512, "height": 512, "num_inference_steps": 10, "guidance_scale": 7.0},
    "standard": {"width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 7.5},
    "high": {"width": 768, "height": 768, "num_inference_steps": 25, "guidance_scale": 8.0},
}

# Enhanced generation function with better error handling and validation
def run_generation(job_id: str, model_name: str, prompt: str, quality_settings: dict, negative_prompt: str = None):
    try:
        job_manager.update_job(job_id, {"status": "processing"})
        job_manager.active_generations += 1

        logger.info(f"[{job_id}] 🚀 Starting generation")
        logger.info(f"[{job_id}] Prompt: {prompt}")

        # Get pipeline (this handles model loading/unloading)
        pipe = model_manager.get_pipeline(model_name)

        # Prepare generation parameters with validation
        width = max(256, min(768, quality_settings.get("width", 512)))
        height = max(256, min(768, quality_settings.get("height", 512)))
        num_steps = max(5, min(30, quality_settings.get("num_inference_steps", 20)))
        guidance = max(1.0, min(15.0, quality_settings.get("guidance_scale", 7.5)))

        # Ensure dimensions are divisible by 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance,
            "generator": torch.manual_seed(42),
            "output_type": "pil"
        }

        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        logger.info(f"[{job_id}] Generation params: {width}x{height}, {num_steps} steps")

        # Generate image with proper context management
        device = next(pipe.parameters()).device
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                result = pipe(**pipe_kwargs)
        else:
            with torch.no_grad():
                result = pipe(**pipe_kwargs)

        # Validate the result
        if not hasattr(result, 'images') or not result.images:
            raise Exception("No images generated")

        image = result.images[0]
        if image is None:
            raise Exception("Generated image is None")

        # Convert to bytes
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True, quality=95)

        if buf.tell() == 0:
            raise Exception("Image buffer is empty")

        # Update job with result
        job_manager.update_job(job_id, {
            "status": "done",
            "image": buf.getvalue(),
            "model_used": model_name,
            "completed_at": time.time(),
            "image_size": f"{width}x{height}",
            "steps": num_steps
        })

        logger.info(f"[{job_id}] ✅ Generation completed successfully")

    except Exception as e:
        logger.error(f"[{job_id}] ❌ Generation failed: {str(e)}")
        import traceback
        logger.error(f"[{job_id}] Traceback: {traceback.format_exc()}")
        job_manager.update_job(job_id, {
            "status": "error",
            "error": str(e)
        })
    finally:
        job_manager.active_generations -= 1
        # Cleanup after each generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting AI Image Generator (Render Optimized)")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        logger.info("Running on CPU")
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")
    if model_manager.current_model:
        del model_manager.current_model
    executor.shutdown(wait=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="AI Image Generator",
    version="5.0.0 (Render Edition)",
    lifespan=lifespan
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
        }
    else:
        gpu_info = {"gpu_available": False}

    return {
        "message": "AI Image Generator Backend (Render Edition) 🚀",
        "status": "ready",
        "available_themes": list(model_manager.model_map.keys()),
        "quality_presets": list(QUALITY_PRESETS.keys()),
        "active_jobs": job_manager.active_generations,
        "max_concurrent": MAX_CONCURRENT_GENERATIONS,
        "current_model": model_manager.current_model_name,
        **gpu_info
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "active_generations": job_manager.active_generations,
        "total_jobs": len(job_manager.jobs),
        "gpu_available": torch.cuda.is_available(),
        "current_model": model_manager.current_model_name,
        "timestamp": time.time()
    }

@app.get("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    prompt: str = Query(..., description="Your text prompt for image generation", min_length=1, max_length=500),
    theme: str = Query("realistic", description="Theme/style of the image"),
    quality: str = Query("standard", description="Quality preset"),
    negative_prompt: Optional[str] = Query(None, description="Things to avoid", max_length=200)
):
    # Validation
    if theme not in model_manager.model_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid theme. Available: {list(model_manager.model_map.keys())}"
        )

    if quality not in QUALITY_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quality. Available: {list(QUALITY_PRESETS.keys())}"
        )

    # Check if we're at capacity
    if job_manager.active_generations >= MAX_CONCURRENT_GENERATIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Server at capacity. Max concurrent generations: {MAX_CONCURRENT_GENERATIONS}. Please wait."
        )

    model_name = model_manager.model_map[theme]
    quality_settings = QUALITY_PRESETS[quality].copy()

    # Enhanced prompt with theme-specific improvements
    enhancers = {
        "realistic": "highly detailed, photorealistic, professional photography, sharp focus, realistic lighting, masterpiece",
        "anime": "anime style, vibrant colors, detailed anime art, high quality illustration, anime masterpiece",
        "artistic": "artistic masterpiece, beautiful composition, detailed artwork, professional art, fine art",
        "digital_art": "digital art, concept art, highly detailed, professional digital artwork, artstation trending"
    }

    enhanced_prompt = f"{prompt}, {enhancers.get(theme, 'high quality, detailed, masterpiece')}"

    # Enhanced negative prompt
    base_negative = "blurry, low quality, distorted, ugly, bad anatomy, low resolution, worst quality, normal quality, jpeg artifacts"
    theme_negative = {
        "realistic": "cartoon, anime, painting, drawing, art, rendered",
        "anime": "realistic, photograph, 3d render",
        "artistic": "photograph, realistic, 3d",
        "digital_art": "photograph, realistic"
    }

    full_negative = f"{base_negative}, {theme_negative.get(theme, '')}"
    if negative_prompt:
        full_negative = f"{full_negative}, {negative_prompt}"

    job_id = str(uuid.uuid4())
    job_data = {
        "status": "queued",
        "created": time.time(),
        "prompt": enhanced_prompt,
        "original_prompt": prompt,
        "theme": theme,
        "quality": quality,
        "negative_prompt": full_negative,
        "model_name": model_name
    }

    job_manager.add_job(job_id, job_data)

    # Submit to thread pool
    future = executor.submit(
        run_generation,
        job_id,
        model_name,
        enhanced_prompt,
        quality_settings,
        full_negative
    )

    logger.info(f"[{job_id}] 📝 Job queued for {theme} generation")

    return {
        "job_id": job_id,
        "status": "queued",
        "estimated_time": quality_settings.get("num_inference_steps", 20) * 2,
        "queue_position": job_manager.active_generations,
        "model_loading": model_manager.current_model_name != model_name
    }

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "done" and "image" in job:
        return Response(
            content=job["image"],
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f"inline; filename=generated_{job_id}.png"
            }
        )

    return {
        "status": job["status"],
        "error": job.get("error"),
        "created": job.get("created"),
        "processing_time": time.time() - job.get("created", time.time()) if job["status"] != "done" else None
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Don't return the actual image data in status
    status_job = {k: v for k, v in job.items() if k != "image"}
    if job["status"] == "processing":
        status_job["estimated_remaining"] = max(0, 60 - (time.time() - job.get("created", time.time())))

    return status_job