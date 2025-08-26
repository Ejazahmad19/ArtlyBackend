import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import asyncio
import aiohttp
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Generator", version="1.0.0")

# Add CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not set!")

# Multiple model options for better quality
MODEL_MAP = {
    "realistic": "stabilityai/stable-diffusion-xl-base-1.0",
    "anime": "cagliostrolab/animagine-xl-3.1",
    "artistic": "runwayml/stable-diffusion-v1-5",
    "photorealistic": "SG161222/RealVisXL_V4.0",
    "digital_art": "prompthero/openjourney-v4",
}

BASE_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Quality presets
QUALITY_PRESETS = {
    "draft": {"width": 512, "height": 512, "guidance_scale": 7.5, "num_inference_steps": 20},
    "standard": {"width": 768, "height": 768, "guidance_scale": 7.5, "num_inference_steps": 30},
    "high": {"width": 1024, "height": 1024, "guidance_scale": 9.0, "num_inference_steps": 40},
    "ultra": {"width": 1536, "height": 1536, "guidance_scale": 10.0, "num_inference_steps": 50},
}

@app.get("/")
def root():
    return {
        "message": "AI Image Generator Backend is running 🚀",
        "available_themes": list(MODEL_MAP.keys()),
        "quality_presets": list(QUALITY_PRESETS.keys())
    }

@app.get("/models")
def get_available_models():
    """Get available themes/models"""
    return {
        "themes": MODEL_MAP,
        "quality_presets": QUALITY_PRESETS
    }

async def wait_for_model(api_url: str, max_wait: int = 300) -> bool:
    """Wait for model to load if it's sleeping"""
    async with aiohttp.ClientSession() as session:
        for _ in range(max_wait // 10):
            try:
                async with session.post(
                    api_url,
                    headers=headers,
                    json={"inputs": "test", "parameters": {"width": 512, "height": 512}},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return True
                    elif response.status == 503:
                        # Model is loading
                        await asyncio.sleep(10)
                        continue
                    else:
                        return False
            except Exception:
                await asyncio.sleep(10)
                continue
    return False

@app.get("/generate")
async def generate(
    prompt: str = Query(..., description="Your text prompt for image generation"),
    theme: str = Query("realistic", description="Theme/style of the image"),
    quality: str = Query("standard", description="Quality preset (draft, standard, high, ultra)"),
    width: Optional[int] = Query(None, description="Custom width (overrides quality preset)"),
    height: Optional[int] = Query(None, description="Custom height (overrides quality preset)"),
    guidance_scale: Optional[float] = Query(None, description="Guidance scale (7.5-15.0)"),
    num_steps: Optional[int] = Query(None, description="Number of inference steps"),
    negative_prompt: str = Query(
        "blurry, low quality, distorted, deformed, extra limbs, bad anatomy, text, watermark", 
        description="Negative prompt to avoid unwanted elements"
    ),
    seed: Optional[int] = Query(None, description="Seed for reproducible results"),
    enhance_prompt: bool = Query(True, description="Automatically enhance the prompt")
):
    try:
        # Validate inputs
        if theme not in MODEL_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid theme. Available: {list(MODEL_MAP.keys())}")
        
        if quality not in QUALITY_PRESETS:
            raise HTTPException(status_code=400, detail=f"Invalid quality. Available: {list(QUALITY_PRESETS.keys())}")

        # Get model URL
        model_name = MODEL_MAP[theme]
        api_url = f"{BASE_URL}{model_name}"

        # Get quality settings
        quality_settings = QUALITY_PRESETS[quality].copy()

        # Override with custom settings if provided
        if width:
            quality_settings["width"] = min(max(width, 256), 2048)  # Clamp between 256-2048
        if height:
            quality_settings["height"] = min(max(height, 256), 2048)
        if guidance_scale:
            quality_settings["guidance_scale"] = min(max(guidance_scale, 1.0), 20.0)
        if num_steps:
            quality_settings["num_inference_steps"] = min(max(num_steps, 10), 100)

        # Enhance prompt if requested
        if enhance_prompt:
            prompt_enhancers = {
                "realistic": "highly detailed, professional photography, sharp focus, realistic lighting, 8k uhd",
                "anime": "anime style, detailed, vibrant colors, manga illustration, high quality",
                "artistic": "artistic, creative, detailed artwork, masterpiece, high resolution",
                "photorealistic": "photorealistic, ultra detailed, professional photo, sharp, clear",
                "digital_art": "digital art, concept art, detailed illustration, artstation trending"
            }
            enhancement = prompt_enhancers.get(theme, "high quality, detailed")
            full_prompt = f"{prompt}, {enhancement}"
        else:
            full_prompt = f"{prompt}, {theme} style"

        # Prepare payload
        payload = {
            "inputs": full_prompt,
            "parameters": {
                **quality_settings,
                "negative_prompt": negative_prompt,
            }
        }

        # Add seed if provided
        if seed is not None:
            payload["parameters"]["seed"] = seed

        logger.info(f"Generating image with prompt: {full_prompt[:100]}...")
        
        # Check if model needs to wake up
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status == 503:
                    # Model is loading, wait for it
                    logger.info("Model is loading, waiting...")
                    if await wait_for_model(api_url):
                        # Try again after model loads
                        async with session.post(api_url, headers=headers, json=payload) as retry_response:
                            if retry_response.status == 200:
                                content = await retry_response.read()
                                generation_time = time.time() - start_time
                                logger.info(f"Image generated successfully in {generation_time:.2f}s")
                                
                                return Response(
                                    content=content, 
                                    media_type="image/png",
                                    headers={
                                        "X-Generation-Time": str(generation_time),
                                        "X-Model-Used": model_name,
                                        "X-Quality": quality
                                    }
                                )
                            else:
                                error_text = await retry_response.text()
                                logger.error(f"Error after retry: {error_text}")
                                raise HTTPException(status_code=500, detail=f"Generation failed: {error_text}")
                    else:
                        raise HTTPException(status_code=503, detail="Model loading timeout")
                
                elif response.status == 200:
                    content = await response.read()
                    generation_time = time.time() - start_time
                    logger.info(f"Image generated successfully in {generation_time:.2f}s")
                    
                    return Response(
                        content=content, 
                        media_type="image/png",
                        headers={
                            "X-Generation-Time": str(generation_time),
                            "X-Model-Used": model_name,
                            "X-Quality": quality
                        }
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Generation error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=error_text)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)