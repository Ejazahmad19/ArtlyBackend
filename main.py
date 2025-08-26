import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Generator", version="1.0.0")

# Add CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not set!")

# Simplified model options (using models that are more likely to work)
MODEL_MAP = {
    "realistic": "stabilityai/stable-diffusion-xl-base-1.0",
    "anime": "stabilityai/stable-diffusion-xl-base-1.0",  # Use same model with different prompts
    "artistic": "stabilityai/stable-diffusion-xl-base-1.0",
    "digital_art": "stabilityai/stable-diffusion-xl-base-1.0",
}

BASE_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Quality presets (more conservative sizes)
QUALITY_PRESETS = {
    "draft": {"width": 512, "height": 512},
    "standard": {"width": 768, "height": 768},
    "high": {"width": 1024, "height": 1024},
    "ultra": {"width": 1024, "height": 1024},  # Keep same as high for stability
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

def wait_for_model(api_url: str, max_wait: int = 120) -> bool:
    """Wait for model to load if it's sleeping - synchronous version"""
    for i in range(max_wait // 10):
        try:
            # Try a simple test request
            test_payload = {"inputs": "test"}
            response = requests.post(api_url, headers=headers, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                return True
            elif response.status_code == 503:
                logger.info(f"Model loading... attempt {i+1}/{max_wait//10}")
                time.sleep(10)
                continue
            else:
                return False
        except Exception as e:
            logger.warning(f"Wait attempt {i+1} failed: {e}")
            time.sleep(10)
            continue
    return False

@app.get("/generate")
def generate(
    prompt: str = Query(..., description="Your text prompt for image generation"),
    theme: str = Query("realistic", description="Theme/style of the image"),
    quality: str = Query("standard", description="Quality preset (draft, standard, high, ultra)"),
    width: Optional[int] = Query(None, description="Custom width (overrides quality preset)"),
    height: Optional[int] = Query(None, description="Custom height (overrides quality preset)"),
    negative_prompt: str = Query(
        "blurry, low quality, distorted, deformed", 
        description="Negative prompt to avoid unwanted elements"
    ),
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
            quality_settings["width"] = min(max(width, 256), 1024)  # Limit to 1024 for stability
        if height:
            quality_settings["height"] = min(max(height, 256), 1024)

        # Enhance prompt based on theme
        if enhance_prompt:
            prompt_enhancers = {
                "realistic": "highly detailed, professional photography, sharp focus, realistic, 8k uhd",
                "anime": "anime style, detailed, vibrant colors, manga illustration, anime art style",
                "artistic": "artistic, creative, detailed artwork, masterpiece, oil painting style",
                "digital_art": "digital art, concept art, detailed illustration, trending on artstation"
            }
            enhancement = prompt_enhancers.get(theme, "high quality, detailed")
            full_prompt = f"{prompt}, {enhancement}"
        else:
            full_prompt = f"{prompt}, {theme} style"

        # Prepare payload - keep it simple
        payload = {
            "inputs": full_prompt,
            "parameters": quality_settings
        }

        # Add negative prompt only if the model supports it
        if negative_prompt and theme == "realistic":
            payload["parameters"]["negative_prompt"] = negative_prompt

        logger.info(f"Generating image with prompt: {full_prompt[:100]}...")
        
        start_time = time.time()
        
        # Make the request
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 503:
            # Model is loading, wait for it
            logger.info("Model is loading, waiting...")
            if wait_for_model(api_url):
                # Try again after model loads
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    generation_time = time.time() - start_time
                    logger.info(f"Image generated successfully in {generation_time:.2f}s")
                    
                    return Response(
                        content=response.content, 
                        media_type="image/png",
                        headers={
                            "X-Generation-Time": str(generation_time),
                            "X-Model-Used": model_name,
                            "X-Quality": quality
                        }
                    )
                else:
                    logger.error(f"Error after retry: {response.text}")
                    raise HTTPException(status_code=500, detail=f"Generation failed: {response.text}")
            else:
                raise HTTPException(status_code=503, detail="Model loading timeout")
        
        elif response.status_code == 200:
            generation_time = time.time() - start_time
            logger.info(f"Image generated successfully in {generation_time:.2f}s")
            
            return Response(
                content=response.content, 
                media_type="image/png",
                headers={
                    "X-Generation-Time": str(generation_time),
                    "X-Model-Used": model_name,
                    "X-Quality": quality
                }
            )
        else:
            logger.error(f"Generation error: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        raise HTTPException(status_code=408, detail="Request timeout - try again or use lower quality")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Simple test endpoint
@app.get("/test")
def test_generation():
    """Test endpoint with a simple prompt"""
    try:
        api_url = f"{BASE_URL}stabilityai/stable-diffusion-xl-base-1.0"
        payload = {
            "inputs": "a beautiful sunset, realistic style",
            "parameters": {"width": 512, "height": 512}
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return Response(content=response.content, media_type="image/png")
        else:
            return {"error": response.text, "status_code": response.status_code}
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)