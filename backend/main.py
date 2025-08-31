import logging
from contextlib import asynccontextmanager
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import time
from typing import Optional, List
from pathlib import Path

import torch
from transformers import AutoProcessor, DiaForConditionalGeneration

from utils import process_audio_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR = Path("upload_files")
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

# Clean up old files on startup (keep last 100 files)
def cleanup_old_files(directory: Path, max_files: int = 100):
    try:
        files = sorted(directory.glob("*.wav"), key=os.path.getctime)
        if len(files) > max_files:
            for file in files[:-max_files]:
                file.unlink()
                logger.info(f"Cleaned up old file: {file}")
    except Exception as e:
        logger.warning(f"Error cleaning up old files: {e}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using DEVICE: {DEVICE}")

class ModelManager:
    """Manages the loading, unloading and access to the Dia model and processor using Hugging Face Transformers."""

    def __init__(self):
        self.device = DEVICE
        self.dtype_map = {
            "cpu": torch.float32,
            "cuda": torch.float16,  
        }
        self.model = None
        self.processor = None
        self.model_id = "nari-labs/Dia-1.6B-0626"
        self.is_loaded = False

    def load_model(self):
        """Load the Dia model and processor with appropriate configuration using Hugging Face Transformers."""
        try:
            if self.is_loaded:
                logger.info("Model is already loaded")
                return
                
            dtype = self.dtype_map.get(self.device, torch.float16)
            logger.info(f"Loading model and processor with {dtype} on {self.device}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            self.model = DiaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            
            self.is_loaded = True
            logger.info("Model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model or processor: {e}")
            self.is_loaded = False
            raise

    def unload_model(self):
        """Cleanup method to properly unload the model and processor."""
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
                
            self.is_loaded = False
            logger.info("Model and processor unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading model or processor: {e}")

    def get_model(self):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_processor(self):
        if not self.is_loaded or self.processor is None:
            raise RuntimeError("Processor not loaded. Call load_model() first.")
        return self.processor

model_manager = ModelManager()

class AudioPrompt(BaseModel):
    sample_rate: int
    audio_data: List[float]  

class GenerateRequest(BaseModel):
    text_input: str
    audio_prompt: Optional[AudioPrompt] = None
    max_new_tokens: int = 1024
    cfg_scale: float = 3.0
    temperature: float = 1.3
    top_p: float = 0.95
    cfg_filter_top_k: int = 35
    speed_factor: float = 0.94

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handle model lifecycle during application startup and shutdown."""
    logger.info("Starting up application...")
    cleanup_old_files(AUDIO_DIR)
    model_manager.load_model()
    yield
    logger.info("Shutting down application...")
    model_manager.unload_model()
    logger.info("Application shut down successfully")

app = FastAPI(
    title="Dia Text-to-Speech API",
    description="API for generating speech using Dia model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "Backend is running",
        "device": DEVICE,
        "model_loaded": model_manager.is_loaded
    }

@app.post("/api/generate")
async def run_inference(request: GenerateRequest):
    """
    Runs Dia inference using the model and processor from model_manager and provided inputs.
    Uses temporary files for audio prompt compatibility with inference.generate.
    """
    if not request.text_input or request.text_input.strip() == "":
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    # Validate text input length
    if len(request.text_input) > 1000:
        raise HTTPException(status_code=400, detail="Text input too long. Maximum 1000 characters allowed.")

    # Generate unique filename
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    output_filepath = AUDIO_DIR / f"{timestamp}_{unique_id}.wav"

    prompt_path_for_generate = None
    
    try:
        # Process audio prompt if provided
        if request.audio_prompt is not None:
            prompt_path_for_generate = process_audio_prompt(request.audio_prompt)
            logger.info(f"Audio prompt processed: {prompt_path_for_generate}")

        model = model_manager.get_model()
        processor = model_manager.get_processor()

        start_time = time.time()

        # Prepare inputs
        processor_inputs = processor(
            text=[request.text_input],
            padding=True,
            return_tensors="pt"
        )
        processor_inputs = {k: v.to(model.device) for k, v in processor_inputs.items()}

        if prompt_path_for_generate is not None:
            processor_inputs["audio_prompt"] = prompt_path_for_generate

        # Generate audio
        with torch.inference_mode():
            logger.info(f"Starting generation with parameters: max_tokens={request.max_new_tokens}, "
                       f"cfg_scale={request.cfg_scale}, temperature={request.temperature}")
            
            outputs = model.generate(
                **processor_inputs,
                max_new_tokens=request.max_new_tokens,
                guidance_scale=request.cfg_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.cfg_filter_top_k
            )
            
            logger.info(f"Generation completed. Output type: {type(outputs)}")

        # Decode and save audio
        decoded = processor.batch_decode(outputs)
        processor.save_audio(decoded, str(output_filepath))
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"Audio saved to {output_filepath}")
        logger.info(f"Generation finished in {generation_time:.2f} seconds.")

        # Return file response
        return FileResponse(
            path=str(output_filepath),
            media_type="audio/wav",
            filename=output_filepath.name,
            headers={
                "X-Generation-Time": f"{generation_time:.2f}",
                "X-File-Size": f"{output_filepath.stat().st_size}"
            }
        )

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory error")
            raise HTTPException(status_code=500, detail="GPU memory exhausted. Try reducing input size.")
        else:
            logger.error(f"Runtime error during inference: {e}")
            raise HTTPException(status_code=500, detail=f"Runtime error: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
    finally:
        # Clean up temporary audio prompt file
        if prompt_path_for_generate and os.path.exists(prompt_path_for_generate):
            try:
                os.remove(prompt_path_for_generate)
                logger.info(f"Cleaned up temporary file: {prompt_path_for_generate}")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file: {e}")

@app.get("/api/stats")
async def get_stats():
    """Get statistics about generated files and system status."""
    try:
        audio_files = list(AUDIO_DIR.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in audio_files)
        
        return {
            "total_files": len(audio_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "device": DEVICE,
            "model_loaded": model_manager.is_loaded,
            "cuda_available": torch.cuda.is_available()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")
