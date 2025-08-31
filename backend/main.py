import logging
from contextlib import asynccontextmanager
import os
import uuid
import json
import re

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import time
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoProcessor, DiaForConditionalGeneration
import soundfile as sf
import numpy as np
from scipy import signal

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

class VapiMessage(BaseModel):
    timestamp: int
    type: str
    status: str
    role: str
    turn: int
    artifact: Dict[str, Any]
    call: Dict[str, Any]
    assistant: Dict[str, Any]

class VapiRequest(BaseModel):
    message: VapiMessage

# Sound effects mapping from your Svelte component
SOUND_EFFECTS = {
    'burps': {'type': 'short', 'duration': 0.8, 'pitch_shift': -4},
    'clears throat': {'type': 'throat', 'duration': 1.2, 'pitch_shift': 0},
    'coughs': {'type': 'short', 'duration': 0.7, 'pitch_shift': 2},
    'exhales': {'type': 'breath', 'duration': 1.5, 'pitch_shift': -2},
    'gasps': {'type': 'breath', 'duration': 0.9, 'pitch_shift': 6},
    'groans': {'type': 'long', 'duration': 1.8, 'pitch_shift': -6},
    'humming': {'type': 'tone', 'duration': 2.0, 'pitch_shift': 0},
    'laughs': {'type': 'laugh', 'duration': 2.5, 'pitch_shift': 4},
    'mumbles': {'type': 'mumble', 'duration': 1.3, 'pitch_shift': -3},
    'screams': {'type': 'scream', 'duration': 1.1, 'pitch_shift': 8},
    'sighs': {'type': 'breath', 'duration': 2.0, 'pitch_shift': -4},
    'sneezes': {'type': 'short', 'duration': 0.6, 'pitch_shift': 10},
}

def generate_sound_effect(effect_name: str, samplerate: int = 24000) -> np.ndarray:
    """Generate a synthetic sound effect based on the effect name."""
    effect_config = SOUND_EFFECTS.get(effect_name, {})
    duration = effect_config.get('duration', 1.0)
    pitch_shift = effect_config.get('pitch_shift', 0)
    effect_type = effect_config.get('type', 'short')
    
    samples = int(duration * samplerate)
    t = np.linspace(0, duration, samples)
    
    if effect_type == 'short':
        # Short burst sound
        freq = 200 + pitch_shift * 20
        sound = np.sin(2 * np.pi * freq * t) * np.exp(-5 * t)
        
    elif effect_type == 'long':
        # Longer sustained sound
        freq = 150 + pitch_shift * 15
        sound = np.sin(2 * np.pi * freq * t) * (1 - t/duration)
        
    elif effect_type == 'breath':
        # Breath-like sound
        freq = 300 + pitch_shift * 25
        envelope = np.exp(-2 * t) * (1 - np.exp(-10 * t))
        sound = np.sin(2 * np.pi * freq * t) * envelope
        
    elif effect_type == 'throat':
        # Throat clearing sound
        freq1 = 100 + pitch_shift * 10
        freq2 = 300 + pitch_shift * 30
        sound = (0.7 * np.sin(2 * np.pi * freq1 * t) + 
                0.3 * np.sin(2 * np.pi * freq2 * t)) * np.exp(-4 * t)
        
    elif effect_type == 'laugh':
        # Laughing sound with variations
        base_freq = 250 + pitch_shift * 20
        mod_freq = 5  # Laugh modulation frequency
        sound = np.sin(2 * np.pi * base_freq * t) * np.sin(2 * np.pi * mod_freq * t)
        sound *= np.exp(-1.5 * t)
        
    elif effect_type == 'mumble':
        # Mumbling sound
        freq = 200 + pitch_shift * 15
        mod = 8  # Modulation for mumble effect
        sound = np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * mod * t))
        sound *= np.exp(-3 * t)
        
    elif effect_type == 'scream':
        # Scream sound
        freq_start = 300 + pitch_shift * 25
        freq_end = 800 + pitch_shift * 50
        freq = np.linspace(freq_start, freq_end, len(t))
        sound = np.sin(2 * np.pi * freq * t) * np.exp(-2 * t)
        
    elif effect_type == 'tone':
        # Simple tone for humming
        freq = 260 + pitch_shift * 20
        sound = np.sin(2 * np.pi * freq * t) * 0.8
        
    else:
        # Default short sound
        freq = 250 + pitch_shift * 20
        sound = np.sin(2 * np.pi * freq * t) * np.exp(-4 * t)
    
    # Normalize and add some noise for realism
    sound = sound / np.max(np.abs(sound)) * 0.7
    if effect_type not in ['tone']:  # Don't add noise to pure tones
        sound += np.random.normal(0, 0.02, len(sound))
    
    return sound

def add_sound_effect_to_audio(audio_path: str, effect_name: str, position: str = "end") -> str:
    """Add a sound effect to the generated audio at specified position."""
    try:
        # Read the original audio
        data, samplerate = sf.read(audio_path)
        
        # Generate the sound effect
        effect_audio = generate_sound_effect(effect_name, samplerate)
        
        # Ensure both arrays are 1D
        if len(data.shape) > 1:
            data = data[:, 0]  # Take first channel if stereo
        
        # Add silence before and after effect
        silence_duration = 0.1  # 100ms silence
        silence_samples = int(silence_duration * samplerate)
        silence = np.zeros(silence_samples)
        
        # Create the effect with silence padding
        padded_effect = np.concatenate([silence, effect_audio, silence])
        
        if position == "end":
            # Add effect to the end
            combined_audio = np.concatenate([data, padded_effect])
        elif position == "beginning":
            # Add effect to the beginning
            combined_audio = np.concatenate([padded_effect, data])
        else:
            # Add effect at a specific time (middle by default)
            insert_point = len(data) // 2
            combined_audio = np.concatenate([
                data[:insert_point],
                padded_effect,
                data[insert_point:]
            ])
        
        # Normalize to prevent clipping
        combined_audio = combined_audio / np.max(np.abs(combined_audio)) * 0.9
        
        # Save the modified audio
        output_path = audio_path.replace('.wav', f'_{effect_name.replace(" ", "_")}.wav')
        sf.write(output_path, combined_audio, samplerate)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error adding sound effect '{effect_name}': {e}")
        return audio_path  # Return original if error

def extract_text_and_effects(text: str) -> tuple:
    """Extract main text and sound effects from text containing (effect) notations."""
    # Pattern to match sound effects in parentheses
    pattern = r'\(([^)]+)\)'
    effects = re.findall(pattern, text)
    
    # Remove effects from the main text
    clean_text = re.sub(pattern, '', text).strip()
    
    # Filter only known effects
    valid_effects = [effect for effect in effects if effect in SOUND_EFFECTS]
    
    return clean_text, valid_effects

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handle model lifecycle during application startup and shutdown."""
    logger.info("Starting up application...")
    model_manager.load_model()
    yield
    logger.info("Shutting down application...")
    model_manager.unload_model()
    logger.info("Application shut down successfully")

app = FastAPI(
    title="Dia Text-to-Speech API",
    description="API for generating speech using Dia model with sound effects",
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
        "model_loaded": model_manager.is_loaded,
        "supported_effects": list(SOUND_EFFECTS.keys())
    }

def extract_text_from_vapi_payload(vapi_request: VapiRequest) -> str:
    """Extract the text to synthesize from Vapi payload."""
    try:
        # Get the last assistant message
        messages = vapi_request.message.artifact.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("message"):
                return msg["message"]
        
        # Fallback to OpenAI formatted messages
        openai_messages = vapi_request.message.artifact.get("messagesOpenAIFormatted", [])
        for msg in reversed(openai_messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        
        return ""
        
    except Exception as e:
        logger.error(f"Error extracting text from Vapi payload: {e}")
        return ""

@app.post("/api/generate")
async def run_inference(
    request: VapiRequest,
    x_effect_type: Optional[str] = Header(None, alias="X-Effect-Type")
):
    """
    Handle Vapi speech-update requests and generate speech using Dia model with sound effects.
    """
    logger.info(f"Received Vapi request type: {request.message.type}, status: {request.message.status}")
    
    # Only process 'stopped' status for speech generation
    if request.message.status != "stopped":
        logger.info(f"Ignoring non-stopped status: {request.message.status}")
        return JSONResponse(content={"status": "ignored", "reason": "only stopped status processed"})
    
    # Extract text from Vapi payload
    text_to_synthesize = extract_text_from_vapi_payload(request)
    
    if not text_to_synthesize or text_to_synthesize.strip() == "":
        logger.warning("No text found to synthesize in Vapi payload")
        return JSONResponse(content={"status": "ignored", "reason": "no text to synthesize"})

    # Extract text and effects
    clean_text, detected_effects = extract_text_and_effects(text_to_synthesize)
    
    if not clean_text:
        logger.warning("No clean text found after removing effects")
        return JSONResponse(content={"status": "ignored", "reason": "no text after effect removal"})

    # Generate unique filename
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    output_filepath = AUDIO_DIR / f"{timestamp}_{unique_id}.wav"

    try:
        model = model_manager.get_model()
        processor = model_manager.get_processor()

        start_time = time.time()

        # Prepare inputs with clean text (without effects)
        processor_inputs = processor(
            text=[clean_text],
            padding=True,
            return_tensors="pt"
        )
        processor_inputs = {k: v.to(model.device) for k, v in processor_inputs.items()}

        # Generate audio
        with torch.inference_mode():
            logger.info(f"Starting generation for text: '{clean_text}'")
            logger.info(f"Detected effects: {detected_effects}")
            
            outputs = model.generate(
                **processor_inputs,
                max_new_tokens=1024,
                guidance_scale=3.0,
                temperature=1.3,
                top_p=0.95,
                top_k=35
            )

        # Decode and save audio
        decoded = processor.batch_decode(outputs)
        processor.save_audio(decoded, str(output_filepath))
        
        # Apply sound effects if any were detected
        final_output_path = str(output_filepath)
        if detected_effects:
            for effect in detected_effects:
                logger.info(f"Adding sound effect: {effect}")
                final_output_path = add_sound_effect_to_audio(final_output_path, effect, "end")
                # Remove intermediate files
                if final_output_path != str(output_filepath) and os.path.exists(str(output_filepath)):
                    os.remove(str(output_filepath))
        
        # Apply header-based effect if specified
        if x_effect_type and x_effect_type in SOUND_EFFECTS:
            logger.info(f"Adding header-based effect: {x_effect_type}")
            final_output_path = add_sound_effect_to_audio(final_output_path, x_effect_type, "end")
            if final_output_path != str(output_filepath) and os.path.exists(str(output_filepath)):
                os.remove(str(output_filepath))
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"Final audio saved to {final_output_path}")
        logger.info(f"Generation finished in {generation_time:.2f} seconds.")

        # Return file response
        return FileResponse(
            path=final_output_path,
            media_type="audio/wav",
            filename=Path(final_output_path).name,
            headers={
                "X-Generation-Time": f"{generation_time:.2f}",
                "X-File-Size": f"{Path(final_output_path).stat().st_size}",
                "X-Effects-Applied": ",".join(detected_effects + ([x_effect_type] if x_effect_type else []))
            }
        )

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/generate-direct")
async def run_inference_direct(request: GenerateRequest):
    """
    Original endpoint for direct API calls with explicit parameters.
    """
    if not request.text_input or request.text_input.strip() == "":
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    # Extract text and effects
    clean_text, detected_effects = extract_text_and_effects(request.text_input)
    
    if not clean_text:
        raise HTTPException(status_code=400, detail="No text found after removing effects")

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

        # Prepare inputs with clean text
        processor_inputs = processor(
            text=[clean_text],
            padding=True,
            return_tensors="pt"
        )
        processor_inputs = {k: v.to(model.device) for k, v in processor_inputs.items()}

        if prompt_path_for_generate is not None:
            processor_inputs["audio_prompt"] = prompt_path_for_generate

        # Generate audio
        with torch.inference_mode():
            logger.info(f"Starting generation for text: '{clean_text}'")
            logger.info(f"Detected effects: {detected_effects}")
            
            outputs = model.generate(
                **processor_inputs,
                max_new_tokens=request.max_new_tokens,
                guidance_scale=request.cfg_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.cfg_filter_top_k
            )

        # Decode and save audio
        decoded = processor.batch_decode(outputs)
        processor.save_audio(decoded, str(output_filepath))
        
        # Apply sound effects if any were detected
        final_output_path = str(output_filepath)
        if detected_effects:
            for effect in detected_effects:
                logger.info(f"Adding sound effect: {effect}")
                final_output_path = add_sound_effect_to_audio(final_output_path, effect, "end")
                if final_output_path != str(output_filepath) and os.path.exists(str(output_filepath)):
                    os.remove(str(output_filepath))
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"Final audio saved to {final_output_path}")
        logger.info(f"Generation finished in {generation_time:.2f} seconds.")

        # Return file response
        return FileResponse(
            path=final_output_path,
            media_type="audio/wav",
            filename=Path(final_output_path).name,
            headers={
                "X-Generation-Time": f"{generation_time:.2f}",
                "X-File-Size": f"{Path(final_output_path).stat().st_size}",
                "X-Effects-Applied": ",".join(detected_effects)
            }
        )

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

@app.get("/api/effects")
async def get_supported_effects():
    """Get list of supported sound effects."""
    return {
        "effects": list(SOUND_EFFECTS.keys()),
        "descriptions": {
            "burps": "Short burping sound",
            "clears throat": "Throat clearing sound",
            "coughs": "Coughing sound",
            "exhales": "Exhalation breath",
            "gasps": "Gasp of surprise",
            "groans": "Groaning sound",
            "humming": "Humming tone",
            "laughs": "Laughter",
            "mumbles": "Mumbling speech",
            "screams": "Scream",
            "sighs": "Sigh of relief",
            "sneezes": "Sneezing sound"
        }
    }

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

# Add this to your requirements.txt:
# soundfile==0.12.1
# scipy==1.13.1
# numpy==1.26.4
