# main.py
import logging
from contextlib import asynccontextmanager
import os
import uuid
import re
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

import torch
from transformers import AutoProcessor, DiaForConditionalGeneration

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import anyio

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("dia-tts")

# =============================================================================
# FS setup
# =============================================================================
AUDIO_DIR = Path("audio_files"); AUDIO_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR = Path("upload_files"); UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# Torch device/dtype
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"DEVICE: {DEVICE}")

# =============================================================================
# Environment / security
# =============================================================================
VAPI_SECRET_EXPECTED = os.getenv("VAPI_SERVER_SECRET", "").strip()
DISABLE_SFX_ENV = os.getenv("DISABLE_SFX", "0").strip() == "1"

# =============================================================================
# Model manager
# =============================================================================
class ModelManager:
    def __init__(self):
        self.device = DEVICE
        self.dtype_map = {"cpu": torch.float32, "cuda": torch.float16}
        self.model = None
        self.processor = None
        self.model_id = os.getenv("DIA_MODEL_ID", "nari-labs/Dia-1.6B-0626")
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        try:
            dtype = self.dtype_map.get(self.device, torch.float16)
            device_map = "auto" if torch.cuda.is_available() else None
            logger.info(f"Loading Dia model ({self.model_id}) dtype={dtype} device_map={device_map}")
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = DiaForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
            )
            self.is_loaded = True
            logger.info("Dia model & processor loaded")
        except Exception as e:
            logger.error(f"Load error: {e}", exc_info=True)
            self.is_loaded = False
            raise

    def unload_model(self):
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_loaded = False
            logger.info("Model unloaded & CUDA cache cleared")
        except Exception as e:
            logger.error(f"Unload error: {e}")

    def get_model(self):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model

    def get_processor(self):
        if not self.is_loaded or self.processor is None:
            raise RuntimeError("Processor not loaded.")
        return self.processor


model_manager = ModelManager()

# =============================================================================
# Schemas
# =============================================================================
class AudioPrompt(BaseModel):
    sample_rate: int
    audio_data: List[float]

class GenerateRequest(BaseModel):
    text_input: str
    audio_prompt: Optional[AudioPrompt] = None
    max_new_tokens: int = 400
    cfg_scale: float = 3.5
    temperature: float = 0.9
    top_p: float = 0.9
    cfg_filter_top_k: int = 50

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

# =============================================================================
# Sound effects & DSP helpers
# =============================================================================
# Safer, shorter SFX to minimize audible tremolo/artifacts
SOUND_EFFECTS = {
    'burps': {'type': 'short', 'duration': 0.5, 'pitch_shift': -4, 'gain': 0.5},
    'clears throat': {'type': 'throat', 'duration': 0.7, 'pitch_shift': 0, 'gain': 0.5},
    'coughs': {'type': 'short', 'duration': 0.5, 'pitch_shift': 2, 'gain': 0.45},
    'exhales': {'type': 'breath', 'duration': 0.9, 'pitch_shift': -2, 'gain': 0.35},
    'gasps': {'type': 'breath', 'duration': 0.6, 'pitch_shift': 6, 'gain': 0.4},
    'groans': {'type': 'long', 'duration': 1.0, 'pitch_shift': -6, 'gain': 0.35},
    'humming': {'type': 'tone', 'duration': 1.0, 'pitch_shift': 0, 'gain': 0.25},
    'laughs': {'type': 'laugh', 'duration': 1.2, 'pitch_shift': 4, 'gain': 0.35},
    'mumbles': {'type': 'mumble', 'duration': 0.8, 'pitch_shift': -3, 'gain': 0.3},
    'screams': {'type': 'scream', 'duration': 0.7, 'pitch_shift': 8, 'gain': 0.35},
    'sighs': {'type': 'breath', 'duration': 1.0, 'pitch_shift': -4, 'gain': 0.35},
    'sneezes': {'type': 'short', 'duration': 0.5, 'pitch_shift': 10, 'gain': 0.4},
}
EFFECT_ALIASES = {k: k for k in SOUND_EFFECTS}

def generate_sound_effect(effect_name: str, sr: int) -> np.ndarray:
    cfg = SOUND_EFFECTS.get(effect_name, {})
    dur = cfg.get('duration', 0.6)
    ps = cfg.get('pitch_shift', 0)
    typ = cfg.get('type', 'short')
    gain = float(cfg.get('gain', 0.35))
    n = int(max(0.1, dur) * sr)
    t = np.linspace(0, max(0.1, dur), n, endpoint=False)

    if typ == 'short':
        f = 200 + ps*20; y = np.sin(2*np.pi*f*t)*np.exp(-6*t)
    elif typ == 'long':
        f = 150 + ps*15; y = np.sin(2*np.pi*f*t)*(1 - t/max(0.1, dur))
    elif typ == 'breath':
        f = 300 + ps*25; env = np.exp(-3*t)*(1 - np.exp(-12*t)); y = np.sin(2*np.pi*f*t)*env
    elif typ == 'throat':
        f1 = 120 + ps*10; f2 = 280 + ps*25
        y = (0.7*np.sin(2*np.pi*f1*t) + 0.3*np.sin(2*np.pi*f2*t))*np.exp(-5*t)
    elif typ == 'laugh':
        base = 230 + ps*20; mod = 4.5
        y = np.sin(2*np.pi*base*t)*np.sin(2*np.pi*mod*t)*np.exp(-2.0*t)
    elif typ == 'mumble':
        f = 190 + ps*15; mod = 7
        y = np.sin(2*np.pi*f*t)*(0.5 + 0.5*np.sin(2*np.pi*mod*t))*np.exp(-4*t)
    elif typ == 'scream':
        f = np.linspace(300+ps*25, 750+ps*50, n); y = np.sin(2*np.pi*f*t)*np.exp(-3*t)
    elif typ == 'tone':
        f = 260 + ps*20; y = np.sin(2*np.pi*f*t)
    else:
        f = 250 + ps*20; y = np.sin(2*np.pi*f*t)*np.exp(-5*t)

    y = y / (np.max(np.abs(y)) + 1e-12)
    y *= gain  # overall lower gain to keep subtle
    if typ != 'tone':
        y += np.random.normal(0, 0.01, size=n)
    return y.astype(np.float64)

def _highpass(y: np.ndarray, sr: int, fc=80.0) -> np.ndarray:
    b, a = signal.butter(2, fc/(sr/2), btype="highpass")
    return signal.lfilter(b, a, y)

def _compress(y: np.ndarray, thr_db=-18.0, ratio=3.0) -> np.ndarray:
    thr = 10**(thr_db/20)
    env = np.maximum(np.abs(y), 1e-12)
    gain = np.where(env > thr, (thr + (env-thr)/ratio)/env, 1.0)
    return y * gain

def _limit_norm(y: np.ndarray, peak_db=-0.3) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-12
    target = 10**(peak_db/20)
    return (y/peak) * target

def _gate(y: np.ndarray, floor_db=-55.0) -> np.ndarray:
    g = 10**(floor_db/20)
    y = y.copy()
    y[np.abs(y) < g] = 0.0
    return y

def _trim_edges(y: np.ndarray, sr: int, head_ms=200, tail_ms=200) -> np.ndarray:
    eps = 1e-4; i0=0
    while i0 < len(y) and abs(y[i0]) < eps: i0 += 1
    i1 = len(y)-1
    while i1 > 0 and abs(y[i1]) < eps: i1 -= 1
    if i1 <= i0: return y
    head = int(sr*head_ms/1000); tail = int(sr*tail_ms/1000)
    i0 = max(0, i0-head); i1 = min(len(y)-1, i1+tail)
    return y[i0:i1+1]

def _mix_with_ducking(voice: np.ndarray, sfx: np.ndarray, sr: int, pos="end", duck_db=1.5) -> np.ndarray:
    if pos == "end":
        insert = len(voice) - int(0.2*sr) if len(voice) > int(0.5*sr) else len(voice)//2
    elif pos == "beginning":
        insert = int(0.15*sr)
    else:
        insert = len(voice)//2
    insert = max(0, min(insert, len(voice)))

    out_len = max(len(voice), insert+len(sfx))
    out = np.zeros(out_len, dtype=np.float64)
    out[:len(voice)] = voice

    fade = int(0.02*sr)
    sfx = sfx.copy()
    if len(sfx) > 2*fade:
        sfx[:fade] *= np.linspace(0,1,fade)
        sfx[-fade:] *= np.linspace(1,0,fade)

    duck = 10**(-duck_db/20)
    s, e = insert, insert+len(sfx)
    out[s:e] *= duck
    out[s:e] += sfx

    return _limit_norm(out, peak_db=-0.3)

def add_sound_effect_to_audio(audio_path: str, effect_name: str, position="end") -> str:
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1: y = y[:,0]
        y = y.astype(np.float64)
        sfx = generate_sound_effect(effect_name, sr)
        mixed = _mix_with_ducking(y, sfx, sr, pos=position, duck_db=1.5)
        out_path = audio_path.replace(".wav", f"_{effect_name.replace(' ','_')}.wav")
        sf.write(out_path, mixed.astype(np.float32), sr)
        return out_path
    except Exception as e:
        logger.error(f"SFX mix error '{effect_name}': {e}", exc_info=True)
        return audio_path

def postprocess_audio_file(path_in: str) -> str:
    """Safer loudness pipeline to avoid 'quiet/warbly' feel."""
    try:
        y, sr = sf.read(path_in)
        if y.ndim > 1:
            y = y[:,0]
        y = y.astype(np.float64)

        # basic cleanup
        y = _trim_edges(y, sr, 200, 200)
        y = _highpass(y, sr, 80.0)

        # RMS make-up to ~ -20 dBFS (≈ 0.1), capped gain
        rms = np.sqrt(np.mean(y**2)) + 1e-12
        target_rms = 10 ** (-20.0 / 20.0)
        gain = min(10.0, target_rms / rms)
        y *= gain

        # gentle compression + limiter
        y = _compress(y, thr_db=-18.0, ratio=2.5)
        y = _gate(y, -55.0)
        peak = np.max(np.abs(y)) + 1e-12
        if peak > 1.0:
            y /= peak
        y = _limit_norm(y, peak_db=-0.3)

        sf.write(path_in, y.astype(np.float32), sr)
        return path_in
    except Exception as e:
        logger.warning(f"Postprocess error: {e}")
        return path_in

# =============================================================================
# Parsing: VAPI payload, SFX, and prompt controls
# =============================================================================
def _extract_text_from_openai_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text",""))
        return " ".join(parts).strip()
    return ""

def extract_text_from_vapi_payload(v: VapiRequest) -> str:
    try:
        art = (v.message.artifact or {})
        for msg in reversed(art.get("messages", []) or []):
            r = (msg.get("role") or "").lower()
            if r in ("assistant","bot") and msg.get("message"):
                return str(msg["message"]).strip()
        for msg in reversed(art.get("messagesOpenAIFormatted", []) or []):
            r = (msg.get("role") or "").lower()
            if r in ("assistant","bot") and "content" in msg:
                s = _extract_text_from_openai_content(msg["content"])
                if s: return s
        if v.message.turn == 0:
            a = v.message.assistant or {}
            fm = a.get("firstMessage")
            if fm: return str(fm).strip()
        return ""
    except Exception as e:
        logger.error(f"Text extraction error: {e}", exc_info=True)
        return ""

def extract_text_and_effects(text: str) -> Tuple[str, List[str]]:
    """Find inline (laughs) etc. and strip them from text."""
    pattern = r'\(([^)]+)\)'
    raw = re.findall(pattern, text)
    clean = re.sub(pattern, '', text).strip()
    effects = []
    for r in raw:
        k = r.strip().lower()
        if k in EFFECT_ALIASES:
            effects.append(EFFECT_ALIASES[k])
    return clean, (effects[:1] if effects else [])

# ---- Prompt controls from text: {#tts temperature=1.1 top_p=0.9 top_k=40 cfg=3.5 max_new_tokens=800 lang=de voice=female_01 sfx=off sfx_pos=end #}
_TTS_FLOAT = {"temperature", "top_p", "cfg", "speed"}
_TTS_INT = {"top_k", "max_new_tokens"}
_TTS_STR = {"lang", "voice", "sfx", "sfx_pos"}

def parse_tts_controls(text: str):
    pattern = r"\{\#tts\s+([^}]*)\#\}"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return text, {}
    m = matches[-1]
    inner = m.group(1)
    clean_text = (text[:m.start()] + text[m.end():]).strip()
    # key=value (value can be token with letters, digits, punctuation - stop at whitespace)
    pairs = re.findall(r"([a-zA-Z_]+)\s*=\s*([^\s]+)", inner)
    controls: Dict[str, Any] = {}
    for k, v in pairs:
        kl = k.lower()
        if kl in _TTS_FLOAT:
            try: controls[kl] = float(v)
            except Exception: pass
        elif kl in _TTS_INT:
            try: controls[kl] = int(float(v))
            except Exception: pass
        elif kl in _TTS_STR:
            controls[kl] = str(v)
    return clean_text, controls

def clamp(x, lo, hi): return max(lo, min(hi, x))

def split_overrides(controls: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (gen_overrides, meta_overrides) where meta holds lang/voice/sfx settings."""
    gen = {}
    meta = {}
    for k, v in controls.items():
        if k in ("temperature", "top_p", "top_k", "cfg", "max_new_tokens", "speed"):
            gen[k] = v
        else:
            meta[k] = v
    return gen, meta

def apply_gen_overrides(base: dict, overrides: dict):
    out = dict(base)
    if "temperature" in overrides:
        out["temperature"] = clamp(overrides["temperature"], 0.1, 2.0)
    if "top_p" in overrides:
        out["top_p"] = clamp(overrides["top_p"], 0.1, 1.0)
    if "top_k" in overrides:
        out["top_k"] = int(clamp(overrides["top_k"], 1, 200))
    if "cfg" in overrides:
        out["guidance_scale"] = clamp(overrides["cfg"], 0.1, 10.0)
    if "max_new_tokens" in overrides:
        out["max_new_tokens"] = int(clamp(overrides["max_new_tokens"], 1, 4096))
    # "speed" placeholder for pacing (handled in DSP if desired)
    return out

# =============================================================================
# FastAPI app
# =============================================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Startup: loading model...")
    model_manager.load_model()
    yield
    logger.info("Shutdown: unloading model...")
    model_manager.unload_model()

app = FastAPI(
    title="Dia Text-to-Speech API",
    description="Dia TTS with prompt-driven SFX and TTS controls (safer defaults)",
    version="1.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Health
# =============================================================================
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_id": model_manager.model_id,
        "model_loaded": model_manager.is_loaded,
        "supported_effects": list(SOUND_EFFECTS.keys()),
        "sfx_enabled_env": not DISABLE_SFX_ENV,
    }

# =============================================================================
# Core generation used by Vapi
# =============================================================================
def _apply_language_and_voice(text: str, meta: Dict[str, Any]) -> str:
    # Best-effort hints for Dia; non-breaking if ignored by the model.
    lang = (meta.get("lang") or "").strip()
    voice = (meta.get("voice") or "").strip()
    prefix_parts = []
    if lang:
        # Common pattern; adapt as needed for the model
        prefix_parts.append(f"[lang={lang}]")
    if voice:
        prefix_parts.append(f"[voice={voice}]")
    if prefix_parts:
        return " ".join(prefix_parts) + " " + text
    return text

async def _generate_audio_to_file(clean_text: str, gen_kwargs: dict, tmp_path: Path):
    model = model_manager.get_model()
    processor = model_manager.get_processor()
    inputs = processor(text=[clean_text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    def _gen():
        with torch.inference_mode():
            return model.generate(**inputs, **gen_kwargs)

    outputs = await anyio.to_thread.run_sync(_gen)
    decoded = processor.batch_decode(outputs)
    processor.save_audio(decoded, str(tmp_path))

def _decide_sfx_enabled(meta: Dict[str, Any]) -> Tuple[bool, str]:
    # sfx=off|on and sfx_pos=beginning|middle|end
    if DISABLE_SFX_ENV:
        return False, "end"
    sfx_flag = (meta.get("sfx") or "").lower()
    if sfx_flag in ("0", "false", "off", "no"):
        return False, "end"
    sfx_pos = (meta.get("sfx_pos") or "end").lower()
    if sfx_pos not in ("beginning", "middle", "end"):
        sfx_pos = "end"
    return True, sfx_pos

async def process_speech_generation(request: VapiRequest, raw_request: Optional[Request] = None):
    # Optional shared-secret check
    if VAPI_SECRET_EXPECTED and raw_request is not None:
        sent = raw_request.headers.get("X-Vapi-Secret", "")
        if sent != VAPI_SECRET_EXPECTED:
            logger.warning("Unauthorized: X-Vapi-Secret mismatch")
            raise HTTPException(status_code=401, detail="Unauthorized")

    text_in = extract_text_from_vapi_payload(request)
    if not text_in:
        logger.warning("No text to synthesize")
        return JSONResponse({"status": "ignored", "reason": "no text to synthesize"})

    # 1) extract inline effects, then parse {#tts ... #} controls
    text_wo_inline, detected_effects = extract_text_and_effects(text_in)
    clean_text, controls = parse_tts_controls(text_wo_inline)

    if not clean_text:
        logger.warning("No clean text after removing effects/controls")
        return JSONResponse({"status": "ignored", "reason": "no valid text"})

    # 2) split overrides into generation + meta (lang/voice/sfx)
    gen_overrides, meta_overrides = split_overrides(controls)

    # 3) safer-generation defaults (reduce tremolo/robotic)
    base_kwargs = dict(
        max_new_tokens=400,
        guidance_scale=3.5,
        temperature=0.9,
        top_p=0.9,
        top_k=50,
    )
    gen_kwargs = apply_gen_overrides(base_kwargs, gen_overrides)

    # 4) apply language/voice hints to text (non-breaking)
    final_text = _apply_language_and_voice(clean_text, meta_overrides)

    # 5) decide SFX behavior
    sfx_enabled, sfx_pos = _decide_sfx_enabled(meta_overrides)
    # Only apply SFX if explicitly present in the prompt and enabled
    if not detected_effects:
        sfx_enabled = False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=AUDIO_DIR) as tmp:
        tmp_path = Path(tmp.name)

    start = time.time()
    try:
        logger.info(
            f"Generate: text={final_text!r} | effects={detected_effects if sfx_enabled else []} "
            f"| gen_overrides={gen_overrides} | meta={meta_overrides}"
        )
        await _generate_audio_to_file(final_text, gen_kwargs, tmp_path)
        postprocess_audio_file(str(tmp_path))

        final_path = str(tmp_path)
        if sfx_enabled:
            eff = detected_effects[0]
            logger.info(f"Mixing SFX from prompt: {eff} pos={sfx_pos}")
            final_path = add_sound_effect_to_audio(final_path, eff, sfx_pos)
            if final_path != str(tmp_path) and os.path.exists(str(tmp_path)):
                try:
                    os.remove(str(tmp_path))
                except Exception:
                    pass

        pretty = AUDIO_DIR / f"{int(time.time())}_{str(uuid.uuid4())[:8]}.wav"
        if final_path != str(pretty):
            data, sr = sf.read(final_path); sf.write(pretty, data, sr)
            try:
                os.remove(final_path)
            except Exception:
                pass

        t = time.time() - start
        # headers for debugging from Vapi
        hdrs = {
            "X-Generation-Time": f"{t:.2f}",
            "X-File-Size": f"{pretty.stat().st_size}",
            "X-Effects-Applied": ",".join(detected_effects if sfx_enabled else []),
            "X-TTS-Gen-Overrides": ",".join(f"{k}={gen_overrides[k]}" for k in gen_overrides),
            "X-TTS-Meta": ",".join(f"{k}={meta_overrides[k]}" for k in meta_overrides),
        }
        return FileResponse(
            path=str(pretty),
            media_type="audio/wav",
            filename=pretty.name,
            headers=hdrs,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        try:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# =============================================================================
# Routes
# =============================================================================
@app.post("/api/generate")
async def run_inference(vapi_request: VapiRequest, req: Request):
    msg = vapi_request.message
    logger.info(f"VAPI: type={msg.type} status={msg.status} turn={msg.turn}")
    # Vapi sends "started" first → acknowledge (no audio yet)
    if msg.status.lower() == "started":
        return JSONResponse({"status": "acknowledged", "reason": "speech generation started"})
    # When "stopped" → synthesize and return a WAV
    elif msg.status.lower() == "stopped":
        return await process_speech_generation(vapi_request, raw_request=req)
    else:
        return JSONResponse({"status": "ignored", "reason": f"unknown status '{msg.status}'"})

@app.post("/api/generate-direct")
async def run_inference_direct(req: GenerateRequest):
    if not req.text_input or req.text_input.strip() == "":
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    # 1) inline effects + controls
    text_wo_inline, effects = extract_text_and_effects(req.text_input)
    clean_text, controls = parse_tts_controls(text_wo_inline)
    if not clean_text:
        raise HTTPException(status_code=400, detail="No text after removing effects/controls")

    gen_overrides, meta_overrides = split_overrides(controls)

    # defaults come from request body, but we still clamp via apply_gen_overrides
    base_kwargs = dict(
        max_new_tokens=req.max_new_tokens,
        guidance_scale=req.cfg_scale,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.cfg_filter_top_k,
    )
    gen_kwargs = apply_gen_overrides(base_kwargs, gen_overrides)
    final_text = _apply_language_and_voice(clean_text, meta_overrides)

    sfx_enabled, sfx_pos = _decide_sfx_enabled(meta_overrides)
    if not effects:
        sfx_enabled = False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=AUDIO_DIR) as tmp:
        tmp_path = Path(tmp.name)

    start = time.time()
    try:
        logger.info(
            f"(direct) Generate: text={final_text!r} | effects={effects if sfx_enabled else []} "
            f"| gen_overrides={gen_overrides} | meta={meta_overrides}"
        )
        await _generate_audio_to_file(final_text, gen_kwargs, tmp_path)
        postprocess_audio_file(str(tmp_path))

        final_path = str(tmp_path)
        if sfx_enabled:
            eff = effects[0]
            final_path = add_sound_effect_to_audio(final_path, eff, sfx_pos)
            if final_path != str(tmp_path) and os.path.exists(str(tmp_path)):
                try:
                    os.remove(str(tmp_path))
                except Exception:
                    pass

        pretty = AUDIO_DIR / f"{int(time.time())}_{str(uuid.uuid4())[:8]}.wav"
        if final_path != str(pretty):
            data, sr = sf.read(final_path); sf.write(pretty, data, sr)
            try:
                os.remove(final_path)
            except Exception:
                pass

        t = time.time() - start
        return FileResponse(
            path=str(pretty),
            media_type="audio/wav",
            filename=pretty.name,
            headers={
                "X-Generation-Time": f"{t:.2f}",
                "X-File-Size": f"{pretty.stat().st_size}",
                "X-Effects-Applied": ",".join(effects if sfx_enabled else []),
                "X-TTS-Gen-Overrides": ",".join(f"{k}={gen_overrides[k]}" for k in gen_overrides),
                "X-TTS-Meta": ",".join(f"{k}={meta_overrides[k]}" for k in meta_overrides),
            },
        )
    except Exception as e:
        logger.error(f"(direct) Inference error: {e}", exc_info=True)
        try:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# =============================================================================
# Info endpoints
# =============================================================================
@app.get("/api/effects")
async def get_supported_effects():
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
            "sneezes": "Sneezing sound",
        },
    }

@app.get("/api/stats")
async def get_stats():
    try:
        files = list(AUDIO_DIR.glob("*.wav"))
        total = sum(f.stat().st_size for f in files)
        return {
            "total_files": len(files),
            "total_size_bytes": total,
            "total_size_mb": total/(1024*1024),
            "device": DEVICE,
            "model_loaded": model_manager.is_loaded,
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving statistics")
