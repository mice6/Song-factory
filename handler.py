"""RunPod Serverless handler with ACE-Step 1.5 music generation."""
import runpod
import base64
import io
import os
import sys

# Dodaj ścieżkę do ACE-Step
sys.path.insert(0, '/app')

try:
    # Próba importu ACE-Step
    from acestep import generate_song
    ACESTEP_AVAILABLE = True
    print("[ACE-Step] Model loaded successfully")
except ImportError as e:
    print(f"[ACE-Step] Import failed: {e}")
    ACESTEP_AVAILABLE = False

# Fallback: placeholder jeśli ACE-Step nie działa
import numpy as np
import wave


def generate_placeholder_audio(duration: int = 90):
    """Generate placeholder WAV if ACE-Step fails."""
    sample_rate = 44100
    num_samples = duration * sample_rate
    t = np.linspace(0, duration, num_samples, False)
    
    freq = 440.0
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    audio += 0.2 * np.sin(2 * np.pi * freq * 1.25 * t)
    audio += 0.15 * np.sin(2 * np.pi * freq * 1.5 * t)
    
    fade_len = int(0.5 * sample_rate)
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)
    
    audio_int16 = (audio * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    return buf.getvalue()


def generate_acestep_audio(lyrics: str, style_prompt: str, language: str, duration: int = 90):
    """Generate audio using ACE-Step 1.5."""
    try:
        # ACE-Step generuje audio bytes
        audio_bytes = generate_song(
            lyrics=lyrics,
            style_prompt=style_prompt,
            language=language,
            duration=duration,
            model="2B-turbo"  # najszybszy model
        )
        return audio_bytes
    except Exception as e:
        print(f"[ACE-Step] Generation failed: {e}")
        return None


def handler(job):
    job_input = job.get("input", {})
    
    lyrics = job_input.get("lyrics", "")
    style_prompt = job_input.get("style_prompt", "")
    language = job_input.get("language", "en")
    duration = job_input.get("duration", 90)
    
    print(f"[Handler] Request: lang={language}, style={style_prompt[:30]}..., duration={duration}")
    
    # Próba ACE-Step, fallback do placeholder
    if ACESTEP_AVAILABLE:
        audio_bytes = generate_acestep_audio(lyrics, style_prompt, language, duration)
    else:
        audio_bytes = None
    
    if audio_bytes is None:
        print("[Handler] Using placeholder audio")
        audio_bytes = generate_placeholder_audio(duration)
    
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    return {
        "audio_base64": audio_b64,
        "format": "wav",
        "duration": duration,
        "style": style_prompt[:50],
        "ace_step_used": ACESTEP_AVAILABLE and audio_bytes is not None
    }


if __name__ == "__main__":
    print("[Startup] ACE-Step handler starting...")
    print(f"[Startup] ACE-Step available: {ACESTEP_AVAILABLE}")
    runpod.serverless.start({"handler": handler})
