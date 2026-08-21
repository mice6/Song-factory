"""RunPod Serverless handler — placeholder music generation."""
import runpod
import base64
import io
import numpy as np
import wave


def generate_placeholder_audio(lyrics: str, style_prompt: str, language: str, duration: int = 90):
    """Generate a placeholder chord progression as WAV."""
    sample_rate = 44100
    num_samples = duration * sample_rate
    t = np.linspace(0, duration, num_samples, False)
    
    # Different base frequency per style
    style_freqs = {
        "synth-pop": 440.0,
        "acoustic": 329.63,
        "cinematic": 261.63,
        "lofi": 349.23,
        "rock": 220.0
    }
    
    freq = 440.0
    for style_id, f in style_freqs.items():
        if style_id in style_prompt.lower():
            freq = f
            break
    
    # Add some harmonics for richer sound
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    audio += 0.2 * np.sin(2 * np.pi * freq * 1.25 * t)
    audio += 0.15 * np.sin(2 * np.pi * freq * 1.5 * t)
    audio += 0.1 * np.sin(2 * np.pi * freq * 2.0 * t)
    
    # Simple envelope
    fade_len = int(0.5 * sample_rate)
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    return buf.getvalue()


def handler(job):
    job_input = job.get("input", {})
    lyrics = job_input.get("lyrics", "")
    style_prompt = job_input.get("style_prompt", "")
    language = job_input.get("language", "en")
    duration = job_input.get("duration", 90)
    
    audio_bytes = generate_placeholder_audio(lyrics, style_prompt, language, duration)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    return {
        "audio_base64": audio_b64,
        "format": "wav",
        "duration": duration,
        "style": style_prompt[:50]
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
