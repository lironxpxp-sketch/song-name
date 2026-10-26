"""Audio helpers built on ffmpeg + numpy: decode, write, encode, chroma."""
import subprocess
import wave
from pathlib import Path

import numpy as np


def decode(path, sr=44100) -> np.ndarray:
    """Any audio file -> mono float32 at `sr`."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
           "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_wav(path, audio: np.ndarray, sr=44100):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def encode_m4a(wav_path, out_path, start=0.0, duration=30.0, fade=0.6):
    """Cut [start, start+duration], fade out, loudness-normalize, AAC 128k."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fade_start = max(0.0, duration - fade)
    af = (f"afade=t=out:st={fade_start:.2f}:d={fade:.2f},"
          "loudnorm=I=-16:TP=-1.5:LRA=11")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
           "-i", str(wav_path), "-af", af, "-ar", "44100", "-ac", "1",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def chroma(audio: np.ndarray, sr=22050, n_fft=4096, hop=2048) -> np.ndarray:
    """12 x frames pitch-class energy, each frame L2-normalized (0 if silent)."""
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frames = 1 + (len(audio) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(audio[idx] * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= 55) & (freqs <= 4200)
    midi = 69 + 12 * np.log2(freqs[band] / 440.0)
    pc = np.round(midi).astype(int) % 12
    out = np.zeros((12, frames))
    for k in range(12):
        out[k] = spec[:, band][:, pc == k].sum(axis=1)
    out = np.sqrt(out)                                 # compress dynamics
    energy = np.linalg.norm(out, axis=0)
    loud = energy > 1e-3 * (energy.max() + 1e-12)
    out[:, loud] /= energy[loud]
    out[:, ~loud] = 0
    return out
