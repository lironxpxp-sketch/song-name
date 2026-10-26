"""Make a piano version of a song clip, render it to audio, and score it.

Generation uses Pop2Piano (Hugging Face model "sweetcocoa/pop2piano"), which turns
pop audio into piano MIDI. It has 21 "composer" styles; we try a few per song and
keep the one whose harmony best matches the original recording.
"""
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from . import audio as A

COMPOSERS = [f"composer{i}" for i in range(1, 22)]


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int = 80
    role: str = ""          # "melody", "chord", "bass" (melody method only)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class Pop2PianoGenerator:
    """Wraps the pretrained model. Loads lazily; the first run downloads ~1 GB."""

    model_id = "sweetcocoa/pop2piano"

    def __init__(self, device="auto"):
        import torch
        from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = Pop2PianoProcessor.from_pretrained(self.model_id)
        self.model = Pop2PianoForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(device).eval()

    def __call__(self, audio: np.ndarray, sr: int, composer: str) -> list[Note]:
        import torch

        inputs = self.processor(audio=audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                input_features=inputs["input_features"].to(self.device),
                composer=composer,
            )
        midi = self.processor.batch_decode(
            token_ids=out.cpu(), feature_extractor_output=inputs
        )["pretty_midi_objects"][0]
        return [Note(n.start, n.end, n.pitch, n.velocity)
                for inst in midi.instruments for n in inst.notes]


# ---------------------------------------------------------------------------
# MIDI file writing (no extra dependency)
# ---------------------------------------------------------------------------
def write_midi(notes: list[Note], path, tempo_bpm=120, ppq=480):
    ticks_per_sec = ppq * tempo_bpm / 60.0
    events = []
    for n in notes:
        on, off = int(round(n.start * ticks_per_sec)), int(round(n.end * ticks_per_sec))
        events.append((on, 1, n.pitch, max(1, min(127, n.velocity))))
        events.append((max(off, on + 1), 0, n.pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))          # note-offs before note-ons

    def vlq(v):
        out = [v & 0x7F]
        v >>= 7
        while v:
            out.insert(0, (v & 0x7F) | 0x80)
            v >>= 7
        return bytes(out)

    body = bytearray()
    body += vlq(0) + b"\xFF\x51\x03" + int(60_000_000 / tempo_bpm).to_bytes(3, "big")
    body += vlq(0) + b"\xC0\x00"                      # program 0: acoustic grand
    last = 0
    for tick, is_on, pitch, vel in events:
        body += vlq(tick - last) + bytes([0x90 if is_on else 0x80, pitch, vel])
        last = tick
    body += vlq(0) + b"\xFF\x2F\x00"
    data = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq)
    data += b"MTrk" + struct.pack(">I", len(body)) + bytes(body)
    Path(path).write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(notes: list[Note], wav_path, soundfont=None, sr=44100) -> Path:
    """Render notes to a WAV. Uses FluidSynth + a piano SoundFont when given
    (much better sound); otherwise a built-in synthesized piano."""
    wav_path = Path(wav_path)
    if soundfont and Path(soundfont).exists() and shutil.which("fluidsynth"):
        mid = wav_path.with_suffix(".mid")
        write_midi(notes, mid)
        subprocess.run(["fluidsynth", "-ni", "-g", "0.6", "-r", str(sr),
                        "-F", str(wav_path), str(soundfont), str(mid)],
                       check=True, capture_output=True)
        return wav_path
    A.write_wav(wav_path, synth_piano(notes, sr), sr)
    return wav_path


def synth_piano(notes: list[Note], sr=44100) -> np.ndarray:
    """Simple additive piano: decaying slightly-inharmonic partials + room reverb."""
    if not notes:
        return np.zeros(sr, dtype=np.float32)
    total = max(n.end for n in notes) + 1.5
    out = np.zeros(int(total * sr), dtype=np.float64)
    for n in notes:
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        held = max(n.end - n.start, 0.05)
        length = held + 0.35
        t = np.arange(int(length * sr)) / sr
        decay = 1.2 + 2.5 * (n.pitch - 21) / 87            # higher keys die faster
        tone = np.zeros_like(t)
        for k in range(1, 9):
            fk = k * f0 * np.sqrt(1 + 0.0004 * k * k)       # string stiffness
            if fk > sr / 2.2:
                break
            tone += np.sin(2 * np.pi * fk * t) * np.exp(-decay * k ** 0.6 * t) / k ** 1.4
        env = np.minimum(1.0, t / 0.004)                    # hammer attack
        rel = t > held
        env[rel] *= np.exp(-(t[rel] - held) / 0.08)         # damper release
        i = int(n.start * sr)
        seg = tone * env * (n.velocity / 127.0) ** 1.5
        out[i:i + len(seg)] += seg[: len(out) - i]
    ir_t = np.arange(int(0.9 * sr)) / sr
    rng = np.random.default_rng(0)
    ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t / 0.25) * 0.02
    ir[0] = 1.0
    out = fftconvolve(out, ir)[: len(out)]
    peak = np.abs(out).max() or 1.0
    return (0.9 * out / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(original: np.ndarray, piano: np.ndarray, notes: list[Note], sr=22050) -> float:
    """0..1, higher = piano version sounds more like the song.

    Mostly harmonic similarity (pitch-class content over time, allowing ~0.5 s of
    timing slack), with penalties for near-empty or wildly over-busy output.
    """
    n = min(len(original), len(piano))
    if n < sr or not notes:
        return 0.0
    co, cp = A.chroma(original[:n], sr), A.chroma(piano[:n], sr)
    k = np.ones(5) / 5                                    # smooth ~0.5 s
    co = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, co)
    cp = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, cp)
    num = (co * cp).sum(axis=0)
    den = np.linalg.norm(co, axis=0) * np.linalg.norm(cp, axis=0)
    active = den > 0
    if active.sum() < 3:
        return 0.0
    sim = float((num[active] / den[active]).mean())
    coverage = active.mean()                              # silence in the piano = bad
    per_sec = len(notes) / (n / sr)
    density = 1.0 if 0.5 <= per_sec <= 25 else 0.7
    return round(sim * (0.5 + 0.5 * coverage) * density, 4)


def first_onset(notes: list[Note]) -> float:
    return min((n.start for n in notes), default=0.0)
