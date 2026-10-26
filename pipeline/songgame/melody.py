"""Melody-first piano arrangement.

What makes a song recognizable is its sung melody, so:
  1. Split the recording into stems (vocals / bass / drums / other) with Demucs.
  2. Track the singer's pitch (pYIN) and turn it into piano notes, snapped to the
     song's key -> right hand, loud.
  3. Detect the chords from the bass + instruments -> left hand: bass note + chord,
     struck on the beat (tempo from the drums), soft.
"""
import numpy as np

from . import audio as A
from .piano import Note

SR = 22050

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)       # natural minor + leading tone


# ---------------------------------------------------------------------------
# Melody: pitch track -> notes
# ---------------------------------------------------------------------------
def _mode_filter(x, w=5):
    """Replace each value by the most common value around it (NaN-aware)."""
    out = x.copy()
    h = w // 2
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        win = x[max(0, i - h): i + h + 1]
        win = win[~np.isnan(win)]
        vals, counts = np.unique(win, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def f0_to_notes(f0, voiced, hop_s, loudness=None, min_dur=0.09, max_gap=0.06):
    """Frame-wise pitch (Hz) -> list of melody Notes."""
    f0 = np.asarray(f0, float)
    voiced = np.asarray(voiced, bool) & (f0 > 0) & np.isfinite(f0)
    midi = np.full(len(f0), np.nan)
    midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440.0)
    midi = _mode_filter(np.round(midi), 5)

    notes, i, n = [], 0, len(midi)
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and midi[j + 1] == midi[i]:
            j += 1
        start, end = i * hop_s, (j + 1) * hop_s
        vel = 96
        if loudness is not None:
            seg = loudness[i:j + 1]
            vel = int(np.clip(78 + 40 * (seg.mean() / (loudness.max() + 1e-9)), 78, 118))
        notes.append(Note(start, end, int(midi[i]), vel, "melody"))
        i = j + 1

    merged = []
    for nt in notes:
        if merged and nt.pitch == merged[-1].pitch and nt.start - merged[-1].end <= max_gap:
            merged[-1].end = nt.end
        else:
            merged.append(nt)
    return [m for m in merged if m.end - m.start >= min_dur]


def estimate_key(chroma_frames):
    """(tonic pitch class, 'major'|'minor') via Krumhansl–Schmuckler."""
    prof = np.asarray(chroma_frames).sum(axis=1)
    best = (-2, 0, "major")
    for tonic in range(12):
        for mode, p in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(prof, np.roll(p, tonic))[0, 1]
            if r > best[0]:
                best = (r, tonic, mode)
    return best[1], best[2]


def snap_to_key(notes, tonic, mode, max_len=0.2):
    """Fix short off-key notes (pitch-tracker slips on slides and consonants).
    Long notes are trusted as sung, so a wrong key guess can't bend the tune."""
    scale = {(tonic + s) % 12 for s in (MAJOR_SCALE if mode == "major" else MINOR_SCALE)}
    for nt in notes:
        if nt.end - nt.start > max_len:
            continue
        if nt.pitch % 12 not in scale:
            down, up = nt.pitch - 1, nt.pitch + 1
            nt.pitch = down if down % 12 in scale else up
    return notes


def place_octave(notes, target=72):
    """Move the whole melody by octaves so it sits around C5 (clear on piano)."""
    if not notes:
        return notes
    med = np.median([n.pitch for n in notes])
    shift = 12 * int(round((target - med) / 12))
    for n in notes:
        n.pitch += shift
    return notes


# ---------------------------------------------------------------------------
# Accompaniment: tempo + chords
# ---------------------------------------------------------------------------
def estimate_beat(drums, sr=SR, hop=256):
    """(period seconds, first beat seconds) from a drum stem; falls back to 0.5 s."""
    n_fft = 1024
    if len(drums) < sr * 4:
        return 0.5, 0.0
    frames = 1 + (len(drums) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    mag = np.abs(np.fft.rfft(drums[idx] * np.hanning(n_fft), axis=1))
    flux = np.maximum(0, np.diff(np.log1p(10 * mag), axis=0)).sum(axis=1)
    flux -= flux.mean()
    if flux.std() < 1e-9:
        return 0.5, 0.0
    fps = sr / hop
    times = ((np.arange(len(flux)) + 1) * hop + n_fft / 2) / sr    # centre of frame i+1

    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    lo, hi = int(0.33 * fps), int(1.0 * fps)                      # 60–180 BPM
    lags = np.arange(lo, hi)
    # Prefer typical pop tempos (~100–130 BPM) when peaks are close.
    weight = np.exp(-0.5 * (np.log2(lags / (0.52 * fps)) / 0.9) ** 2)
    lag = lags[np.argmax(ac[lo:hi] * weight)]

    # Refine period and phase together on a fine grid (comb over the whole clip).
    pos = np.maximum(flux, 0)
    best = (-1.0, lag / fps, 0.0)
    for period in np.linspace(lag / fps * 0.97, lag / fps * 1.03, 61):
        for phase in np.arange(0, period, 0.005):
            beats = np.arange(phase, times[-1], period)
            k = np.clip(np.round((beats * sr - n_fft / 2) / hop - 1).astype(int), 0, len(pos) - 1)
            score = pos[k].sum() + 0.5 * (pos[np.clip(k - 1, 0, None)].sum() + pos[np.clip(k + 1, None, len(pos) - 1)].sum())
            if score > best[0]:
                best = (score, period, phase)
    return float(best[1]), float(best[2])


def _chord_templates():
    names, temps = [], []
    for r in range(12):
        for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            t[[(r + i) % 12 for i in iv]] = 1
            t[r] += 0.3
            names.append((r, q))
            temps.append(t / np.linalg.norm(t))
    return names, np.array(temps)


def detect_chords(chroma_frames, hop_s, switch_penalty=0.12, min_dur=0.45):
    """Chroma (12 x frames) -> list of (start, end, root, quality) via Viterbi."""
    names, temps = _chord_templates()
    c = np.asarray(chroma_frames)
    sims = temps @ c                                     # chords x frames
    silent = np.linalg.norm(c, axis=0) == 0
    n_ch, n_fr = sims.shape
    if n_fr == 0:
        return []
    score = sims[:, 0].copy()
    back = np.zeros((n_ch, n_fr), int)
    for f in range(1, n_fr):
        stay = score
        best_prev = np.argmax(score)
        move = score[best_prev] - switch_penalty
        choose_move = move > stay
        back[:, f] = np.where(choose_move, best_prev, np.arange(n_ch))
        score = np.where(choose_move, move, stay) + sims[:, f]
    path = [int(np.argmax(score))]
    for f in range(n_fr - 1, 0, -1):
        path.append(back[path[-1], f])
    path = path[::-1]

    segs = []
    for f, k in enumerate(path):
        k = -1 if silent[f] else k
        if segs and segs[-1][2] == k:
            segs[-1][1] = f + 1
        else:
            segs.append([f, f + 1, k])
    # absorb very short chords into their neighbour
    out = []
    for s in segs:
        if out and ((s[1] - s[0]) * hop_s < min_dur or s[2] == out[-1][2]):
            out[-1][1] = s[1]
        else:
            out.append(s)
    return [(a * hop_s, b * hop_s, names[k][0], names[k][1]) for a, b, k in out if k >= 0]


def accompaniment(chords, period, phase, end_time, melody_low=64, beats_per_chord=2):
    """Sparse left hand: bass note each bar, chord every `beats_per_chord` beats,
    kept well below the melody so the tune stays clear."""
    notes = []
    top = min(64, melody_low - 3)
    step = period * beats_per_chord
    for (start, end, root, q) in chords:
        tones = [(root + i) % 12 for i in ((0, 4, 7) if q == "" else (0, 3, 7))]
        chord = sorted(top - ((top - pc) % 12) for pc in tones)            # close voicing
        bass = 40 + ((root - 4) % 12)                                       # E2..D#3
        k = int(np.ceil((start - phase) / step))
        hits = [phase + i * step for i in range(k, int((end - phase) / step) + 1)]
        hits = [h for h in hits if start - 1e-6 <= h < end]
        if not hits or hits[0] - start > period * 0.75:                   # chord change between hits
            hits = [start] + hits
        for i, h in enumerate(hits):
            dur = min(step * 0.97, end - h, end_time - h)
            if dur <= 0.08:
                continue
            if i % 2 == 0:
                notes.append(Note(h, h + min(step * 1.95, end - h), bass, 58, "bass"))
            for p in chord:
                notes.append(Note(h + 0.012, h + dur, p, 44, "chord"))
    return notes


def tidy_melody(notes, min_dur=0.12, gap=0.03):
    """One clean line: no overlaps, no tiny blips, repeated pitches joined."""
    notes = sorted(notes, key=lambda n: n.start)
    out = []
    for n in notes:
        if out and n.pitch == out[-1].pitch and n.start - out[-1].end < 0.12:
            out[-1].end = max(out[-1].end, n.end)
            continue
        out.append(n)
    kept = []
    for n in (n for n in out if n.end - n.start >= min_dur):   # drop blips, then re-join
        if kept and n.pitch == kept[-1].pitch and n.start - kept[-1].end < 0.2:
            kept[-1].end = max(kept[-1].end, n.end)
        else:
            kept.append(n)
    out = kept
    for a, b in zip(out, out[1:]):
        a.end = min(a.end, b.start - gap) if b.start - gap > a.start + 0.05 else a.end
    return out


def find_chorus(mix, vocals, sr=SR, period=0.5, phase=0.0, min_len=14.0):
    """Start time (s) of the most chorus-like part: loud, full, sung, and
    entered with a lift in energy. Snapped to a bar line."""
    hop = int(0.1 * sr)
    def rms(x):
        n = len(x) // hop
        return np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    e, v = rms(mix), rms(vocals)
    e, v = e / (e.max() + 1e-12), v / (v.max() + 1e-12)
    total = len(e) * 0.1
    bar = period * 4
    best, best_t = -1.0, 0.0
    t = phase % bar
    win, look = int(8 / 0.1), int(2 / 0.1)
    while t <= total - min_len + 1e-6:
        i = int(round(t / 0.1))
        body = e[i:i + win].mean() + 0.6 * (v[i:i + win] > 0.15).mean()
        before = e[max(0, i - look):i].mean() if i > 0 else e[i:i + look].mean()
        lift = max(0.0, e[i:i + look].mean() - before)
        score = body + 0.8 * lift
        if score > best:
            best, best_t = score, t
        t += bar / 2                                          # half-bar candidates
    return round(best_t, 2)


def cut_and_stretch(notes, start, stretch, length):
    """Keep notes from `start` on, move them to t=0 and slow down by `stretch`."""
    out = []
    for n in notes:
        if n.end <= start:
            continue
        s0 = max(n.start, start)
        if s0 - n.start > 0.15 and n.role == "melody":       # half-cut note: skip
            continue
        a, b = (s0 - start) * stretch, (n.end - start) * stretch
        if a >= length:
            continue
        out.append(Note(a, min(b, length), n.pitch, n.velocity, n.role))
    return out


def stretch_for(tempo, target=100.0, max_stretch=1.35):
    """How much to slow a song down: fast songs get calmer, slow ones untouched."""
    return round(float(np.clip(tempo / target, 1.0, max_stretch)), 3)


# ---------------------------------------------------------------------------
# Full pipeline (needs torch + demucs + librosa; run in the cloud workflow)
# ---------------------------------------------------------------------------
class MelodyArranger:
    def __init__(self, device="cpu", target_bpm=100, clip_seconds=30):
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        self.model = get_model("htdemucs")
        self.model.eval()
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.target_bpm = target_bpm
        self.clip_seconds = clip_seconds

    def separate(self, mono, sr):
        from demucs.apply import apply_model
        import librosa

        msr = self.model.samplerate
        if sr != msr:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=msr)
        wav = self.torch.tensor(np.stack([mono, mono]), dtype=self.torch.float32)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with self.torch.no_grad():
            out = apply_model(self.model, ((wav - mean) / std)[None], device=self.device,
                              progress=False)[0]
        out = out * std + mean
        stems = {}
        for name, s in zip(self.model.sources, out):
            stems[name] = librosa.resample(s.mean(0).numpy(), orig_sr=msr, target_sr=SR)
        return stems

    def __call__(self, audio, sr, style="melody"):
        import librosa

        stems = self.separate(audio, sr)
        vocals = stems["vocals"]
        hop = 256
        f0, voiced, _ = librosa.pyin(vocals, fmin=70, fmax=1000, sr=SR,
                                     frame_length=2048, hop_length=hop)
        rms = librosa.feature.rms(y=vocals, frame_length=2048, hop_length=hop)[0]
        rms = np.pad(rms, (0, max(0, len(f0) - len(rms))))[: len(f0)]
        melody = f0_to_notes(np.nan_to_num(f0), voiced, hop / SR, rms)

        harmony = stems["other"] + stems["bass"]
        chroma = A.chroma(harmony, SR, n_fft=4096, hop=1024)
        tonic, mode = estimate_key(chroma)
        melody = tidy_melody(place_octave(snap_to_key(melody, tonic, mode)))
        chords = detect_chords(chroma, 1024 / SR)
        period, phase = estimate_beat(stems["drums"])
        end = len(vocals) / SR
        low = min((n.pitch for n in melody), default=64)
        notes = melody + accompaniment(chords, period, phase, end, low)

        mix = sum(stems.values())
        start = find_chorus(mix, vocals, SR, period, phase)
        stretch = stretch_for(60 / period, self.target_bpm)
        notes = cut_and_stretch(notes, start, stretch, self.clip_seconds)
        mel = [n for n in notes if n.role == "melody"]
        self.last_info = {"key": f"{tonic}{'' if mode == 'major' else 'm'}",
                          "tempo": round(60 / period), "stretch": stretch,
                          "source_offset": start, "melody_notes": len(mel),
                          "voiced": round(float(np.mean(voiced)), 3)}
        return notes


# ---------------------------------------------------------------------------
# Instrumental: the real recording with the vocals removed
# ---------------------------------------------------------------------------
class InstrumentalMaker:
    """Removes the singing with Demucs and picks the chorus. Sounds like the real
    song (because it is), minus the words that would give the answer away."""

    def __init__(self, device="cpu", model="htdemucs_ft"):
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        self.model = get_model(model)
        self.model.eval()
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"

    def __call__(self, mp3_path, wav_out):
        from demucs.apply import apply_model
        from scipy.signal import resample_poly

        sr = self.model.samplerate
        wav = self.torch.tensor(A.decode_stereo(mp3_path, sr))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with self.torch.no_grad():
            out = apply_model(self.model, ((wav - mean) / std)[None], device=self.device,
                              progress=False)[0]
        stems = {n: (s * std + mean).numpy() for n, s in zip(self.model.sources, out)}
        inst = sum(v for k, v in stems.items() if k != "vocals")
        mono = lambda x: resample_poly(x.mean(0), 1, 2) if sr == 44100 else x.mean(0)
        period, phase = estimate_beat(mono(stems["drums"]))
        start = find_chorus(mono(inst), mono(stems["vocals"]), SR, period, phase)
        peak = np.abs(inst).max() or 1.0
        A.write_wav(wav_out, inst / peak * 0.95, sr)
        voc = np.sqrt((stems["vocals"] ** 2).mean()) / (np.sqrt((inst ** 2).mean()) + 1e-9)
        self.last_info = {"source_offset": start, "tempo": round(60 / period),
                          "vocal_level": round(float(voc), 3)}
        return start
