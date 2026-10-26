"""Offline tests for the melody-first arranger (no Demucs/librosa needed).

Run:  python tests/test_melody.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from songgame import audio as A, melody as M                     # noqa: E402
from songgame.build import clip_start                             # noqa: E402
from songgame.piano import Note, synth_piano                      # noqa: E402

SR = 22050
PROG = [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]]   # C G Am F
HZ = lambda m: 440 * 2 ** ((m - 69) / 12)


def chords(prog, secs=24, per=2.0):
    out, t = [], 0.0
    while t < secs - 1e-9:
        for p in prog[int(round(t / per)) % len(prog)]:
            out.append(Note(t, t + per * 0.95, p, 80))
        t += per
    return out


def test_f0_to_notes():
    hop = 256 / SR
    tune = [(0.5, 1.0, 64), (1.0, 1.5, 62), (1.5, 2.5, 60), (3.0, 3.4, 67)]
    t = np.arange(int(4 / hop)) * hop
    f0 = np.zeros_like(t)
    rng = np.random.default_rng(0)
    for a, b, m in tune:
        sel = (t >= a) & (t < b)
        vib = 0.3 * np.sin(2 * np.pi * 5.5 * t[sel])                 # vibrato ±0.3 semitone
        f0[sel] = HZ(m + vib + rng.normal(0, 0.05, sel.sum()))
    f0[int(2.0 / hop)] = HZ(71)                                       # one-frame glitch
    f0[int(1.2 / hop)] = 0                                            # dropout inside a note
    notes = M.f0_to_notes(f0, f0 > 0, hop)
    got = [(round(n.start, 1), n.pitch) for n in notes]
    print("  melody:", got)
    assert [n.pitch for n in notes] == [64, 62, 60, 67], got
    assert abs(notes[0].start - 0.5) < 0.05 and abs(notes[2].end - 2.5) < 0.05
    assert all(n.role == "melody" for n in notes)


def test_key_and_snap():
    c = A.chroma(synth_piano(chords(PROG), SR), SR, hop=1024)
    assert M.estimate_key(c) == (0, "major"), M.estimate_key(c)
    am = A.chroma(synth_piano(chords([[57, 60, 64], [52, 56, 59], [57, 60, 64], [50, 53, 57]]), SR), SR, hop=1024)
    assert M.estimate_key(am) == (9, "minor"), M.estimate_key(am)
    notes = [Note(0, 1, 61), Note(1, 2, 66), Note(2, 3, 60)]
    assert [n.pitch for n in M.snap_to_key(notes, 0, "major")] == [60, 65, 60]
    mel = M.place_octave([Note(0, 1, 52), Note(1, 2, 55)])
    assert 66 <= np.median([n.pitch for n in mel]) <= 78


def test_chords():
    hop = 1024
    c = A.chroma(synth_piano(chords(PROG, 16), SR), SR, hop=hop)
    segs = M.detect_chords(c, hop / SR)
    names = [("C C# D D# E F F# G G# A A# B".split()[r] + q) for _, _, r, q in segs]
    print("  chords:", names, [round(a, 1) for a, *_ in segs])
    assert names[:8] == ["C", "G", "Am", "F", "C", "G", "Am", "F"], names
    assert all(abs(a - 2 * i) < 0.25 for i, (a, *_) in enumerate(segs[:8]))


def test_beat():
    for bpm in (96, 120, 128):
        period = 60 / bpm
        x = np.zeros(int(20 * SR))
        rng = np.random.default_rng(1)
        for k in range(int(20 / period)):
            i = int((0.2 + k * period) * SR)
            n = int(0.03 * SR)
            x[i:i + n] += rng.standard_normal(n) * np.exp(-np.arange(n) / 200)
        p, ph = M.estimate_beat(x)
        print(f"  {bpm} bpm -> {60 / p:.1f} bpm, phase {ph:.2f}")
        assert abs(60 / p - bpm) < 3, (bpm, 60 / p)
        assert min(abs(((ph - 0.2) % period)), period - ((ph - 0.2) % period)) < 0.05


def test_accompaniment_and_start():
    segs = [(0.0, 2.0, 0, ""), (2.0, 4.0, 9, "m")]
    acc = M.accompaniment(segs, 0.5, 0.0, 4.0, melody_low=67)
    assert acc and all(n.pitch < 67 for n in acc)
    assert {n.role for n in acc} == {"bass", "chord"}
    first = [n.pitch % 12 for n in acc if n.role == "chord" and n.start < 0.1]
    assert sorted(first) == [0, 4, 7]
    am = sorted({n.pitch % 12 for n in acc if n.role == "chord" and 2 <= n.start < 2.1})
    assert am == [0, 4, 9]
    # clip starts at the singing, not the intro
    notes = acc + [Note(6.0, 6.5, 72, 100, "melody")]
    assert clip_start(notes) == 5.6
    late = acc + [Note(20.0, 21, 72, 100, "melody")]           # singing too late -> intro
    assert clip_start(late) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
