"""Offline tests: no network, no model. Fakes Deezer and Pop2Piano.

Run:  python -m pytest tests   (or: python tests/test_offline.py)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from songgame import audio as A, build as B, names                       # noqa: E402
from songgame.deezer import Deezer                                         # noqa: E402
from songgame.piano import Note, render, score, synth_piano, write_midi    # noqa: E402

# I-V-vi-IV in C, two seconds per chord
PROG = [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]]
WRONG = [[61, 65, 68], [63, 66, 70], [56, 59, 63], [66, 70, 73]]


def chords(prog, secs=24, per=2.0, vel=90):
    out, t = [], 0.0
    while t < secs:
        for p in prog[int(t / per) % len(prog)]:
            out.append(Note(t, t + per * 0.9, p, vel))
        t += per
    return out


# --- names ------------------------------------------------------------------
def test_names():
    assert names.strip_version("Bohemian Rhapsody - Remastered 2011") == "Bohemian Rhapsody"
    assert names.strip_version("Stay (feat. Justin Bieber)") == "Stay"
    assert names.strip_version("Don't Stop Me Now (Live)") == "Don't Stop Me Now"
    assert names.strip_version("Hey Jude (Remastered)") == "Hey Jude"
    assert names.strip_version("Sweet Child O' Mine") == "Sweet Child O' Mine"
    assert names.normalize("Don’t Stop!") == "dont stop"
    assert names.normalize("שִׁיר לַשָּׁלוֹם") == "שיר לשלום"
    assert names.normalize("תל-אביב") == "תל אביב"
    assert names.answers_for("Stay (feat. Justin Bieber)", "Stay") == ["Stay"]
    a = names.answers_for("Song A / Song B")
    assert a[0] == "Song A / Song B" and "Song A" in a
    assert (names.dedup_key("Hey Jude - Remastered 2015", "The Beatles")
            == names.dedup_key("Hey Jude", "The Beatles"))
    assert (names.dedup_key("Stay", "The Kid LAROI, Justin Bieber")
            == names.dedup_key("Stay (feat. X)", "The Kid LAROI"))


# --- midi + render + score ------------------------------------------------------
def test_midi_file_is_valid():
    with tempfile.TemporaryDirectory() as d:
        p = write_midi(chords(PROG, 4), Path(d) / "x.mid")
        data = p.read_bytes()
        assert data[:4] == b"MThd" and b"MTrk" in data and data.endswith(b"\xFF\x2F\x00")


def test_score_prefers_correct_harmony():
    sr = 22050
    orig = synth_piano(chords(PROG), sr)
    orig = orig + 0.05 * np.random.default_rng(1).standard_normal(len(orig))  # "drums"
    good_n, bad_n = chords(PROG, vel=70), chords(WRONG)
    good = score(orig, synth_piano(good_n, sr), good_n, sr)
    bad = score(orig, synth_piano(bad_n, sr), bad_n, sr)
    empty = score(orig, np.zeros_like(orig), [], sr)
    print(f"  score good={good:.3f} wrong-chords={bad:.3f} empty={empty:.3f}")
    assert good > 0.7 and bad < good - 0.3 and empty == 0.0


# --- fakes ---------------------------------------------------------------------
class FakeResp:
    def __init__(self, data=None, body=b""):
        self._d, self._b = data, body
    def raise_for_status(self): pass
    def json(self): return self._d
    def iter_content(self, n): yield self._b
    def __enter__(self): return self
    def __exit__(self, *a): pass


def make_tracks(prefix, n, start_id):
    return [{"id": start_id + i, "readable": True,
             "title": f"{prefix} Song {i}" + (" - Remastered 2011" if i == 0 else ""),
             "title_short": f"{prefix} Song {i}",
             "artist": {"name": f"Artist {i}"}, "album": {"cover_medium": "c.jpg"},
             "link": f"https://deezer.com/track/{start_id + i}"} for i in range(n)]


class FakeSession:
    def __init__(self, mp3_bytes):
        self.mp3 = mp3_bytes
        self.calls = []
        shared = make_tracks("Hit", 1, 900)        # same song in two playlists
        self.data = {
            "/chart/0/tracks": make_tracks("Chart", 3, 100) + shared,
            "/search/playlist": [{"id": 7, "title": "Some fan list", "nb_tracks": 9,
                                  "user": {"name": "randomfan"}},
                                 {"id": 8, "title": "Top Israel", "nb_tracks": 3,
                                  "user": {"name": "Deezer Charts"}}],
            "/playlist/8/tracks": make_tracks("Isr", 3, 200) + shared,
        }
        self.quota_once = True

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls.append(url)
        if url.startswith("https://cdn/"):
            return FakeResp(body=self.mp3)
        path = url.replace("https://api.deezer.com", "")
        if path == "/search/playlist" and self.quota_once:   # simulate quota hit
            self.quota_once = False
            return FakeResp({"error": {"code": 4, "message": "Quota limit exceeded"}})
        if path == "/search":
            q = (params or {}).get("q", "")
            if "Nobody" in q:
                return FakeResp({"data": []})
            return FakeResp({"data": [{**make_tracks("Searched", 1, 700)[0], "preview": "x"}]})
        if path.startswith("/track/"):
            return FakeResp({"id": int(path.split("/")[-1]), "preview": "https://cdn/x.mp3"})
        return FakeResp({"data": self.data[path]})


def fake_generator(audio, sr, composer):
    if composer == "composer1":
        return [Note(n.start + 1.0, n.end + 1.0, n.pitch, n.velocity) for n in chords(PROG, 23)]
    if composer == "composer2":
        return chords(WRONG)
    return [Note(0.5, 0.7, 60)]                       # nearly empty


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # "Original" preview: chords starting at 1 s, encoded as mp3 like Deezer's.
        orig = np.concatenate([np.zeros(44100), synth_piano(chords(PROG), 44100)])
        A.write_wav(d / "orig.wav", orig)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", d / "orig.wav", "-t", "30",
                        "-b:a", "128k", d / "orig.mp3"], check=True)
        sess = FakeSession((d / "orig.mp3").read_bytes())
        dz = Deezer(session=sess, min_interval=0)
        cfg = {
            "clip_seconds": 30, "min_score": 0.45, "soundfont": None, "method": "pop2piano",
            "composers": ["composer1", "composer2", "composer3"],
            "languages": {"he": {"sources": [
                {"type": "chart", "label": "Chart", "max_tracks": 10},
                {"type": "playlist_search", "query": "Top Israel", "max_tracks": 10},
                {"type": "playlist", "playlist_id": 999, "label": "broken"},   # error path
            ]}},
        }
        logs = []                     # playlist 999 simulates a network failure:
        orig_get = sess.get
        def get(url, **kw):
            if "/playlist/999" in url:
                raise OSError("network down")
            return orig_get(url, **kw)
        sess.get = get

        songs = B.collect(cfg, "he", d / "out", dz, log=logs.append)
        ids = [s["deezer_id"] for s in songs]
        assert len(songs) == 7, ids                                   # 4 + 4 - 1 duplicate
        assert ids[:2] == [100, 200]                                  # round-robin mix
        hit = next(s for s in songs if s["deezer_id"] == 900)
        assert hit["sources"] == ["Chart", "Top Israel"]
        assert songs[0]["title"] == "Chart Song 0" and songs[0]["answers"][0] == "Chart Song 0"
        assert any("[id 8]" in l for l in logs), logs                 # picked Deezer's list
        assert any("broken" in l and "network down" in l for l in logs)

        cfg2 = {"languages": {"he": {"sources": [{"type": "songs", "label": "picked",
                 "songs": ["Artist - Song", "Nobody - Nothing"]}]}}}
        picked = B.collect(cfg2, "he", d / "tmp", dz, log=logs.append)
        assert [p["deezer_id"] for p in picked] == [700]
        assert any("not found on Deezer: Nobody - Nothing" in l for l in logs)

        m = B.build(cfg, "he", d / "out", limit=2, variants=3,
                    generator=fake_generator, dz=dz, log=logs.append)
        assert len(m) == 2
        s = m[100]
        print("  picked:", s["composer"], s["score"], "offset", s["piano_offset_sec"])
        assert s["composer"] == "composer1" and s["status"] == "ok"
        assert abs(s["piano_offset_sec"] - 0.85) < 0.01
        clip = d / "out" / "he" / s["piano_file"]
        dur = len(A.decode(clip, 22050)) / 22050
        assert 21 < dur <= 30.1, dur
        assert A.decode(clip, 22050)[:6615].std() > 0.01             # sound within 0.3 s

        # resume: nothing new until limit grows
        n_calls = len(sess.calls)
        B.build(cfg, "he", d / "out", limit=2, generator=fake_generator, dz=dz, log=logs.append)
        assert len(sess.calls) == n_calls

        # regenerate a specific song with untried composers only
        cfg["composers"] = None
        m = B.build(cfg, "he", d / "out", ids=[100], variants=2,
                    generator=fake_generator, dz=dz, log=logs.append)
        assert len(m[100]["tried_composers"]) == 5
        man = json.loads((d / "out" / "he" / "manifest.json").read_text())
        assert man["count"] == 2 and man["songs"][0]["deezer_id"] == 100

        # instrumental method: real audio minus vocals, from the chorus
        class FakeMaker:
            def __call__(self, mp3, wav_out):
                st = A.decode(mp3, 44100)
                A.write_wav(wav_out, np.stack([st, st * 0.5]), 44100)
                self.last_info = {"source_offset": 4.5, "tempo": 120, "vocal_level": 0.1}
                return 4.5
        cfg_i = dict(cfg, method="instrumental")
        m = B.build(cfg_i, "he", d / "out", ids=[200], generator=FakeMaker(), dz=dz, log=logs.append)
        s = m[200]
        assert s["method"] == "instrumental" and s["piano_offset_sec"] == 4.5 and "score" not in s
        clip = d / "out" / "he" / s["piano_file"]
        out = subprocess.run(["ffmpeg", "-i", clip], capture_output=True, text=True).stderr
        assert "stereo" in out, out
        dur = len(A.decode(clip, 22050)) / 22050
        assert 21.5 < dur < 22.2, dur                                     # 26.5 s source - 4.5 s

        page = B.review("he", d / "out")
        assert "Chart Song 0" in page.read_text() and "originals/100.mp3" in page.read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
