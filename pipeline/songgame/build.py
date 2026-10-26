"""Pipeline steps: collect songs -> make piano clips -> manifest the app reads."""
import json
import random
import tempfile
import time
from pathlib import Path

from . import audio as A
from . import names
from .deezer import Deezer, DeezerError
from .piano import COMPOSERS, first_onset, render, score

GEN_SR = 44100     # Pop2Piano input rate
SCORE_SR = 22050


def _load(path, default):
    p = Path(path)
    return json.loads(p.read_text("utf-8")) if p.exists() else default


def _save(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# 1. Collect
# ---------------------------------------------------------------------------
def collect(cfg, lang, out_dir, dz: Deezer = None, log=print):
    """Build out/<lang>/songs.json from the language's sources, mixed and deduped."""
    dz = dz or Deezer()
    sources = cfg["languages"][lang]["sources"]
    lists = []
    for src in sources:
        label = src.get("label") or src.get("query") or str(src.get("playlist_id", "chart"))
        limit = src.get("max_tracks", 100)
        try:
            if src["type"] == "chart":
                tracks = dz.chart_tracks(limit, src.get("genre_id", 0))
            elif src["type"] == "songs":
                tracks = []
                for item in src["songs"][:limit]:
                    artist, _, title = item.partition(" - ")
                    hit = dz.search_track(artist.strip(), title.strip())
                    if hit:
                        tracks.append(hit)
                        log(f"    ✓ {item}  ->  {hit['artist']['name']} - {hit['title']}")
                    else:
                        log(f"    ! not found on Deezer: {item}")
            elif src["type"] == "playlist":
                tracks = dz.playlist_tracks(src["playlist_id"], limit)
            elif src["type"] == "playlist_search":
                pl = dz.find_playlist(src["query"], src.get("creator_contains", "deezer"))
                if not pl:
                    log(f"  ! no playlist found for '{src['query']}'")
                    continue
                note = "" if pl["_matched_creator"] else "  (not by Deezer editors - check it)"
                log(f"  '{src['query']}' -> {pl['title']} by {pl['user']['name']} "
                    f"[id {pl['id']}]{note}")
                tracks = dz.playlist_tracks(pl["id"], limit)
            else:
                raise ValueError(f"unknown source type {src['type']}")
        except (DeezerError, OSError) as e:
            log(f"  ! {label}: {e}")
            continue
        lists.append((label, tracks))
        log(f"  {label}: {len(tracks)} tracks")

    songs, by_key = [], {}
    # Round-robin across sources so any prefix of the list mixes eras/charts.
    for i in range(max((len(t) for _, t in lists), default=0)):
        for label, tracks in lists:
            if i >= len(tracks):
                continue
            t = tracks[i]
            if not t.get("readable", True) or not t.get("id"):
                continue
            artist = t.get("artist", {}).get("name", "")
            key = names.dedup_key(t.get("title", ""), artist)
            if key in by_key:
                if label not in by_key[key]["sources"]:
                    by_key[key]["sources"].append(label)
                continue
            song = {
                "deezer_id": t["id"],
                "title": names.strip_version(t.get("title_short") or t["title"]),
                "artist": artist,
                "answers": names.answers_for(t["title"], t.get("title_short")),
                "cover": t.get("album", {}).get("cover_medium"),
                "deezer_link": t.get("link"),
                "sources": [label],
            }
            by_key[key] = song
            songs.append(song)
    _save(Path(out_dir) / lang / "songs.json", songs)
    log(f"{len(songs)} unique songs saved for '{lang}'")
    return songs


# ---------------------------------------------------------------------------
# 2. Build piano clips
# ---------------------------------------------------------------------------
def make_piano(song, lang_dir, generator, composers, cfg, dz: Deezer, log=print):
    """Generate one or more piano versions for a song and keep the best."""
    sid = song["deezer_id"]
    mp3 = dz.download_preview(sid, lang_dir / "originals" / f"{sid}.mp3")
    gen_audio = A.decode(mp3, GEN_SR)
    ref_audio = A.decode(mp3, SCORE_SR)
    best = None
    with tempfile.TemporaryDirectory() as tmp:
        for comp in composers:
            t0 = time.time()
            notes = generator(gen_audio, GEN_SR, comp)
            wav = Path(tmp) / f"{comp}.wav"
            render(notes, wav, cfg.get("soundfont"))
            info = getattr(generator, "last_info", None)
            if info and "source_offset" in info:
                s = info["voiced"]          # cut/slowed clip: harmony score no longer lines up
            else:
                s = score(ref_audio, A.decode(wav, SCORE_SR), notes, SCORE_SR)
            log(f"    {comp}: {len(notes)} notes, score {s:.3f} ({time.time()-t0:.0f}s)"
                + (f" {info}" if info else ""))
            if best is None or s > best[0]:
                best = (s, comp, notes, wav)
        s, comp, notes, wav = best
        start = clip_start(notes, cfg.get("clip_seconds", 30))
        clip = lang_dir / "piano" / f"{sid}.m4a"
        A.encode_m4a(wav, clip, start, cfg.get("clip_seconds", 30))
    song.update({
        "piano_file": f"piano/{sid}.m4a",
        "piano_offset_sec": round(start, 2),   # where the clip starts in the real preview
        "score": s,
        "composer": comp,
        "tried_composers": sorted(set(song.get("tried_composers", [])) | set(composers)),
        "status": "ok" if s >= cfg.get("min_score", 0.45) else "needs_review",
        "method": cfg.get("method", "melody"),
    })
    info = getattr(generator, "last_info", None)
    if info:
        song["arrangement"] = info
        if "source_offset" in info:
            song["piano_offset_sec"] = round(info["source_offset"] + start / info.get("stretch", 1), 2)
        if info.get("melody_notes", 99) < 12:
            song["status"] = "needs_review"      # hardly any singing found
    return song


def clip_start(notes, clip_len=30.0):
    """Start where the singing starts (that's what people recognize), unless
    that leaves too short a clip; otherwise start at the first note."""
    mel = [n.start for n in notes if n.role == "melody"]
    if mel and min(mel) <= clip_len - 18:
        return round(max(0.0, min(mel) - 0.4), 2)
    return round(max(0.0, first_onset(notes) - 0.15), 2)


def _pick_composers(song, n, cfg):
    if cfg.get("method", "melody") == "melody":
        return ["melody"]
    tried = set(song.get("tried_composers", []))
    pool = [c for c in cfg.get("composers") or COMPOSERS if c not in tried] or COMPOSERS
    rng = random.Random(song["deezer_id"] + len(tried))
    return rng.sample(pool, min(n, len(pool)))


def build(cfg, lang, out_dir, limit=None, variants=None, ids=None, generator=None,
          dz: Deezer = None, log=print):
    """Make piano clips for songs that don't have one yet (or for `ids`, again)."""
    lang_dir = Path(out_dir) / lang
    songs = _load(lang_dir / "songs.json", None)
    if songs is None:
        raise SystemExit(f"No song list for '{lang}'. Run: python -m songgame collect --lang {lang}")
    manifest = {s["deezer_id"]: s for s in _load(lang_dir / "manifest.json", {"songs": []})["songs"]}
    variants = variants or cfg.get("variants_per_song", 3)
    dz = dz or Deezer()

    if ids:
        todo = [manifest.get(i) or next(s for s in songs if s["deezer_id"] == i) for i in ids]
    else:
        todo = [s for s in songs if s["deezer_id"] not in manifest]
        if limit is not None:
            have = len(manifest)
            todo = todo[: max(0, limit - have)]
    if not todo:
        log("Nothing to do.")
        return manifest

    if generator is None:
        if cfg.get("method", "melody") == "melody":
            log("Loading the vocal separation model (first run downloads it)...")
            from .melody import MelodyArranger
            generator = MelodyArranger(cfg.get("device", "auto"), cfg.get("target_bpm", 100),
                                       cfg.get("clip_seconds", 30))
        else:
            log("Loading Pop2Piano (first run downloads the model)...")
            from .piano import Pop2PianoGenerator
            generator = Pop2PianoGenerator(cfg.get("device", "auto"))

    for i, song in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {song['artist']} - {song['title']}")
        try:
            done = make_piano(dict(song), lang_dir, generator,
                              _pick_composers(song, variants, cfg), cfg, dz, log)
        except Exception as e:                       # keep going on bad tracks
            import traceback
            log(f"    ! skipped: {e!r}\n" + "".join(traceback.format_exc().splitlines(True)[-6:]))
            continue
        manifest[done["deezer_id"]] = done
        _write_manifest(lang_dir, lang, manifest, songs)
    return manifest


def _write_manifest(lang_dir, lang, manifest, songs):
    order = {s["deezer_id"]: i for i, s in enumerate(songs)}
    items = sorted(manifest.values(), key=lambda s: order.get(s["deezer_id"], 1e9))
    _save(lang_dir / "manifest.json", {
        "language": lang,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(items),
        "note": "Fetch the real song at play time with GET https://api.deezer.com/track/<deezer_id> "
                "and use its 'preview' field; preview links expire.",
        "songs": items,
    })


# ---------------------------------------------------------------------------
# 3. Review page
# ---------------------------------------------------------------------------
def review(lang, out_dir):
    import html
    lang_dir = Path(out_dir) / lang
    songs = _load(lang_dir / "manifest.json", {"songs": []})["songs"]
    songs = sorted(songs, key=lambda s: s.get("score", 0))      # worst first
    rows = []
    for s in songs:
        flag = "⚠️ " if s.get("status") == "needs_review" else ""
        rows.append(f"""
<div class="card">
  <img src="{html.escape(s.get('cover') or '')}" alt="">
  <div class="meta">
    <b>{flag}{html.escape(s['title'])}</b><br>{html.escape(s['artist'])}
    <small>id {s['deezer_id']} · score {s.get('score', 0):.2f} · {s.get('composer', '')}</small>
    <label>Piano</label><audio controls preload="none" src="{s['piano_file']}"></audio>
    <label>Original</label><audio controls preload="none" src="originals/{s['deezer_id']}.mp3"></audio>
  </div>
</div>""")
    page = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Piano review – {lang}</title>
<style>
body{{font:15px system-ui,sans-serif;margin:16px;background:#f6f5f2;color:#222}}
.card{{display:flex;gap:12px;background:#fff;border-radius:10px;padding:12px;margin:10px 0}}
img{{width:84px;height:84px;border-radius:6px;object-fit:cover}}
.meta{{flex:1;min-width:0}} small{{display:block;color:#777;margin:2px 0 6px}}
label{{font-size:12px;color:#777}} audio{{width:100%;height:34px}}
</style>
<h1>Piano versions ({lang}) – {len(songs)} songs, lowest score first</h1>
<p>Bad one? Regenerate it: <code>python -m songgame build --lang {lang} --ids ID1,ID2 --variants 5</code></p>
{''.join(rows)}"""
    path = lang_dir / "review.html"
    path.write_text(page, "utf-8")
    return path
