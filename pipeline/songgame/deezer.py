"""Deezer public API: charts, playlists, track lookup, preview download.

No key or login is needed. Deezer allows ~50 requests per 5 seconds; the client
throttles itself and retries when it hits the quota.
"""
import time
from pathlib import Path

import requests

API = "https://api.deezer.com"


class DeezerError(RuntimeError):
    pass


class Deezer:
    def __init__(self, session=None, min_interval=0.12, max_retries=5):
        self.s = session or requests.Session()
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def get(self, path, **params):
        url = path if path.startswith("http") else f"{API}{path}"
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.s.get(url, params=params or None, timeout=20)
            r.raise_for_status()
            data = r.json()
            err = data.get("error") if isinstance(data, dict) else None
            if not err:
                return data
            if err.get("code") == 4:              # quota exceeded: back off
                time.sleep(2 * (attempt + 1))
                continue
            raise DeezerError(f"{url}: {err}")
        raise DeezerError(f"{url}: still over quota after {self.max_retries} tries")

    def _paged(self, path, limit):
        items, url, params = [], path, {"limit": min(limit, 100)}
        while url and len(items) < limit:
            page = self.get(url, **params)
            items.extend(page.get("data", []))
            url, params = page.get("next"), {}
        return items[:limit]

    # --- sources -----------------------------------------------------------
    def chart_tracks(self, limit=100, genre_id=0):
        return self._paged(f"/chart/{genre_id}/tracks", limit)

    def search_playlists(self, query, limit=25):
        return self.get("/search/playlist", q=query, limit=limit).get("data", [])

    def find_playlist(self, query, creator_contains="deezer"):
        """Best-matching playlist for a search, preferring Deezer's own editors."""
        results = self.search_playlists(query)
        if not results:
            return None
        want = (creator_contains or "").lower()
        preferred = [p for p in results
                     if want and want in (p.get("user", {}).get("name", "").lower())]
        exact = [p for p in preferred if p.get("title", "").lower() == query.lower()]
        pick = (exact or preferred or results)[0]
        pick["_matched_creator"] = bool(preferred)
        return pick

    def playlist_tracks(self, playlist_id, limit=100):
        return self._paged(f"/playlist/{playlist_id}/tracks", limit)

    def search_track(self, artist, title):
        """Best Deezer match for an artist + title. The title must really match
        (no 'similar' results), and studio versions beat live/remix ones."""
        from .names import normalize
        want_t, want_a = normalize(title), normalize(artist)
        seen, cands = set(), []
        for q in (f'artist:"{artist}" track:"{title}"', f"{artist} {title}", title):
            for t in self.get("/search", q=q, limit=25).get("data", []):
                if t["id"] in seen or not t.get("readable", True) or not t.get("preview"):
                    continue
                seen.add(t["id"])
                cands.append(t)
        best, best_score = None, 0.0
        for rank, t in enumerate(cands):
            got = normalize(t.get("title_short") or t.get("title", ""))
            full = normalize(t.get("title", ""))
            if got == want_t:
                sc = 3.0
            elif got.startswith(want_t) or want_t.startswith(got):
                sc = 2.0
            elif want_t in full:
                sc = 1.0
            else:
                continue
            a = normalize(t.get("artist", {}).get("name", ""))
            if want_a and (want_a in a or a in want_a or set(want_a.split()) & set(a.split())):
                sc += 2.0
            if any(w in full for w in ("live", "remix", "acoustic", "karaoke", "instrumental",
                                       "cover", "sped up", "slowed", "version", "גרסת", "הופעה", "לייב")):
                sc -= 1.5
            sc -= rank * 0.01
            if sc > best_score:
                best, best_score = t, sc
        return best if best_score >= 2.5 else None

    def track(self, track_id):
        return self.get(f"/track/{track_id}")

    # --- audio ---------------------------------------------------------------
    def download_preview(self, track_id, dest: Path):
        """Download the 30 s preview. Links can expire, so fetch a fresh one."""
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
        url = self.track(track_id).get("preview")
        if not url:
            raise DeezerError(f"track {track_id} has no preview")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with self.s.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    f.write(chunk)
        tmp.replace(dest)
        return dest
