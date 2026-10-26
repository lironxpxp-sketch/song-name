"""Command line:

  python -m songgame find-playlists "Top Israel"
  python -m songgame collect --lang he
  python -m songgame build   --lang he --limit 20
  python -m songgame build   --lang he --ids 123,456 --variants 5   # redo bad ones
  python -m songgame review  --lang he
"""
import argparse
import json
import os
from pathlib import Path

from . import build as B
from .deezer import Deezer

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    p = argparse.ArgumentParser(prog="songgame")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--out", default=str(ROOT / "output"))
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("find-playlists", help="search Deezer playlists (to pin IDs in config)")
    f.add_argument("query")

    c = sub.add_parser("collect", help="fetch song lists from Deezer")
    c.add_argument("--lang", required=True)

    b = sub.add_parser("build", help="generate piano clips")
    b.add_argument("--lang", required=True)
    b.add_argument("--limit", type=int, help="stop when this many songs are done in total")
    b.add_argument("--variants", type=int, help="piano versions to try per song")
    b.add_argument("--ids", help="comma-separated Deezer IDs to (re)generate")
    b.add_argument("--all", action="store_true", help="regenerate every song already made")

    r = sub.add_parser("review", help="write a listening page for checking quality")
    r.add_argument("--lang", required=True)

    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text("utf-8"))
    cfg["soundfont"] = os.environ.get("SONGGAME_SOUNDFONT") or cfg.get("soundfont")

    if a.cmd == "find-playlists":
        for pl in Deezer().search_playlists(a.query):
            print(f"{pl['id']:>12}  {pl['nb_tracks']:>4} tracks  {pl['title']}  "
                  f"— by {pl['user']['name']}")
    elif a.cmd == "collect":
        B.collect(cfg, a.lang, a.out)
    elif a.cmd == "build":
        ids = [int(x) for x in a.ids.split(",")] if a.ids else None
        if a.all:
            man = Path(a.out) / a.lang / "manifest.json"
            ids = [s["deezer_id"] for s in json.loads(man.read_text("utf-8"))["songs"]] if man.exists() else None
        B.build(cfg, a.lang, a.out, a.limit, a.variants, ids)
        print(f"Done. Listen: python -m songgame review --lang {a.lang}")
    elif a.cmd == "review":
        print(f"Open {B.review(a.lang, a.out)} in a browser")


if __name__ == "__main__":
    main()
