# Piano Guess 🎹

A song-guessing game for phones. You hear a hit played on piano, 2 seconds at a time,
and say (or type) its name. English and Hebrew libraries.

**Play:** https://lironxpxp-sketch.github.io/song-name/ — open on your phone, then
Share → *Add to Home Screen*.

## How it works

| Part | Where | What it does |
|---|---|---|
| `pipeline/` | GitHub Actions → **Make piano clips** | Takes chart hits from Deezer, turns each 30 s preview into a piano version with the Pop2Piano AI model, keeps the best of several styles, saves clips to `site/data/` |
| `site/` | GitHub Pages | The game. Plays the piano clip in growing steps, listens for the answer, then plays the real song from Deezer |

## Adding songs (from the GitHub app)

Actions → **Make piano clips** → Run workflow.

- **Library size per language**: total songs wanted, e.g. `30`. Only new songs are made.
- **Piano styles per song**: `2` is a good balance; more takes longer but sounds better.
- **Redo IDs**: regenerate specific songs that came out badly (IDs are in `site/data/<lang>/manifest.json`).

Each song takes a few minutes. The game updates automatically when the run finishes.

## Development

- `node site/match.test.js` – answer matching tests (typos, partial names, Hebrew)
- `python pipeline/tests/test_offline.py` – pipeline tests
- Song sources: `pipeline/config.json`
