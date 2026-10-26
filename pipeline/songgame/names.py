"""Title/artist cleanup: accepted answers and duplicate detection.

The app does the fuzzy matching at play time (typos, partial names); this module
prepares clean answer strings so that matching starts from the right text.
"""
import re
import unicodedata

# Hebrew points and cantillation marks (niqqud), which players won't type or say.
_HEBREW_MARKS = re.compile(r"[֑-ׇ]")
_BRACKETS = re.compile(r"\s*[\(\[\{][^\)\]\}]*[\)\]\}]\s*")
_VERSION_WORDS = re.compile(
    r"\b(feat|ft|featuring|remaster(ed)?|radio edit|edit|version|live|mix|remix|"
    r"mono|stereo|from|prod|acoustic|deluxe|bonus|explicit|clean)\b",
    re.IGNORECASE,
)
# A trailing " - Remastered 2011" style suffix.
_DASH_SUFFIX = re.compile(r"\s+[-–—]\s+(.*)$")


def normalize(text: str) -> str:
    """Lowercase, strip niqqud/punctuation/accents, collapse spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _HEBREW_MARKS.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"['’`\"״׳]", "", text)          # apostrophes, geresh, gershayim
    text = re.sub(r"[^\w\s]", " ", text)            # other punctuation -> space
    return re.sub(r"\s+", " ", text).strip()


def strip_version(title: str) -> str:
    """'Song (feat. X) - Remastered 2011' -> 'Song'."""
    title = title or ""
    m = _DASH_SUFFIX.search(title)
    if m and _VERSION_WORDS.search(m.group(1)):
        title = title[: m.start()]
    # Remove bracketed parts that are clearly version info.
    def _drop(match):
        return " " if _VERSION_WORDS.search(match.group(0)) else match.group(0)
    title = _BRACKETS.sub(_drop, title)
    return re.sub(r"\s+", " ", title).strip()


def answers_for(title: str, title_short: str | None = None) -> list[str]:
    """Accepted answer strings, most complete first, deduplicated."""
    candidates = [
        strip_version(title_short or title),
        strip_version(title),
        _BRACKETS.sub(" ", title or ""),              # without any brackets
        strip_version((title or "").split(" - ")[0]),
        strip_version((title or "").split("/")[0]),   # 'Song A / Song B' medleys
    ]
    out, seen = [], set()
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        key = normalize(c)
        if c and key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def dedup_key(title: str, artist: str) -> str:
    primary_artist = re.split(r",|&| feat\.? | ft\.? | x ", artist or "", flags=re.I)[0]
    return f"{normalize(strip_version(title))}|{normalize(primary_artist)}"
