"use strict";
const STEP = 2;            // seconds added per step
const ROUND = 10;          // songs per game
const AUTO_NEXT = 12;      // seconds on the reveal screen before the next song

const T = {
  en: {
    title: "Piano Guess", sub: "Hear the hit as piano. Name it as fast as you can.",
    start: "Start", ready: "{n} songs ready", loading: "Loading songs…",
    none: "Songs are still being made. Check back soon.",
    how: "<b>How to play:</b> you hear 2 seconds of a hit played on piano. Tap <b>+2 sec</b> to hear more. Say the song name out loud, or type it. Fewer seconds = more points.",
    sec: "<b>{n}</b> sec", more: "+2 sec", giveUp: "Give up", placeholder: "Type the song name…",
    listening: "Listening…", heard: "“{t}” – not it", notIt: "Not it – try again",
    correct: "Correct!", missed: "The song was", next: "Next song", nextIn: "Next song ({n})",
    stay: "Stay here", round: "Song {i} of {n}", pts: "{n} pts", gain: "+{n} points · {s} sec",
    final: "Final score", again: "Play again", home: "Home", last: "See results",
    micDenied: "Microphone is blocked – allow it in settings, or type instead.",
    noMic: "Voice isn't supported in this browser – type instead.",
    noSpeech: "Didn't catch that – tap the mic and try again",
    playReal: "▶ Play the real song", offline: "Couldn't load songs. Check your connection.",
  },
  he: {
    title: "נחש את השיר", sub: "שומעים את הלהיט בפסנתר. מנחשים כמה שיותר מהר.",
    start: "התחל", ready: "{n} שירים מוכנים", loading: "טוען שירים…",
    none: "השירים עוד בהכנה. נסו שוב בקרוב.",
    how: "<b>איך משחקים:</b> שומעים 2 שניות של להיט בפסנתר. לוחצים <b>+2 שנ׳</b> כדי לשמוע עוד. אומרים את שם השיר בקול או מקלידים. פחות שניות = יותר נקודות.",
    sec: "<b>{n}</b> שנ׳", more: "+2 שנ׳", giveUp: "ויתרתי", placeholder: "הקלידו את שם השיר…",
    listening: "מקשיב…", heard: "„{t}” – לא זה", notIt: "לא זה – נסו שוב",
    correct: "נכון!", missed: "השיר היה", next: "לשיר הבא", nextIn: "לשיר הבא ({n})",
    stay: "להישאר כאן", round: "שיר {i} מתוך {n}", pts: "{n} נק׳", gain: "+{n} נקודות · {s} שנ׳",
    final: "ניקוד סופי", again: "משחק חדש", home: "למסך הבית", last: "לתוצאות",
    micDenied: "המיקרופון חסום – אפשרו אותו בהגדרות, או הקלידו.",
    noMic: "זיהוי קולי לא נתמך בדפדפן הזה – הקלידו.",
    noSpeech: "לא שמעתי – לחצו על המיקרופון ונסו שוב",
    playReal: "▶ השמע את השיר המקורי", offline: "לא הצלחתי לטעון שירים. בדקו את החיבור.",
  },
};

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

const S = {
  lang: store.get("lang", (navigator.language || "").startsWith("he") ? "he" : "en"),
  lib: {},          // lang -> songs
  queue: [], i: 0, step: 1, maxSteps: 15, score: 0, results: [],
  song: null, done: false, nextTimer: null, rec: null, listening: false, micBlocked: false,
};
const piano = new Audio();
const real = new Audio();
piano.preload = real.preload = "auto";
let stopAt = 0, raf = 0;

const t = (k, vars) => (T[S.lang][k] || k).replace(/\{(\w+)\}/g, (_, v) => vars?.[v] ?? "");

function show(id) {
  for (const s of document.querySelectorAll("section")) s.hidden = s.id !== id;
}

// ---------------------------------------------------------------- home
function applyLang() {
  document.documentElement.lang = S.lang;
  document.documentElement.dir = S.lang === "he" ? "rtl" : "ltr";
  document.title = t("title");
  for (const el of document.querySelectorAll("[data-t]")) el.innerHTML = t(el.dataset.t);
  for (const b of document.querySelectorAll("[data-lang]")) b.classList.toggle("on", b.dataset.lang === S.lang);
  $("input").placeholder = t("placeholder");
}

async function loadLib(lang) {
  if (S.lib[lang]) return S.lib[lang];
  const r = await fetch(`data/${lang}/manifest.json`, { cache: "no-cache" });
  if (r.status === 404) return (S.lib[lang] = []);
  if (!r.ok) throw new Error(r.status);
  const m = await r.json();
  return (S.lib[lang] = (m.songs || []).filter((s) => s.piano_file)
    .map((s) => ({ ...s, url: `data/${lang}/${s.piano_file}` })));
}

async function refreshHome() {
  applyLang();
  const st = $("status"), lang = S.lang;
  st.textContent = t("loading");
  $("start").disabled = true;
  try {
    const songs = await loadLib(lang);
    if (lang !== S.lang) return;
    st.textContent = songs.length ? t("ready", { n: songs.length }) : t("none");
    $("start").disabled = !songs.length;
  } catch {
    st.textContent = t("offline");
  }
}

for (const b of document.querySelectorAll("[data-lang]")) {
  b.onclick = () => { S.lang = b.dataset.lang; store.set("lang", S.lang); refreshHome(); };
}

// ---------------------------------------------------------------- game setup
function buildQueue(songs) {
  const key = "played_" + S.lang;
  const recent = new Set(store.get(key, []));
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.random() * (i + 1) | 0; [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const fresh = shuffle(songs.filter((s) => !recent.has(s.deezer_id)));
  const old = shuffle(songs.filter((s) => recent.has(s.deezer_id)));
  return fresh.concat(old).slice(0, ROUND);
}

function unlockAudio() {
  // iOS only lets audio play after a tap. The piano starts inside this tap anyway;
  // prime the real-song player now so it can start by itself later.
  if (real.src) return;
  real.muted = true;
  real.src = S.queue[0].url;
  real.play().then(() => { real.pause(); real.muted = false; }).catch(() => { real.muted = false; });
}

$("start").onclick = () => {
  S.queue = buildQueue(S.lib[S.lang] || []);
  if (!S.queue.length) return;
  S.i = 0; S.score = 0; S.results = [];
  piano.src = S.queue[0].url;
  unlockAudio();
  startSong();
};

function startSong() {
  clearTimeout(S.nextTimer);
  real.pause();
  S.song = S.queue[S.i];
  S.step = 1; S.done = false;
  S.maxSteps = Math.ceil(30 / STEP);
  piano.muted = false;
  if (!piano.src.endsWith(S.song.url)) piano.src = S.song.url;
  piano.onloadedmetadata = () => {
    if (isFinite(piano.duration)) S.maxSteps = Math.max(1, Math.ceil((piano.duration - 0.3) / STEP));
    drawBar();
  };
  const next = S.queue[S.i + 1];
  if (next) fetch(next.url).catch(() => {});        // warm the cache
  $("input").value = ""; hideSugg(); feedback("");
  $("roundLbl").textContent = t("round", { i: S.i + 1, n: S.queue.length });
  $("ptsLbl").textContent = t("pts", { n: S.score });
  drawBar();
  show("game");
  playSegment();
}

// ---------------------------------------------------------------- playback
function drawBar() {
  const bar = $("bar");
  if (bar.children.length !== S.maxSteps) {
    bar.innerHTML = "";
    for (let k = 0; k < S.maxSteps; k++) bar.appendChild(document.createElement("span")).appendChild(document.createElement("i"));
  }
  [...bar.children].forEach((s, k) => s.classList.toggle("open", k < S.step));
  $("secs").innerHTML = t("sec", { n: S.step * STEP });
  $("more").disabled = S.step >= S.maxSteps;
  fillBar(0);
}

function fillBar(time) {
  [...$("bar").children].forEach((s, k) => {
    const f = Math.max(0, Math.min(1, (time - k * STEP) / STEP));
    s.firstChild.style.transform = `scaleX(${f})`;
  });
}

function setPlayIcon(playing) {
  $("playIcon").innerHTML = playing ? '<path d="M6 5h4v14H6zm8 0h4v14h-4z"/>' : '<path d="M8 5v14l11-7z"/>';
}

function playSegment() {
  stopListening();
  stopAt = S.step * STEP;
  piano.currentTime = 0;
  const p = piano.play();
  if (p) p.catch(() => setPlayIcon(false));
  setPlayIcon(true);
  cancelAnimationFrame(raf);
  const tick = () => {
    fillBar(piano.currentTime);
    if (piano.currentTime >= stopAt || piano.ended) {
      piano.pause();
      setPlayIcon(false);
      fillBar(stopAt);
      onSegmentEnd();
      return;
    }
    if (!piano.paused) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

function stopPiano() { piano.pause(); cancelAnimationFrame(raf); setPlayIcon(false); }

function onSegmentEnd() {
  // Hands-free: listen for the answer right after the music stops.
  if (!S.done && !S.micBlocked && document.activeElement !== $("input")) listen(true);
}

$("play").onclick = () => (piano.paused ? playSegment() : stopPiano());
$("more").onclick = () => { if (S.step < S.maxSteps) { S.step++; drawBar(); playSegment(); } };
$("giveup").onclick = () => finishSong(false);
$("quit").onclick = () => { stopPiano(); stopListening(); real.pause(); clearTimeout(S.nextTimer); refreshHome(); show("home"); };

// ---------------------------------------------------------------- guessing
function feedback(text, cls) {
  const f = $("feedback");
  f.textContent = text;
  f.className = "feedback" + (cls ? " " + cls : "");
}

function tryGuess(texts, final) {
  if (S.done) return true;
  const list = (Array.isArray(texts) ? texts : [texts]).filter(Boolean);
  if (list.some((x) => Match.isCorrect(x, S.song.answers || [S.song.title]))) {
    finishSong(true);
    return true;
  }
  if (final && list.length) {
    feedback(t("heard", { t: list[0] }), "bad");
    const g = $("game").querySelector(".guess");
    g.classList.remove("shake"); void g.offsetWidth; g.classList.add("shake");
  }
  return false;
}

// typing + suggestions
const titles = () => (S.lib[S.lang] || []).map((s) => s.title);
function hideSugg() { $("sugg").hidden = true; }
$("input").addEventListener("input", (e) => {
  const list = Match.suggest(e.target.value, titles(), 5);
  const box = $("sugg");
  box.innerHTML = "";
  for (const title of list) {
    const b = document.createElement("button");
    b.textContent = title;
    b.onpointerdown = (ev) => { ev.preventDefault(); submitTyped(title); };
    box.appendChild(b);
  }
  box.hidden = !list.length;
});
$("input").addEventListener("focus", () => { stopListening(); });
$("input").addEventListener("blur", () => setTimeout(hideSugg, 150));
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter") submitTyped(e.target.value); });
function submitTyped(text) {
  hideSugg();
  if (!text.trim()) return;
  if (!tryGuess(text, true)) { $("input").value = ""; }
  else $("input").blur();
}

// voice
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SR) $("mic").style.opacity = 0.4;

function listen(auto) {
  if (!SR) { if (!auto) feedback(t("noMic"), "bad"); return; }
  if (S.listening || S.done) return;
  stopPiano();
  const rec = new SR();
  rec.lang = S.lang === "he" ? "he-IL" : "en-US";
  rec.interimResults = true;
  rec.maxAlternatives = 5;
  rec.continuous = false;
  S.rec = rec; S.listening = true;
  $("mic").classList.add("on");
  feedback(t("listening"), "live");
  let heard = [];
  rec.onresult = (e) => {
    const res = e.results[e.resultIndex];
    const alts = [...res].map((a) => a.transcript);
    // Join everything said so far too (Safari splits phrases).
    const joined = [...e.results].map((r) => r[0].transcript).join(" ");
    heard = [joined, ...alts];
    feedback(alts[0], "live");
    if (tryGuess(heard, false)) rec.abort();
  };
  rec.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      if (!auto) { S.micBlocked = true; feedback(t("micDenied"), "bad"); }
      else feedback("");
    } else if (e.error === "no-speech") {
      if (!auto) feedback(t("noSpeech"));
      else feedback("");
    }
  };
  rec.onend = () => {
    S.listening = false; S.rec = null;
    $("mic").classList.remove("on");
    if (S.done) return;
    if (heard.length) tryGuess(heard, true);
    else if ($("feedback").textContent === t("listening")) feedback("");
  };
  try { rec.start(); } catch { S.listening = false; $("mic").classList.remove("on"); }
}

function stopListening() {
  if (S.rec) { try { S.rec.abort(); } catch {} }
  S.listening = false;
  $("mic").classList.remove("on");
}

$("mic").onclick = () => { S.micBlocked = false; S.listening ? stopListening() : listen(false); };

// ---------------------------------------------------------------- reveal
function finishSong(won) {
  if (S.done) return;
  S.done = true;
  stopPiano(); stopListening(); hideSugg(); $("input").blur();
  const pts = won ? S.maxSteps + 1 - S.step : 0;
  S.score += pts;
  S.results.push({ song: S.song, pts, secs: S.step * STEP });
  const played = store.get("played_" + S.lang, []).filter((id) => id !== S.song.deezer_id);
  store.set("played_" + S.lang, [S.song.deezer_id, ...played].slice(0, 60));
  if (won && navigator.vibrate) navigator.vibrate([30, 40, 30]);

  $("verdict").textContent = won ? t("correct") : t("missed");
  $("verdict").className = "verdict " + (won ? "good" : "bad");
  $("cover").src = S.song.cover || "";
  $("rTitle").textContent = S.song.title;
  $("rArtist").textContent = S.song.artist;
  $("gain").textContent = won ? t("gain", { n: pts, s: S.step * STEP }) : "";
  $("playReal").hidden = true;
  $("playReal").textContent = t("playReal");
  show("reveal");
  playReal(S.song);
  startCountdown();
}

function deezerTrack(id) {
  // Deezer's API has no CORS headers, so use its JSONP mode.
  return new Promise((resolve, reject) => {
    const cb = "dz_" + Math.random().toString(36).slice(2);
    const s = document.createElement("script");
    const done = () => { delete window[cb]; s.remove(); clearTimeout(timer); };
    const timer = setTimeout(() => { done(); reject(new Error("timeout")); }, 8000);
    window[cb] = (d) => { done(); resolve(d); };
    s.onerror = () => { done(); reject(new Error("load")); };
    s.src = `https://api.deezer.com/track/${id}?output=jsonp&callback=${cb}`;
    document.head.appendChild(s);
  });
}

async function playReal(song) {
  try {
    const d = await deezerTrack(song.deezer_id);
    if (S.song !== song || !d.preview) return;
    real.src = d.preview;
    real.onloadedmetadata = () => { real.currentTime = song.piano_offset_sec || 0; };
    await real.play();
  } catch {
    if (S.song === song && real.src) $("playReal").hidden = false;   // autoplay blocked
  }
}
$("playReal").onclick = () => { $("playReal").hidden = true; real.play().catch(() => {}); };

function nextLabel() { return S.i + 1 >= S.queue.length ? t("last") : t("next"); }

function startCountdown() {
  let n = AUTO_NEXT;
  const btn = $("next");
  $("stay").hidden = false;
  const step = () => {
    if (n <= 0) return goNext();
    btn.textContent = S.i + 1 >= S.queue.length ? t("last") : t("nextIn", { n });
    n--;
    S.nextTimer = setTimeout(step, 1000);
  };
  step();
}

$("stay").onclick = () => { clearTimeout(S.nextTimer); $("next").textContent = nextLabel(); $("stay").hidden = true; };
$("next").onclick = () => goNext();

function goNext() {
  clearTimeout(S.nextTimer);
  real.pause();
  S.i++;
  if (S.i >= S.queue.length) return showSummary();
  startSong();
}

// ---------------------------------------------------------------- summary
function showSummary() {
  $("final").textContent = S.score;
  const list = $("list");
  list.innerHTML = "";
  for (const r of S.results) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<img alt=""><div><b></b><small></small></div><span class="p"></span>`;
    row.querySelector("img").src = r.song.cover || "";
    row.querySelector("b").textContent = r.song.title;
    row.querySelector("small").textContent = r.song.artist;
    const p = row.querySelector(".p");
    p.textContent = r.pts ? "+" + r.pts : "–";
    if (!r.pts) p.classList.add("z");
    list.appendChild(row);
  }
  show("summary");
}
$("again").onclick = () => $("start").onclick();
$("homeBtn").onclick = () => { refreshHome(); show("home"); };

refreshHome();
