// Forgiving answer matching for typed and spoken guesses (English + Hebrew).
(function (root) {
  "use strict";

  function normalize(s) {
    s = (s || "").normalize("NFKC").replace(/[֑-ׇ]/g, "");   // niqqud
    s = s.normalize("NFKD").replace(/[̀-ͯ]/g, "");            // accents
    s = s.toLowerCase().replace(/&/g, " and ");
    s = s.replace(/['’`"״׳]/g, "");                                     // apostrophes, geresh
    s = s.replace(/[^\p{L}\p{N}\s]/gu, " ");
    return s.replace(/\s+/g, " ").trim();
  }

  function lev(a, b) {
    if (a === b) return 0;
    if (!a.length) return b.length;
    if (!b.length) return a.length;
    let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      for (let j = 1; j <= b.length; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1,
          prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      }
      prev = cur;
    }
    return prev[b.length];
  }

  const STOP = new Set(["the", "a", "an", "of", "to", "in", "on", "and", "is", "it",
    "i", "my", "me", "you", "your", "we", "be", "for", "at", "by", "with", "this", "that",
    "של", "את", "על", "עם", "זה", "אני", "אתה", "את", "כי", "לא", "כל", "יש"]);
  const HEB_PREFIX = /^[והבלמשכ]/;

  // Letters allowed to differ, by length.
  const wordTol = (n) => (n >= 8 ? 2 : n >= 4 ? 1 : 0);
  const fullTol = (n) => (n <= 4 ? 0 : n <= 8 ? 1 : n <= 15 ? 2 : 3);

  function wordEq(w, t) {
    if (w === t) return true;
    if (lev(w, t) <= wordTol(t.length)) return true;
    // Hebrew: "והשמש" ~ "השמש", "בלילה" ~ "לילה"
    if (HEB_PREFIX.test(w) && w.length - 1 >= 3 && w.slice(1) === t) return true;
    if (HEB_PREFIX.test(t) && t.length - 1 >= 3 && t.slice(1) === w) return true;
    return false;
  }

  function matchesAnswer(guess, answer) {
    const g = normalize(guess), a = normalize(answer);
    if (!g || !a) return false;
    if (g === a) return true;
    if (lev(g.replace(/ /g, ""), a.replace(/ /g, "")) <= fullTol(a.replace(/ /g, "").length)) return true;
    if ((" " + g + " ").includes(" " + a + " ")) return true;          // said with extra words

    // Part of the name: enough of the title's meaningful words.
    const all = a.split(" ");
    const words = all.filter((t) => !STOP.has(t)).length ? all.filter((t) => !STOP.has(t)) : all;
    if (words.length < 2) return false;                                 // 1-word titles need the word
    const said = g.split(" ");
    const hit = words.filter((t) => said.some((w) => wordEq(w, t)));
    const letters = a.replace(/ /g, "").length;
    const hitLetters = hit.join("").length;
    return hit.length >= Math.ceil(words.length / 2) && hitLetters >= Math.max(5, 0.4 * letters);
  }

  function isCorrect(guess, answers) {
    return (answers || []).some((a) => matchesAnswer(guess, a));
  }

  // Autocomplete: titles from the library that fit what's been typed so far.
  function suggest(query, titles, max) {
    const q = normalize(query);
    if (q.length < 2) return [];
    const seen = new Set(), out = [];
    for (const title of titles) {
      const t = normalize(title);
      if (!t || seen.has(t)) continue;
      let s = 0;
      if (t.startsWith(q)) s = 3;
      else if (t.split(" ").some((w) => w.startsWith(q))) s = 2;
      else if (q.length >= 3 && t.includes(q)) s = 1;
      else if (q.length >= 4 && t.split(" ").some((w) => lev(w.slice(0, q.length), q) <= 1)) s = 0.5;
      if (s) { seen.add(t); out.push([s, title]); }
    }
    out.sort((x, y) => y[0] - x[0] || x[1].length - y[1].length);
    return out.slice(0, max || 6).map((x) => x[1]);
  }

  const api = { normalize, lev, isCorrect, matchesAnswer, suggest };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Match = api;
})(this);
