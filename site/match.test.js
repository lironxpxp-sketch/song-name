// node site/match.test.js
const assert = require("assert");
const { isCorrect, suggest, normalize } = require("./match.js");

const yes = (g, a) => assert.ok(isCorrect(g, a), `should accept "${g}" for ${a}`);
const no = (g, a) => assert.ok(!isCorrect(g, a), `should reject "${g}" for ${a}`);

// exact / case / punctuation
yes("bohemian rhapsody", ["Bohemian Rhapsody"]);
yes("Don't stop me now", ["Don't Stop Me Now"]);
yes("dont stop me now", ["Don't Stop Me Now"]);
// spelling mistakes
yes("bohemian rapsody", ["Bohemian Rhapsody"]);
yes("bohemain rhapsody", ["Bohemian Rhapsody"]);
yes("shape of yuo", ["Shape of You"]);
yes("despasito", ["Despacito"]);
// part of the name
yes("bohemian", ["Bohemian Rhapsody"]);
yes("rolling in the", ["Rolling in the Deep"]);
yes("sweet child", ["Sweet Child O' Mine"]);
// extra words (artist, "I think it's")
yes("queen bohemian rhapsody", ["Bohemian Rhapsody"]);
yes("i think it's hey jude", ["Hey Jude"]);
// wrong
no("hello", ["Bohemian Rhapsody"]);
no("love", ["Crazy in Love"]);
no("deep", ["Rolling in the Deep"]);
no("shape", ["Shape of You"]);
no("stay", ["Hey Jude"]);
no("yesterday", ["Yellow"]);
no("", ["Yellow"]);
// short titles must be (nearly) exact
yes("yellow", ["Yellow"]);
yes("hello", ["Hello"]);
yes("helo", ["Hello"]);       // one letter off is fine
no("help", ["Hello"]);

// Hebrew
yes("ירושלים של זהב", ["ירושלים של זהב"]);
yes("ירושלים של זהב", ["יְרוּשָׁלַיִם שֶׁל זָהָב"]);
yes("ירושלים", ["ירושלים של זהב"]);
yes("ירושליים של זהב", ["ירושלים של זהב"]);
yes("שיר לשלום", ["שיר לשלום"]);
yes("תל אביב", ["תל-אביב"]);
yes("הפרח בגני", ["הפרח בגני"]);
no("זהב", ["ירושלים של זהב"]);
yes("שלום", ["שיר לשלום"]);     // main word of the title
no("שיר", ["שיר לשלום"]);

// suggestions
const lib = ["Bohemian Rhapsody", "Hey Jude", "Hello", "Hotel California", "Shape of You", "ירושלים של זהב"];
assert.deepStrictEqual(suggest("h", lib), []);
assert.deepStrictEqual(suggest("he", lib), ["Hello", "Hey Jude"]);
assert.ok(suggest("cali", lib).includes("Hotel California"));
assert.deepStrictEqual(suggest("ירו", lib), ["ירושלים של זהב"]);
assert.strictEqual(normalize("  Don’t-Stop!! "), "dont stop");
console.log("match tests ok");
