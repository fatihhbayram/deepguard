// R16-T8 precondition remediation — English left in the Turkish customer copy.
//
// The Turkish reviewer packet's Rapor-A (R16P-008) printed raw API words and an English paragraph
// inside a Turkish report: `completed`, `2/2 complete`, `UNVERIFIED · AVAILABLE`, `SUCCESS`, the
// fired rule's meaning, `threshold_not_reached`, `ENRICHMENT_COMPLETE`, `DECIDED`, `(completed)`,
// and two labels that Turkish casing turned into `CLAİM` and `CHECKPOİNT`. Each is pinned below
// against the `raporA` fixture, which has that report's shape.
//
// The audit at the end is the structural half: every customer-facing page is rendered in both
// languages, and every word the Turkish document shares with the English one must be either a
// value from the record or a word classified in `CLASSIFIED` below. A new English word in the
// Turkish copy fails it until someone translates it or classifies it on purpose.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";

register("./support/tsx-loader.mjs", import.meta.url);

const { createElement } = await import("react");
const { renderToStaticMarkup } = await import("react-dom/server");
const { default: Report } = await import("../app/app/report/[id]/page.tsx");
const { default: Dashboard } = await import("../app/app/page.tsx");
const { default: Landing } = await import("../app/page.tsx");
const { default: Login } = await import("../app/login/page.tsx");
const { default: ErrorPage } = await import("../app/error.tsx");
const { FixedLocale } = await import("../app/i18n/client.tsx");
const { CANONICAL, translateCanonical } = await import("../app/i18n/core.ts");
const { FIXTURES, R9_200_SUMMARY } = await import("./support/report-fixtures.mjs");

process.env.API_INTERNAL_URL = "http://api.invalid";

const LOCALES = ["en", "tr"];

const escapeHtml = (value) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#x27;");

function inLocale(locale, page) {
  return renderToStaticMarkup(createElement(FixedLocale, { locale }, page));
}

async function report(payload, locale, status = 200) {
  globalThis.fetch = async (url) =>
    String(url).includes("/feedback")
      ? new Response("{}", { status: 404 })
      : new Response(JSON.stringify(payload), { status });
  const page = await Report({ params: Promise.resolve({ id: payload.id }), searchParams: Promise.resolve({}) });
  return inLocale(locale, page);
}

async function dashboard(locale, list, listStatus = 200) {
  globalThis.fetch = async (url) => {
    const path = new URL(String(url)).pathname;
    if (path === "/health") return Response.json({ status: "ok", database: "ok" });
    if (path === "/api/v1/auth/me") return Response.json({ id: "u", email: "reader@example.com", role: "USER" });
    if (path === "/api/v1/analyses") return Response.json(list, { status: listStatus });
    return new Response("{}", { status: 404 });
  };
  return inLocale(locale, await Dashboard({ searchParams: Promise.resolve({}) }));
}

const LIST = Object.values(FIXTURES).map((payload, index) => ({
  ...payload,
  id: `${String(index).padStart(8, "0")}-${payload.id.slice(9)}`,
}));

/** Each customer-facing surface, rendered in each language. */
const PAGES = {};
for (const locale of LOCALES) {
  for (const [name, payload] of Object.entries(FIXTURES)) {
    (PAGES[`report:${name}`] ??= {})[locale] = await report(payload, locale);
  }
  (PAGES["report:unavailable"] ??= {})[locale] = await report(FIXTURES.raporA, locale, 503);
  (PAGES["dashboard"] ??= {})[locale] = await dashboard(locale, LIST);
  (PAGES["dashboard:empty"] ??= {})[locale] = await dashboard(locale, []);
  (PAGES["dashboard:listError"] ??= {})[locale] = await dashboard(locale, {}, 503);
  (PAGES["landing"] ??= {})[locale] = inLocale(locale, createElement(Landing));
  (PAGES["login"] ??= {})[locale] = inLocale(locale, await Login({ searchParams: Promise.resolve({}) }));
  (PAGES["login:failed"] ??= {})[locale] = inLocale(locale, await Login({ searchParams: Promise.resolve({ error: "1" }) }));
  (PAGES["error"] ??= {})[locale] = inLocale(
    locale,
    createElement(ErrorPage, { error: Object.assign(new Error("x"), { digest: "d1g3st" }), retry: () => {} }),
  );
}

/** What a reader sees: the text, without markup, attributes, styles or scripts. */
function visible(html) {
  return html
    .replace(/<(style|script)[^>]*>[\s\S]*?<\/\1>/g, " ")
    .replace(/<[^>]*>/g, " ")
    .replaceAll("&quot;", '"')
    .replaceAll("&#x27;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&")
    .replace(/\s+/g, " ");
}

const RAPOR_A = { en: PAGES["report:raporA"].en, tr: PAGES["report:raporA"].tr };
const RAPOR_A_TEXT = { en: visible(RAPOR_A.en), tr: visible(RAPOR_A.tr) };

/* ------------------------------------------------------------------ *
 * Rapor-A, leak by leak
 * ------------------------------------------------------------------ */

test("Rapor-A: the analysis status is `tamamlandı`, not `completed`", () => {
  assert.ok(RAPOR_A.tr.includes('<span lang="tr" data-canonical="completed">tamamlandı</span>'));
  assert.ok(!/\bcompleted\b/.test(RAPOR_A_TEXT.tr), "`completed` is still painted");
  assert.ok(/\bcompleted\b/.test(RAPOR_A_TEXT.en), "the English report keeps the API's word");
});

test("Rapor-A: decision coverage reads `2/2 tam`, not `2/2 complete`", () => {
  assert.ok(RAPOR_A_TEXT.tr.includes("Karar kapsamı: 2/2 tam"));
  assert.ok(!RAPOR_A_TEXT.tr.includes("2/2 complete"));
  assert.ok(RAPOR_A_TEXT.en.includes("Decision coverage: 2/2 complete"));
});

test("Rapor-A: the provenance axes read `doğrulanmamış · okunabildi`, not `UNVERIFIED · AVAILABLE`", () => {
  assert.ok(RAPOR_A_TEXT.tr.includes("doğrulanmamış · okunabildi"));
  assert.ok(!RAPOR_A_TEXT.tr.includes("UNVERIFIED"));
  assert.ok(!/\bAVAILABLE\b/.test(RAPOR_A_TEXT.tr));
  assert.ok(RAPOR_A.tr.includes('data-canonical="UNVERIFIED"') && RAPOR_A.tr.includes('data-canonical="AVAILABLE"'));
  assert.ok(RAPOR_A_TEXT.en.includes("UNVERIFIED · AVAILABLE"));
});

test("Rapor-A: every signal status reads `başarılı`, not `SUCCESS`", () => {
  assert.ok(!/\bSUCCESS\b/.test(RAPOR_A_TEXT.tr));
  // Six detector panels, each with its own status.
  assert.equal(RAPOR_A.tr.split('<span lang="tr" data-canonical="SUCCESS">başarılı</span>').length - 1, 6);
});

test("Rapor-A: the fired rule's meaning is Turkish, not the API's English paragraph", () => {
  assert.ok(!RAPOR_A_TEXT.tr.includes(R9_200_SUMMARY));
  assert.ok(!RAPOR_A_TEXT.tr.includes("Both detectors this ruleset takes a decision from"));
  assert.ok(RAPOR_A_TEXT.tr.includes(CANONICAL.ruleSummary.tr[R9_200_SUMMARY]));
  assert.ok(RAPOR_A.tr.includes(`data-canonical="${escapeHtml(R9_200_SUMMARY)}"`), "the API's sentence is kept as data-canonical");
  assert.ok(RAPOR_A_TEXT.en.includes(R9_200_SUMMARY));
});

test("Rapor-A: a detector's condition is its Turkish sentence alone, not `· threshold_not_reached`", () => {
  for (const raw of ["threshold_not_reached", "threshold_reached"]) {
    assert.ok(!RAPOR_A_TEXT.tr.includes(raw), raw);
    assert.ok(RAPOR_A_TEXT.en.includes(`· ${raw}`), raw);
    assert.ok(RAPOR_A.tr.includes(`<span data-canonical="${raw}"><span lang="tr">`), raw);
  }
  assert.ok(RAPOR_A_TEXT.tr.includes("Eşiğine ulaşmadı"));
});

test("Rapor-A: the enrichment and decision states read in Turkish, not `ENRICHMENT_COMPLETE` and `DECIDED`", () => {
  assert.ok(RAPOR_A.tr.includes('<span lang="tr" data-canonical="ENRICHMENT_COMPLETE">tamamlandı</span>'));
  assert.ok(RAPOR_A.tr.includes('<span lang="tr" data-canonical="DECIDED">karar verildi</span>'));
  assert.ok(!RAPOR_A_TEXT.tr.includes("ENRICHMENT_COMPLETE"));
  assert.ok(!/\bDECIDED\b/.test(RAPOR_A_TEXT.tr));
});

test("Rapor-A: a component's state is its Turkish sentence alone, not `…(completed)`", () => {
  assert.ok(!RAPOR_A_TEXT.tr.includes("(completed)"));
  assert.ok(RAPOR_A.tr.includes('<span data-canonical="completed"><span lang="tr">Okumayla tamamlandı</span></span>'));
  assert.ok(RAPOR_A.en.includes("(<span>completed</span>)"));
});

/**
 * The page's text as the browser paints it under `text-transform: uppercase`: each text node is
 * upper-cased under the nearest `lang` above it, `<html lang>` being the reader's language.
 * Returns the upper-cased runs only, each with the language it was cased in.
 */
function uppercased(html, documentLang) {
  const runs = [];
  const stack = [{ upper: false, lang: documentLang }];
  const VOID = new Set(["br", "img", "input", "meta", "link", "hr", "source", "col", "wbr"]);
  const body = html.replace(/<(style|script)[^>]*>[\s\S]*?<\/\1>/g, "");
  for (const [, close, tag, attrs, text] of body.matchAll(/<(\/?)([a-zA-Z0-9]+)([^>]*)>|([^<]+)/g)) {
    if (text !== undefined) {
      const top = stack[stack.length - 1];
      if (top.upper && text.trim()) runs.push({ text: text.toLocaleUpperCase(top.lang), lang: top.lang, source: text });
      continue;
    }
    if (close) {
      if (!VOID.has(tag)) stack.pop();
      continue;
    }
    if (VOID.has(tag) || attrs.endsWith("/")) continue;
    const top = stack[stack.length - 1];
    const lang = attrs.match(/\blang="([^"]+)"/)?.[1] ?? top.lang;
    const upper = /\bclass="[^"]*\buppercase\b/.test(attrs) || (top.upper && !/\bclass="[^"]*\bnormal-case\b/.test(attrs));
    stack.push({ upper, lang });
  }
  return runs;
}

test("Rapor-A: no Turkish-cased English label — `CLAİM` and `CHECKPOİNT` are gone", () => {
  const turkish = uppercased(RAPOR_A.tr, "tr").map((run) => run.text).join("\n");
  for (const broken of ["CLAİM", "CHECKPOİNT"]) assert.ok(!turkish.includes(broken), broken);
  assert.ok(turkish.includes("İDDİA OLUŞTURUCU"));
  assert.ok(turkish.includes("MODEL KONTROL NOKTASI"));
  const english = uppercased(RAPOR_A.en, "en").map((run) => run.text).join("\n");
  assert.ok(english.includes("CLAIM GENERATOR") && english.includes("CHECKPOINT"));
});

test("Rapor-A: every one of the reviewer's leaks is absent from the Turkish text", () => {
  for (const leak of [
    "2/2 complete",
    "UNVERIFIED · AVAILABLE",
    "SUCCESS",
    R9_200_SUMMARY,
    "threshold_not_reached",
    "ENRICHMENT_COMPLETE",
    "DECIDED",
    "(completed)",
    "CLAİM",
    "CHECKPOİNT",
  ]) {
    assert.ok(!RAPOR_A_TEXT.tr.includes(leak), leak);
  }
  assert.ok(!/\bcompleted\b/.test(RAPOR_A_TEXT.tr), "completed");
});

/* ------------------------------------------------------------------ *
 * The lookup the fix goes through is the strict one
 * ------------------------------------------------------------------ */

test("the new domains keep the own-key lookup: an inherited name or an unknown word comes back raw", () => {
  for (const domain of [
    "signalStatus",
    "coverageStatus",
    "provenanceStatus",
    "provenanceAvailability",
    "enrichmentState",
    "decisionState",
    "ruleSummary",
  ]) {
    for (const raw of ["constructor", "toString", "__proto__", "hasOwnProperty", "MYSTERY", "success", ""]) {
      assert.equal(translateCanonical("tr", domain, raw), raw, `${domain}: ${raw}`);
    }
  }
  assert.equal(translateCanonical("tr", "ruleSummary", `${R9_200_SUMMARY} `), `${R9_200_SUMMARY} `);
});

test("every rule summary the API can send has a Turkish entry, and every entry is one the API sends", () => {
  // `apps/api/app/risk_trace.py` is read, not imported: each `"R…": (…)` entry's adjacent string
  // literals joined, as Python joins them.
  const source = readFileSync(new URL("../../api/app/risk_trace.py", import.meta.url), "utf8");
  const sentences = new Set();
  for (const [, body] of source.matchAll(/^\s+"R[0-9-]+":\s*((?:\(\s*(?:"[^"\n]*"\s*)+\))|"[^"\n]*")/gm)) {
    sentences.add([...body.matchAll(/"([^"\n]*)"/g)].map((match) => match[1]).join(""));
  }
  assert.ok(sentences.size >= 20, `only ${sentences.size} summaries read`);
  assert.deepEqual([...sentences].filter((s) => !Object.hasOwn(CANONICAL.ruleSummary.tr, s)), [], "untranslated");
  assert.deepEqual(Object.keys(CANONICAL.ruleSummary.tr).filter((s) => !sentences.has(s)), [], "stale");
  assert.ok(sentences.has(R9_200_SUMMARY));
});

test("a rule summary's Turkish keeps every rule, ruleset, verdict and study id of its English", () => {
  for (const [english, turkish] of Object.entries(CANONICAL.ruleSummary.tr)) {
    for (const token of english.match(/\b(?:R\d+-T\d+|[A-Z][A-Z_]{3,}|HIGH)\b/g) ?? []) {
      assert.ok(turkish.includes(token), `${token} in: ${english}`);
    }
    // No Fake/Real wording the English does not have.
    for (const stronger of ["sahte", "gerçek olduğu", "kesin"]) {
      assert.ok(!turkish.toLowerCase().includes(stronger) || english.toLowerCase().includes("genuine"), `${stronger} in: ${turkish}`);
    }
  }
});

/* ------------------------------------------------------------------ *
 * The structural audit: every English word in the Turkish copy is classified
 * ------------------------------------------------------------------ */

/*
 * A word the Turkish customer copy shares with the English, on purpose. Three classes:
 *
 * - `technical`: a canonical or technical name — a product, a provider, a model, a standard, a
 *   format, a unit, a risk level the rulesets are written in. Left raw and never translated.
 *   Never drawn under `uppercase` in a Turkish-cased run (see the casing test).
 * - `shared`: a word Turkish spells the same way (`risk`, `model`, `video`, `web`). Turkish; not
 *   a leak.
 * - `browser`: drawn by the browser rather than this application — the file input's "Choose
 *   File" / "No file chosen". None appears in the markup, so none is listed; noted here so the
 *   class is not forgotten in manual QA.
 *
 * Anything else this audit finds is a localization defect.
 */
const CLASSIFIED = {
  // technical — products, providers, models, standards
  InspectRoot: "technical",
  NVIDIA: "technical",
  NVCF: "technical",
  SVD: "technical",
  "EfficientNet-B7": "technical",
  LipForensics: "technical",
  AASIST: "technical",
  C2PA: "technical",
  SDK: "technical",
  Content: "technical", // "Content Credentials", the C2PA standard's own name
  Credentials: "technical",
  ffprobe: "technical",
  demuxer: "technical",
  codec: "technical",
  hash: "technical",
  Bona: "technical", // "Bona fide logit", the AASIST checkpoint's class name
  fide: "technical",
  "anti-spoofing": "technical", // the model family's name, glossed in Turkish beside it
  deepfake: "technical",
  English: "technical", // the language selector names each language in itself
  // technical — formats and units
  "SHA-256": "technical",
  URL: "technical",
  UTC: "technical",
  PDF: "technical",
  MP4: "technical",
  MOV: "technical",
  fps: "technical",
  s: "technical", // seconds, after a duration
  API: "technical",
  // technical — the rulesets' own level and verdict names, written raw in both languages
  HIGH: "technical",
  MANIPULATION_DETECTED: "technical",
  // shared — Turkish spells these the same way
  Risk: "shared",
  risk: "shared",
  Model: "shared",
  model: "shared",
  Video: "shared",
  video: "shared",
  Web: "shared",
  veto: "shared",
  // shared — loanwords Turkish text uses and cases as its own: `MANİFEST` (TDK's spelling) and
  // `LOGİT` are correct Turkish capitals, unlike an untranslated English label such as `CLAİM`.
  manifest: "shared",
  logit: "shared",
};

// A study id (`R4-T1`, `R7-T5`) is a canonical name for a piece of this project's evidence.
const STUDY_ID = /^R\d+-T\d+$/;

// A word starts at a letter that follows no letter, digit or apostrophe: `InspectRoot'a` and
// `0.98'in` are one Turkish word each, and `6d8f0a1c` is one identifier.
const WORD = /(?<![\p{L}\p{N}'’])\p{L}[\p{L}\p{N}_-]*/gu;
const words = (text) => new Set((text.match(WORD) ?? []).filter((word) => /^[A-Za-z_-]+[A-Za-z0-9_-]*$/.test(word)));

/** Every word of every value a record holds, except a word the reader's language names. */
function recordWords() {
  const named = new Set(Object.values(CANONICAL).flatMap((domain) => Object.keys(domain.tr)));
  const out = new Set(["reader", "example", "com", "d1g3st"]);
  const walk = (value) => {
    if (typeof value === "string" && !named.has(value)) for (const word of value.match(WORD) ?? []) out.add(word);
    else if (Array.isArray(value)) value.forEach(walk);
    else if (value && typeof value === "object") Object.values(value).forEach(walk);
  };
  walk(LIST);
  walk(Object.values(FIXTURES));
  // The dashboard shortens an id and a calibration to their first eight characters.
  for (const analysis of LIST) {
    out.add(analysis.id.slice(0, 8));
    if (analysis.risk_calibration_id) out.add(analysis.risk_calibration_id.slice(0, 8));
  }
  return out;
}

test("every English word in the Turkish customer copy is classified (structural audit)", () => {
  const record = recordWords();
  const unclassified = {};
  for (const [name, { en, tr }] of Object.entries(PAGES)) {
    const english = words(visible(en));
    for (const word of words(visible(tr))) {
      if (!english.has(word) || record.has(word) || Object.hasOwn(CLASSIFIED, word) || STUDY_ID.test(word)) continue;
      (unclassified[word] ??= []).push(name);
    }
  }
  assert.deepEqual(Object.keys(unclassified).sort(), [], "English in the Turkish copy that is neither record nor classified");
});

test("no technical word is drawn under Turkish casing on any customer page", () => {
  const technical = new Set(Object.entries(CLASSIFIED).filter(([, kind]) => kind === "technical").map(([word]) => word));
  const broken = [];
  for (const [name, { tr }] of Object.entries(PAGES)) {
    for (const run of uppercased(tr, "tr")) {
      if (run.lang !== "tr") continue;
      for (const word of run.source.match(WORD) ?? []) {
        if (technical.has(word) && word.toLocaleUpperCase("tr") !== word.toLocaleUpperCase("en")) {
          broken.push(`${name}: ${word} → ${word.toLocaleUpperCase("tr")}`);
        }
      }
    }
  }
  assert.deepEqual(broken, []);
});

test("no Turkish tooltip carries a known API word in English (Architect fix, R16-T8)", () => {
  // A `title` is a hover tooltip, so it is presentation: an enum this build names in Turkish must
  // not reach the reader there either. The record's word is kept as `data-canonical` instead.
  const known = new Set(
    Object.values(CANONICAL).flatMap((domain) =>
      Object.entries(domain.tr).filter(([value, name]) => value !== name).map(([value]) => value),
    ),
  );
  const leaks = [];
  for (const [name, { tr }] of Object.entries(PAGES)) {
    for (const [, raw] of tr.matchAll(/\stitle="([^"]*)"/g)) {
      const title = raw.replaceAll("&quot;", '"').replaceAll("&#x27;", "'").replaceAll("&amp;", "&");
      for (const value of known) {
        const at = title.indexOf(value);
        if (at === -1) continue;
        const before = title[at - 1] ?? " ";
        const after = title[at + value.length] ?? " ";
        if (!/[\p{L}\p{N}_]/u.test(before) && !/[\p{L}\p{N}_]/u.test(after)) leaks.push(`${name}: ${value} in "${title}"`);
      }
    }
  }
  assert.deepEqual(leaks, []);
  // The record's word is still on the element, out of the reader's way.
  assert.ok(RAPOR_A.tr.includes('data-canonical="SUCCESS"'));
  assert.ok(!/\stitle="(?:SUCCESS|completed|DECIDED|ENRICHMENT_COMPLETE|threshold_not_reached)"/.test(RAPOR_A.tr));
});
