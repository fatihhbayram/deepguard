// R16-T3 — the forensic report in two languages, and the evidence in one.
//
// The report route itself is rendered here: the real page, the real payload parser and the real
// `<Copy>` leaves, fed an API response through a replaced `fetch`. Each fixture is rendered with
// the language fixed to English and then to Turkish — the switch a reader makes with the selector
// — and the two documents are compared. What has to change is the words; what may not change by
// a single byte is anything the record holds.
//
// Run by `npm test` (`node --test`). `./support/tsx-loader.mjs` transpiles the TypeScript with the
// application's own compiler and stands in for `next/headers`; nothing else is replaced.

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./support/tsx-loader.mjs", import.meta.url);

const { createElement } = await import("react");
const { renderToStaticMarkup } = await import("react-dom/server");
const { default: Report } = await import("../app/app/report/[id]/page.tsx");
const { FixedLocale } = await import("../app/i18n/client.tsx");
const { REPORT_COPY, copySegments, translateCopy } = await import("../app/i18n/core.ts");
const { FIXTURES, canonicalValues } = await import("./support/report-fixtures.mjs");
const { reportPageCopy, reportVocabularyCopy } = await import("./support/report-copy-sources.mjs");

process.env.API_INTERNAL_URL = "http://api.invalid";

/** The report for one payload, rendered in one language. */
async function render(payload, locale) {
  globalThis.fetch = async (url) =>
    String(url).includes("/feedback")
      ? new Response("{}", { status: 404 })
      : new Response(JSON.stringify(payload), { status: 200 });

  const page = await Report({
    params: Promise.resolve({ id: payload.id }),
    searchParams: Promise.resolve({}),
  });
  return renderToStaticMarkup(createElement(FixedLocale, { locale }, page));
}

/** A value as the markup spells it. */
function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#x27;");
}

function occurrences(haystack, needle) {
  let count = 0;
  for (let at = haystack.indexOf(needle); at !== -1; at = haystack.indexOf(needle, at + 1)) {
    count += 1;
  }
  return count;
}

/**
 * The record's values in the order the document prints them.
 *
 * React draws a value from the record as a text node of its own — a field, a table cell, a slot
 * in a sentence (`renderToStaticMarkup` separates adjacent text with `<!-- -->`) — so the evidence
 * is every text node that is exactly one of those values. That is what lets `processing` the
 * state be told apart from "processing" the English word.
 */
function evidenceSequence(html, values) {
  const escaped = new Set([...values].map(escapeHtml));
  return html
    .split(/<[^>]*>/)
    .filter((node) => escaped.has(node));
}

const RENDERS = {};
for (const [name, payload] of Object.entries(FIXTURES)) {
  RENDERS[name] = { en: await render(payload, "en"), tr: await render(payload, "tr") };
}

/* ------------------------------------------------------------------ *
 * The evidence does not move
 * ------------------------------------------------------------------ */

for (const [name, payload] of Object.entries(FIXTURES)) {
  test(`${name}: every value from the record is printed identically in English and Turkish`, () => {
    const { en, tr } = RENDERS[name];

    assert.ok(!en.includes("Report unavailable"), "the fixture must parse as an analysis");

    const values = canonicalValues(payload);
    const english = evidenceSequence(en, values);

    // The identity of the evidence is on every report, whatever else the fixture leaves out.
    for (const value of [payload.id, payload.created_at, payload.original_sha256]) {
      assert.ok(english.includes(escapeHtml(value)), `${value} is not printed`);
    }

    // The same values, the same number of times, in the same order: the Turkish document is the
    // English one with its words replaced.
    assert.deepEqual(evidenceSequence(tr, values), english);
  });

  test(`${name}: switching language changes the report's words`, () => {
    const { en, tr } = RENDERS[name];
    assert.notEqual(tr, en);
    assert.ok(en.includes("Forensic Evidence Report"));
    assert.ok(tr.includes("Adli Kanıt Raporu"));
    assert.ok(!tr.includes("Forensic Evidence Report"));
  });
}

test("the hash is printed whole, in both languages, beside its own label", () => {
  const sha = FIXTURES.v5Processing.original_sha256;
  assert.match(RENDERS.v5Processing.en, new RegExp(`SHA-256 of the submitted media.*?${sha}`, "s"));
  assert.match(RENDERS.v5Processing.tr, new RegExp(`Gönderilen medyanın SHA-256 değeri.*?${sha}`, "s"));
});

test("a host is drawn into the Turkish sentence exactly as recorded", () => {
  const host = FIXTURES.v5Processing.source_host;
  assert.ok(RENDERS.v5Processing.tr.includes(`Medya ${host} adresinden, sunulan tek bir dosya olarak edinildi.`));
  assert.ok(RENDERS.v5Processing.en.includes(`Media was acquired from ${host} as a single served file.`));
});

test("the API's own rule summary is carried verbatim and not translated", () => {
  const summary = escapeHtml(FIXTURES.v5Processing.risk_trace.rule_summary);
  assert.equal(occurrences(RENDERS.v5Processing.tr, summary), 1);
});

/* ------------------------------------------------------------------ *
 * Unknown values stay raw
 * ------------------------------------------------------------------ */

test("a value this build has no name for is drawn exactly as the API sent it, in Turkish too", () => {
  const { en, tr } = RENDERS.unknownValues;
  for (const raw of [
    "MYSTERY_LEVEL",
    "x9-v9.9.9",
    "RX-404",
    "mystery_signal",
    "mystery_condition",
    "mystery_role",
    "mystery_reason",
    "MYSTERY_DECISION_STATE",
    "ENRICHMENT_MYSTERY",
    "mystery_state",
    "mystery-provider/mystery_component",
    "MYSTERY_STATUS",
  ]) {
    assert.ok(en.includes(raw), `${raw} missing from the English report`);
    assert.equal(occurrences(tr, raw), occurrences(en, raw), raw);
  }
  // The record's own word, never a placeholder in either language.
  for (const placeholder of ["Bilinmeyen", "Unknown value", "undefined", "null"]) {
    assert.ok(!tr.includes(`>${placeholder}<`), placeholder);
  }
  // A role, a detector or a reason this build does not know is not dressed as an English
  // sentence in a Turkish document: it claims no language at all.
  for (const raw of ["mystery_role", "mystery_signal", "mystery_reason"]) {
    assert.ok(tr.includes(`<span>${raw}</span>`), raw);
  }
});

/* ------------------------------------------------------------------ *
 * Nothing is left untranslated, and nothing is translated twice
 * ------------------------------------------------------------------ */

test("the Turkish report contains no English copy (every known fixture)", () => {
  for (const name of Object.keys(FIXTURES)) {
    if (name === "unknownValues") continue;
    // The product's name is marked English on purpose; nothing else may be.
    assert.equal(occurrences(RENDERS[name].tr, 'lang="en"'), 1, name);
    assert.ok(RENDERS[name].tr.includes('<span lang="en" translate="no">InspectRoot</span>'), name);
  }
});

test("every sentence the report can print has a Turkish entry, and every entry is still used", async () => {
  const sources = new Set([...reportPageCopy(), ...(await reportVocabularyCopy())]);
  const entries = new Set(Object.keys(REPORT_COPY.tr));

  const missing = [...sources].filter((sentence) => !entries.has(sentence));
  assert.deepEqual(missing, [], "sentences with no Turkish entry");

  const stale = [...entries].filter((sentence) => !sources.has(sentence));
  assert.deepEqual(stale, [], "Turkish entries for sentences the report no longer prints");
});

test("a translation keeps exactly the slots its English has", () => {
  const slots = (text) =>
    copySegments(text)
      .filter((segment) => typeof segment !== "string")
      .map((segment) => segment.slot)
      .sort();
  for (const [english, turkish] of Object.entries(REPORT_COPY.tr)) {
    assert.deepEqual(slots(turkish), slots(english), english);
  }
});

test("no Turkish entry renames the product or the evidence", () => {
  for (const [english, turkish] of Object.entries(REPORT_COPY.tr)) {
    assert.ok(!turkish.includes("DeepGuard"), english);
    // A figure, a rule or a ruleset written into the English is written into the Turkish.
    for (const token of english.match(/\b(?:\d+(?:\.\d+)?%?|R\d-T\d|p7-v1\.0\.0|r7-v4\.0\.0|HIGH|C2PA|NVIDIA|AASIST|NVCF|SHA-256)\b/g) ?? []) {
      assert.ok(turkish.includes(token.replace(/^(\d+(?:\.\d+)?)%$/, "%$1")), `${token} in: ${english}`);
    }
  }
});

/* ------------------------------------------------------------------ *
 * The lookup itself
 * ------------------------------------------------------------------ */

test("English is the key and is returned unchanged", () => {
  assert.deepEqual(translateCopy("en", "Decision breakdown"), { text: "Decision breakdown", lang: "en" });
  assert.deepEqual(translateCopy("tr", "Decision breakdown"), { text: "Karar dökümü", lang: "tr" });
});

test("a sentence nobody translated is drawn in English and says so", () => {
  assert.deepEqual(translateCopy("tr", "A sentence with no entry."), {
    text: "A sentence with no entry.",
    lang: "en",
  });
  assert.deepEqual(translateCopy("tr", "constructor"), { text: "constructor", lang: "en" });
});

test("a template is cut at its slots and nowhere else", () => {
  assert.deepEqual(copySegments("Decision coverage: {usable}/{total} {status}"), [
    "Decision coverage: ",
    { slot: "usable" },
    "/",
    { slot: "total" },
    " ",
    { slot: "status" },
  ]);
  assert.deepEqual(copySegments("no slots {1} here { x }"), ["no slots {1} here { x }"]);
});
