// R16-T5 — the dashboard in two languages, and the evidence in one.
//
// The dashboard route itself is rendered, as `report-i18n.test.mjs` renders the report: the real
// page fed the R16-T3 fixtures as its analysis list through a replaced `fetch`, once with the
// language fixed to English and once to Turkish. The queue row, the evidence drawer and the
// methodology notes are all in the markup (a closed `<details>` is still rendered), so the two
// documents can be compared whole.

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./support/tsx-loader.mjs", import.meta.url);

const { createElement } = await import("react");
const { renderToStaticMarkup } = await import("react-dom/server");
const { default: Dashboard } = await import("../app/app/page.tsx");
const { FixedLocale } = await import("../app/i18n/client.tsx");
const { CANONICAL, REPORT_COPY } = await import("../app/i18n/core.ts");
const { FIXTURES } = await import("./support/report-fixtures.mjs");
const { dashboardPageCopy, dashboardVocabularyCopy } = await import("./support/report-copy-sources.mjs");
const { valueTokens } = await import("./support/value-tokens.mjs");

process.env.API_INTERNAL_URL = "http://api.invalid";

// One list holding every fixture, each under its own id so the rows are distinct.
const LIST = Object.values(FIXTURES).map((payload, index) => ({
  ...payload,
  id: `${String(index).padStart(8, "0")}-${payload.id.slice(9)}`,
}));

async function render(locale) {
  globalThis.fetch = async (url) => {
    const path = new URL(String(url)).pathname;
    if (path === "/health") {
      return Response.json({ status: "ok", database: "ok" });
    }
    if (path === "/api/v1/auth/me") {
      return Response.json({ id: "u", email: "reader@example.com", role: "USER" });
    }
    if (path === "/api/v1/analyses") {
      return Response.json(LIST);
    }
    return new Response("{}", { status: 404 });
  };
  const page = await Dashboard({ searchParams: Promise.resolve({}) });
  return renderToStaticMarkup(createElement(FixedLocale, { locale }, page));
}

const escapeHtml = (value) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#x27;");

/** Every string and number in the list, and each way the dashboard prints a number. */
function recordValues() {
  const values = new Set();
  const walk = (value) => {
    if (typeof value === "string" && value.length > 0) values.add(value);
    else if (typeof value === "number") {
      for (const shown of [String(value), value.toFixed(2), value.toFixed(4), `${(value * 100).toFixed(2)}%`]) {
        values.add(shown);
      }
    } else if (Array.isArray(value)) value.forEach(walk);
    else if (value && typeof value === "object") Object.values(value).forEach(walk);
  };
  walk(LIST);
  for (const analysis of LIST) {
    values.add(analysis.id.slice(0, 8));
    if (analysis.risk_calibration_id) values.add(`${analysis.risk_calibration_id.slice(0, 8)}…`);
  }
  return new Set([...values].map(escapeHtml));
}

const EN = await render("en");
const TR = await render("tr");

// The canonical words R16-T1/T2 already name in the reader's language through `<CanonicalLabel>`
// — a status chip, a health state, a role — keeping the API's own word as the tooltip. Those are
// checked as chips below; everything else from the record must be the same text in both.
const CHIP_WORDS = new Set(
  Object.values(CANONICAL).flatMap((domain) =>
    Object.entries(domain.tr)
      .filter(([value, name]) => value !== name)
      .map(([value]) => value),
  ),
);
const VALUES = new Set([...recordValues()].filter((value) => !CHIP_WORDS.has(value)));
const nodes = (html) => html.split(/<[^>]*>/).filter((node) => node.length > 0);

test("the dashboard renders every row in both languages", () => {
  for (const analysis of LIST) {
    assert.ok(EN.includes(analysis.id.slice(0, 8)), analysis.id);
    assert.ok(TR.includes(analysis.id.slice(0, 8)), analysis.id);
  }
  assert.notEqual(TR, EN);
  assert.ok(TR.includes("Bu sonuçlar nasıl yorumlanmalı"));
  assert.ok(!TR.includes("How to interpret these results"));
});

test("every value from the records is printed identically in English and Turkish", () => {
  const english = nodes(EN).filter((node) => VALUES.has(node));
  assert.ok(english.length > 80, `only ${english.length} values found`);
  assert.deepEqual(nodes(TR).filter((node) => VALUES.has(node)), english);
});

test("every record value inside a sentence is printed identically, in the same order", () => {
  const english = valueTokens(EN, VALUES);
  assert.ok(english.length > nodes(EN).filter((node) => VALUES.has(node)).length, "found nothing in sentences");
  assert.deepEqual(valueTokens(TR, VALUES), english);
});

test("every record value inside a tooltip is carried into the Turkish tooltip unchanged", () => {
  const titles = (html) => [...html.matchAll(/title="([^"]*)"/g)].map((match) => match[1]);
  // A translated chip keeps the API's own word on hover; that title exists only where the painted
  // word differs, which is the Turkish document.
  const [en, tr] = [titles(EN), titles(TR).filter((title) => !CHIP_WORDS.has(title))];
  assert.equal(tr.length, en.length);
  for (const value of VALUES) {
    if (value.length < 3) continue;
    const count = (list) => list.filter((title) => title.includes(value)).length;
    assert.equal(count(tr), count(en), value);
  }
});

test("a translated chip names the canonical word it stands for, and keeps that word exactly", () => {
  const chips = [...TR.matchAll(/<span lang="tr" title="([^"]*)">([^<]*)<\/span>/g)];
  assert.ok(chips.length >= LIST.length, "every row's status is a chip");
  for (const [, value, name] of chips) {
    const domain = Object.values(CANONICAL).find((table) => Object.hasOwn(table.tr, value));
    assert.ok(domain, value);
    assert.equal(name, domain.tr[value], value);
  }
  // And the status every row holds is the one its chip carries.
  const statuses = chips.map(([, value]) => value).filter((value) => value in CANONICAL.analysisStatus.tr);
  assert.deepEqual(statuses, LIST.map((analysis) => analysis.status));
});

test("the Turkish dashboard draws no English copy", () => {
  // Only the product's name is marked English; an unknown value claims no language at all.
  const english = [...TR.matchAll(/<(\w+) lang="en"[^>]*>(.*?)<\/\1>/g)].map((match) => match[2]);
  assert.deepEqual([...new Set(english)].filter((text) => text !== "InspectRoot"), []);
  for (const html of [EN, TR]) {
    assert.ok(!/\{[A-Za-z0-9_]+\}/.test(nodes(html).join("\n")), "an unfilled slot");
  }
});

test("every sentence the dashboard can print has a Turkish entry", async () => {
  const sources = new Set([...dashboardPageCopy(), ...(await dashboardVocabularyCopy())]);
  const entries = new Set(Object.keys(REPORT_COPY.tr));
  assert.deepEqual([...sources].filter((sentence) => !entries.has(sentence)), []);
});

test("the Risk note keeps the verdicts it explains as the table's own words", () => {
  for (const [english, turkish] of [
    ["Manipulation detected", "Manipülasyon tespit edildi"],
    ["Inconclusive", "Sonuçsuz"],
    ["High risk", "Yüksek risk"],
  ]) {
    assert.ok(EN.includes(`<span lang="en">${english}</span>`), english);
    assert.ok(TR.includes(`<span lang="tr">${turkish}</span>`), turkish);
  }
});

/* ------------------------------------------------------------------ *
 * The methodology describes the rulesets as they are (R16-T5, F1)
 * ------------------------------------------------------------------ */

/** One methodology note's text, found by its term. */
function note(html, term) {
  // The methodology panel only: the evidence drawer below labels its fields with the same words.
  const panel = html.slice(html.indexOf(html === TR ? "Bu sonuçlar nasıl yorumlanmalı" : "How to interpret these results"));
  html = panel.slice(0, panel.indexOf("</details>"));
  const start = html.indexOf(`<span lang="${panel.includes("Bu sonuçlar") ? "tr" : "en"}">${term}</span></dt>`);
  assert.ok(start !== -1, term);
  const end = html.indexOf("</dd>", start);
  return html.slice(start, end).replace(/<[^>]*>/g, "");
}

test("the face-manipulation note says it is a calibrated deciding detector under the current rulesets", () => {
  const en = note(EN, "Face manipulation");
  const tr = note(TR, "Yüz manipülasyonu");
  for (const ruleset of ["r4-v2.0.0", "r5-v3.0.0", "r7-v4.0.0", "r9-v5.0.0"]) {
    assert.ok(en.includes(ruleset) && tr.includes(ruleset), ruleset);
  }
  assert.ok(en.includes("one of the deciding detectors") && en.includes("R4-T1"));
  assert.ok(tr.includes("karar veren dedektörlerden biridir") && tr.includes("R4-T1"));
  // The wording R16-T5 removed: it told the reader this detector never reaches the verdict.
  for (const stale of ["uncalibrated", "does not affect the risk classification", "no threshold is applied"]) {
    assert.ok(!en.includes(stale), stale);
  }
  for (const stale of ["kalibre edilmemiştir", "risk sınıflandırmasını etkilemez", "hiçbir eşik uygulanmaz"]) {
    assert.ok(!tr.includes(stale), stale);
  }
});

test("the mouth-dynamics note says it was calibrated, and is evidence only under the current rulesets", () => {
  const en = note(EN, "Mouth dynamics");
  const tr = note(TR, "Ağız dinamiği");
  for (const text of [en, tr]) {
    for (const token of ["R5-T3", "r5-v3.0.0", "R7-T6", "r7-v4.0.0", "r9-v5.0.0"]) assert.ok(text.includes(token), token);
  }
  assert.ok(en.includes("evidence only") && tr.includes("yalnızca kanıttır"));
  for (const stale of ["uncalibrated", "does not affect the risk classification"]) assert.ok(!en.includes(stale), stale);
  for (const stale of ["kalibre edilmemiştir", "risk sınıflandırmasını etkilemez"]) assert.ok(!tr.includes(stale), stale);
});

test("no part of the dashboard calls a detector uncalibrated", () => {
  assert.ok(!/uncalibrated/i.test(EN));
  assert.ok(!TR.includes("kalibre edilmemiştir"));
});
