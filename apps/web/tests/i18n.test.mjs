// R16-T1 — the localization module's contract, run with `node --test` (no bundler, no new
// dependency: Node strips the types from `core.ts` itself).
//
// Three things are pinned here. The store keeps the reader's choice in the browser and nowhere
// else. An interface string falls back to English. And a value the API produced that this
// application has no name for is drawn exactly as it arrived — the rule the task calls the
// canonical fallback, and the one most worth a regression test.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  CANONICAL,
  createLocaleStore,
  DEFAULT_LOCALE,
  isLocale,
  LOCALE_STORAGE_KEY,
  MESSAGES,
  translate,
  translateCanonical,
} from "../app/i18n/core.ts";

/** A `localStorage` stand-in that records every write it is asked to make. */
function memoryStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  const writes = [];
  return {
    data,
    writes,
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => {
      writes.push([key, value]);
      data.set(key, String(value));
    },
  };
}

/* ------------------------------------------------------------------ *
 * Canonical fallback
 * ------------------------------------------------------------------ */

test("a known canonical value is named in the reader's language", () => {
  assert.equal(translateCanonical("tr", "jobStatus", "failed"), "başarısız");
  assert.equal(translateCanonical("tr", "jobStatus", "completed"), "tamamlandı");
  assert.equal(translateCanonical("en", "jobStatus", "failed"), "failed");
});

test("an unknown canonical value is returned exactly as it arrived, in every locale", () => {
  for (const locale of ["en", "tr"]) {
    for (const raw of [
      "UNEXPECTED_STATE",
      "FAILED", // a known word in another case is a different word on the wire
      "Failed",
      " failed",
      "",
      "çok_yeni_durum",
      "constructor", // must not resolve through the prototype chain
      "toString",
      "__proto__",
    ]) {
      assert.equal(translateCanonical(locale, "jobStatus", raw), raw, `${locale}: ${raw}`);
    }
  }
});

test("the fallback never substitutes a generic placeholder", () => {
  const out = translateCanonical("tr", "jobStatus", "UNEXPECTED_STATE");
  assert.doesNotMatch(out, /unknown|bilinmeyen/i);
});

test("an unknown domain or locale degrades to the raw value, not a crash", () => {
  assert.equal(translateCanonical("tr", "noSuchDomain", "failed"), "failed");
  assert.equal(translateCanonical("de", "jobStatus", "failed"), "failed");
});

/* ------------------------------------------------------------------ *
 * Interface strings
 * ------------------------------------------------------------------ */

test("interface strings switch with the locale", () => {
  assert.equal(translate("en", "nav.jobs"), "Detection jobs");
  assert.equal(translate("tr", "nav.jobs"), "Tespit işleri");
});

test("an interface key with no entry falls back to English, then to the key", () => {
  assert.equal(translate("de", "nav.jobs"), "Detection jobs");
  assert.equal(translate("tr", "nav.noSuchKey"), "nav.noSuchKey");
});

/* ------------------------------------------------------------------ *
 * Persistence
 * ------------------------------------------------------------------ */

test("the default is the product's existing language", () => {
  assert.equal(DEFAULT_LOCALE, "en");
  assert.equal(createLocaleStore(() => memoryStorage()).get(), "en");
});

test("a switch is persisted in browser storage, under one key, and read back", () => {
  const storage = memoryStorage();
  const store = createLocaleStore(() => storage);

  store.set("tr");
  assert.equal(store.get(), "tr");
  assert.deepEqual(storage.writes, [[LOCALE_STORAGE_KEY, "tr"]]);

  // A fresh store over the same storage is a page reload: the choice survives it.
  assert.equal(createLocaleStore(() => storage).get(), "tr");

  store.set("en");
  assert.equal(store.get(), "en");
  assert.deepEqual([...storage.data.keys()], [LOCALE_STORAGE_KEY]);
});

test("subscribers hear a switch, and stop hearing after unsubscribing", () => {
  const store = createLocaleStore(() => memoryStorage());
  let calls = 0;
  const unsubscribe = store.subscribe(() => calls++);
  store.set("tr");
  assert.equal(calls, 1);
  unsubscribe();
  store.set("en");
  assert.equal(calls, 1);
});

test("a stored value that is not a supported locale reads as the default", () => {
  for (const junk of ["de", "TR", "", "null"]) {
    const store = createLocaleStore(() => memoryStorage({ [LOCALE_STORAGE_KEY]: junk }));
    assert.equal(store.get(), "en", junk);
  }
});

test("an unsupported locale is never written", () => {
  const storage = memoryStorage();
  createLocaleStore(() => storage).set("de");
  assert.deepEqual(storage.writes, []);
});

test("storage that throws, or does not exist, leaves the reader in English", () => {
  const throwing = {
    getItem() {
      throw new Error("SecurityError");
    },
    setItem() {
      throw new Error("QuotaExceededError");
    },
  };
  const store = createLocaleStore(() => throwing);
  assert.doesNotThrow(() => store.set("tr"));
  assert.equal(store.get(), "en");

  const absent = createLocaleStore(() => null);
  assert.equal(absent.get(), "en");
  assert.doesNotThrow(() => absent.set("tr"));
  assert.equal(
    createLocaleStore(() => {
      throw new Error("localStorage is not available");
    }).get(),
    "en",
  );
});

test("isLocale accepts exactly the supported locales", () => {
  assert.ok(isLocale("en"));
  assert.ok(isLocale("tr"));
  for (const v of ["EN", "de", "", null, undefined, 1]) {
    assert.equal(isLocale(v), false, String(v));
  }
});

/* ------------------------------------------------------------------ *
 * No payload, URL or server involvement
 * ------------------------------------------------------------------ */

test("the localization module reaches no network, cookie, header or URL", () => {
  const sources = ["../app/i18n/core.ts", "../app/i18n/client.tsx"].map((path) =>
    readFileSync(new URL(path, import.meta.url), "utf8")
      // Comments describe what the module does not do; only code is checked.
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/^\s*\/\/.*$/gm, ""),
  );
  for (const source of sources) {
    for (const forbidden of [
      /\bfetch\s*\(/,
      /document\.cookie/,
      /\bcookies\s*\(/,
      /\bheaders\s*\(/,
      /\buseRouter\b/,
      /\bredirect\s*\(/,
      /history\.(push|replace)State/,
      /location\.(href|assign|replace|search)/,
      /sessionStorage/,
    ]) {
      assert.doesNotMatch(source, forbidden);
    }
  }
});

/* ------------------------------------------------------------------ *
 * R16-T2 — the interface around the evidence
 * ------------------------------------------------------------------ */

/** A source file as code only, so a sentence quoted in a comment does not count as written. */
function code(path) {
  return readFileSync(new URL(path, import.meta.url), "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");
}

test("every interface string has a Turkish entry, and Turkish has no key English lacks", () => {
  const en = Object.keys(MESSAGES.en).sort();
  const tr = Object.keys(MESSAGES.tr).sort();
  assert.deepEqual(tr, en);
  for (const locale of ["en", "tr"]) {
    for (const [key, value] of Object.entries(MESSAGES[locale])) {
      assert.ok(value.trim().length > 0, `${locale}: ${key} is empty`);
    }
  }
});

test("every canonical domain names the same values in both languages", () => {
  for (const [domain, tables] of Object.entries(CANONICAL)) {
    // `webMessage` is the exception by design: English is the source spelling, so its
    // English table is empty and every sentence reaches the page as it was written.
    if (domain === "webMessage") {
      assert.deepEqual(tables.en, {});
      continue;
    }
    assert.deepEqual(Object.keys(tables.tr).sort(), Object.keys(tables.en).sort(), domain);
  }
});

test("each canonical table's English is the identity, except for the feedback labels", () => {
  for (const [domain, tables] of Object.entries(CANONICAL)) {
    if (domain.startsWith("feedback")) {
      continue;
    }
    for (const [value, label] of Object.entries(tables.en)) {
      assert.equal(label, value, `${domain}: ${value}`);
    }
  }
});

test("the feedback labels' English is the wording user-feedback.ts already uses", () => {
  const source = code("../app/user-feedback.ts");
  for (const domain of ["feedbackAssessment", "feedbackClaimedLabel"]) {
    for (const [value, label] of Object.entries(CANONICAL[domain].en)) {
      assert.ok(
        source.includes(`${value}: "${label}"`),
        `${domain}: ${value} is not "${label}" in user-feedback.ts`,
      );
    }
  }
});

test("every translated web sentence is still written, word for word, by the code that sends it", () => {
  // A sentence edited in a route handler and not here would silently stop translating; this
  // makes that drift a failing test instead.
  const sources = [
    "../app/submit/route.ts",
    "../app/submit-feedback/route.ts",
    "../app/analysis.ts",
  ]
    .map(code)
    .join("\n");
  for (const sentence of Object.keys(CANONICAL.webMessage.tr)) {
    assert.ok(sources.includes(`"${sentence}"`), `no longer written anywhere: ${sentence}`);
  }
});

test("the API's own refusal text, and sentences carrying a status code, reach the page untouched", () => {
  for (const raw of [
    "Unsupported media type.",
    "The API refused the submission (HTTP 413).",
    "The feedback was refused (HTTP 500).",
    "the api could not be reached.", // a known sentence in another case is a different sentence
  ]) {
    assert.equal(translateCanonical("tr", "webMessage", raw), raw);
    assert.equal(translateCanonical("en", "webMessage", raw), raw);
  }
  assert.equal(
    translateCanonical("tr", "webMessage", "The API could not be reached."),
    "API'ye ulaşılamadı.",
  );
});

test("an unknown analysis status, role or health word is shown exactly as the API spelled it", () => {
  assert.equal(translateCanonical("tr", "analysisStatus", "completed"), "tamamlandı");
  assert.equal(translateCanonical("tr", "analysisStatus", "archived"), "archived");
  assert.equal(translateCanonical("tr", "userRole", "ADMIN"), "YÖNETİCİ");
  assert.equal(translateCanonical("tr", "userRole", "AUDITOR"), "AUDITOR");
  assert.equal(translateCanonical("tr", "userRole", "admin"), "admin");
  assert.equal(translateCanonical("tr", "healthState", "ok"), "tamam");
  assert.equal(translateCanonical("tr", "healthState", "rebooting"), "rebooting");
});
