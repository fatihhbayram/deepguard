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
  createLocaleStore,
  DEFAULT_LOCALE,
  isLocale,
  LOCALE_STORAGE_KEY,
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
