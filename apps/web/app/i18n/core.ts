/**
 * The presentation layer's language, and nothing below it (R16-T1).
 *
 * **Translation happens at the last step before a word is drawn, and nowhere else.** The API
 * speaks one language — its own canonical spellings, `completed`, `FAILED`, `UNREVIEWED` — and
 * every value that crosses the wire, every value written to the database and every value the
 * risk engine reads stays in that spelling. A locale is a property of the reader's browser. It
 * is not in the URL, not in a cookie, not in a request header and not in any payload; switching
 * it changes what is painted and changes nothing that is sent, stored or decided.
 *
 * Two kinds of lookup, and they fail differently on purpose:
 *
 * - `translate()` is for this application's own interface strings — a navigation label, the
 *   selector's name. Those keys are written in this repository, so a missing Turkish entry
 *   falls back to the English one, and only a key nobody defined falls back to itself.
 * - `translateCanonical()` is for a value the API produced. A value this file has no entry for
 *   is returned **exactly as it arrived** — not `Unknown`, not `Bilinmeyen`, not title-cased.
 *   A status the application does not recognise is the case a reader most needs to see as it
 *   is, and a friendly placeholder would hide exactly that.
 *
 * No imports, no React and no `window`: this file is the part that can be tested by
 * `node --test` without a bundler. The browser half — the hook, the selector — is in
 * `./client.tsx`, and it reaches `localStorage` only through `createLocaleStore` below.
 */

export const LOCALES = ["en", "tr"] as const;

export type Locale = (typeof LOCALES)[number];

/** The product's existing language. What the server renders, and what a new reader sees. */
export const DEFAULT_LOCALE: Locale = "en";

/** The one `localStorage` key this feature owns. */
export const LOCALE_STORAGE_KEY = "inspectroot.locale";

export function isLocale(value: unknown): value is Locale {
  return typeof value === "string" && (LOCALES as readonly string[]).includes(value);
}

/* ------------------------------------------------------------------ *
 * Dictionaries
 * ------------------------------------------------------------------ */

/*
 * Deliberately small. R16-T1 proves the mechanism on the admin rail, the selector and the job
 * status chips; the rest of the interface and the report belong to R16-T2 and R16-T3.
 */
const MESSAGES = {
  en: {
    "nav.label": "Administration",
    "nav.group.operations": "Operations",
    "nav.group.governance": "Governance",
    "nav.group.access": "Access",
    "nav.accounts": "Accounts",
    "nav.jobs": "Detection jobs",
    "nav.reviewQueue": "Review queue",
    "nav.analytics": "Operational summary",
    "nav.audit": "Audit log",
    "nav.apiKeys": "API keys",
    "nav.backToWorkspace": "Back to workspace",
    "nav.menu": "Menu",
    "nav.close": "Close",
    "nav.closeNavigation": "Close navigation",
    "locale.selector": "Language",
    "locale.en": "English",
    "locale.tr": "Türkçe",
  },
  tr: {
    "nav.label": "Yönetim",
    "nav.group.operations": "Operasyon",
    "nav.group.governance": "Yönetişim",
    "nav.group.access": "Erişim",
    "nav.accounts": "Hesaplar",
    "nav.jobs": "Tespit işleri",
    "nav.reviewQueue": "İnceleme kuyruğu",
    "nav.analytics": "Operasyon özeti",
    "nav.audit": "Denetim kaydı",
    "nav.apiKeys": "API anahtarları",
    "nav.backToWorkspace": "Çalışma alanına dön",
    "nav.menu": "Menü",
    "nav.close": "Kapat",
    "nav.closeNavigation": "Gezinmeyi kapat",
    "locale.selector": "Dil",
    "locale.en": "English",
    "locale.tr": "Türkçe",
  },
} as const satisfies Record<Locale, Record<string, string>>;

export type MessageKey = keyof (typeof MESSAGES)["en"];

/*
 * Display names for values the API owns, by domain. The keys are the API's canonical spellings
 * and are matched exactly — `completed` and `COMPLETED` are two different words to this table,
 * because they are two different words on the wire. English is listed too, as the identity, so
 * that a domain's known values are written down in one place; it is never a rewrite.
 */
const CANONICAL = {
  // `app/db/models.py` — the four values `jobs.status` can hold.
  jobStatus: {
    en: { queued: "queued", processing: "processing", completed: "completed", failed: "failed" },
    tr: { queued: "kuyrukta", processing: "işleniyor", completed: "tamamlandı", failed: "başarısız" },
  },
} as const satisfies Record<string, Record<Locale, Record<string, string>>>;

export type CanonicalDomain = keyof typeof CANONICAL;

/* ------------------------------------------------------------------ *
 * Lookups
 * ------------------------------------------------------------------ */

/** An interface string. Falls back to English, then to the key itself. */
export function translate(locale: Locale, key: MessageKey): string {
  const table: Record<string, string> = MESSAGES[locale] ?? MESSAGES[DEFAULT_LOCALE];
  return table[key] ?? (MESSAGES[DEFAULT_LOCALE] as Record<string, string>)[key] ?? key;
}

/**
 * A value the API produced, as the reader's locale names it.
 *
 * **An unrecognised value is returned unchanged, byte for byte.** No `Unknown`, no English
 * fallback that might itself be a guess, no case folding. `Object.hasOwn` rather than `in` or
 * a plain index, so a value spelled like an inherited property (`constructor`, `toString`) is
 * treated as the unknown word it is rather than resolving to a function.
 */
export function translateCanonical(
  locale: Locale,
  domain: CanonicalDomain,
  value: string,
): string {
  const table: Record<string, string> | undefined = CANONICAL[domain]?.[locale];
  if (table !== undefined && Object.hasOwn(table, value)) {
    return table[value];
  }
  return value;
}

/* ------------------------------------------------------------------ *
 * Persistence
 * ------------------------------------------------------------------ */

/** The slice of `Storage` this feature uses, so a test can hand it a plain object. */
export type LocaleStorage = Pick<Storage, "getItem" | "setItem">;

/**
 * The reader's choice, kept in their browser and nowhere else.
 *
 * `storage` is a getter rather than a value because `localStorage` itself can throw on access —
 * a private window, blocked site data — and so can every read and write. Each of those is
 * caught and read as "no stored choice": the worst a broken store can do is leave the reader in
 * English, never break the page.
 *
 * `subscribe` is the `useSyncExternalStore` contract. Listeners hear a change made in this tab
 * through `set`, and `notify` exists for the browser's `storage` event, which is how another
 * tab's choice reaches this one.
 */
export function createLocaleStore(storage: () => LocaleStorage | null) {
  const listeners = new Set<() => void>();

  function get(): Locale {
    try {
      const stored = storage()?.getItem(LOCALE_STORAGE_KEY);
      return isLocale(stored) ? stored : DEFAULT_LOCALE;
    } catch {
      return DEFAULT_LOCALE;
    }
  }

  function notify(): void {
    for (const listener of listeners) {
      listener();
    }
  }

  function set(locale: Locale): void {
    if (!isLocale(locale)) {
      return;
    }
    try {
      storage()?.setItem(LOCALE_STORAGE_KEY, locale);
    } catch {
      // Not persisted. `get` reads the store, so an unwritable store cannot hold a choice and
      // the reader stays in the default language. There is no second place to keep it.
    }
    notify();
  }

  function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }

  return { get, set, subscribe, notify };
}
