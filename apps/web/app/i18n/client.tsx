"use client";

/**
 * The browser half of localization (R16-T1): the hook, two leaf renderers and the selector.
 *
 * **No provider, no context, no layout change.** The locale is an external store — the reader's
 * `localStorage` — and `useSyncExternalStore` is React's own way to read one. Every component
 * that calls `useLocale` subscribes to the same module-level store, so a switch in the selector
 * repaints every translated word on the page without a tree-wide wrapper to carry it.
 *
 * **The server always renders the default language**, because the server cannot see
 * `localStorage` and this feature is not allowed to tell it — no cookie, no header, no URL
 * segment. `getServerSnapshot` returns `DEFAULT_LOCALE`, React hydrates against that, and then
 * re-renders with the stored choice. The cost is one frame of English for a Turkish reader on a
 * full load; the alternative is sending the preference to the server, which this task rules out.
 *
 * Translated words are leaves. `AdminSidebar` stays a server component and keeps its route
 * table; it hands `<T k="nav.jobs" />` down where it used to write `Detection jobs`, and only
 * that word is a client island.
 */

import { useEffect, useSyncExternalStore } from "react";

import {
  type CanonicalDomain,
  createLocaleStore,
  DEFAULT_LOCALE,
  isLocale,
  LOCALE_STORAGE_KEY,
  LOCALES,
  type Locale,
  type MessageKey,
  translate,
  translateCanonical,
} from "./core";

const store = createLocaleStore(() =>
  typeof window === "undefined" ? null : window.localStorage,
);

/*
 * Another tab's choice. The browser fires `storage` only in the *other* tabs, so this listener
 * never hears this tab's own writes — those reach subscribers through `store.set`.
 */
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === LOCALE_STORAGE_KEY) {
      store.notify();
    }
  });
}

export function useLocale(): [Locale, (locale: Locale) => void] {
  const locale = useSyncExternalStore(store.subscribe, store.get, () => DEFAULT_LOCALE);
  return [locale, store.set];
}

/** One interface string, in the reader's language. */
export function T({ k }: { k: MessageKey }) {
  const [locale] = useLocale();
  return <>{translate(locale, k)}</>;
}

/**
 * A value the API produced, named in the reader's language — or, when this application has no
 * name for it, shown exactly as the API spelled it. See `translateCanonical`.
 *
 * `title` carries the canonical spelling whenever the painted word differs from it, so a reader
 * looking at `başarısız` can still find out the record says `failed`.
 */
export function CanonicalLabel({ domain, value }: { domain: CanonicalDomain; value: string }) {
  const [locale] = useLocale();
  const label = translateCanonical(locale, domain, value);
  return <span title={label === value ? undefined : value}>{label}</span>;
}

/**
 * The language switch.
 *
 * A native `<select>`, so it is keyboard- and screen-reader-complete without any code here.
 * Changing it writes `localStorage` and repaints; it does not navigate, does not fetch and does
 * not touch the address bar. It also keeps `<html lang>` in step with what is painted, which is
 * the one piece of the document outside React's tree that a language change has to reach.
 */
export function LanguageSelector() {
  const [locale, setLocale] = useLocale();

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  return (
    <label className="flex items-center justify-between gap-2 px-3 text-[12px] text-muted">
      <span>{translate(locale, "locale.selector")}</span>
      <select
        value={locale}
        onChange={(event) => {
          if (isLocale(event.target.value)) {
            setLocale(event.target.value);
          }
        }}
        className="rounded-md border border-line bg-ink px-2 py-1 text-[12px] text-bone"
      >
        {LOCALES.map((option) => (
          <option key={option} value={option}>
            {translate(option, `locale.${option}`)}
          </option>
        ))}
      </select>
    </label>
  );
}
