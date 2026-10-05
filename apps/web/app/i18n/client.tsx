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

import {
  createContext,
  Fragment,
  type ReactNode,
  useContext,
  useEffect,
  useSyncExternalStore,
} from "react";

import {
  type CanonicalDomain,
  copySegments,
  createLocaleStore,
  DEFAULT_LOCALE,
  isLocale,
  LOCALE_STORAGE_KEY,
  LOCALES,
  type Locale,
  type MessageKey,
  translate,
  translateCanonical,
  translateCopy,
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

/*
 * A language fixed for everything rendered inside it, whatever the reader chose (R16-T3).
 *
 * The page never sets one: in the browser every word follows the reader's stored choice. It
 * exists because that choice lives in `localStorage`, which a render outside a browser cannot
 * reach — the server always paints the default language — and the report has to be renderable
 * in each language, deterministically, to prove that switching changes its words and none of
 * its evidence (`tests/report-i18n.test.mjs`).
 */
const FixedLocaleContext = createContext<Locale | null>(null);

export function FixedLocale({ locale, children }: { locale: Locale; children: ReactNode }) {
  return <FixedLocaleContext.Provider value={locale}>{children}</FixedLocaleContext.Provider>;
}

export function useLocale(): [Locale, (locale: Locale) => void] {
  const fixed = useContext(FixedLocaleContext);
  const locale = useSyncExternalStore(store.subscribe, store.get, () => DEFAULT_LOCALE);
  return [fixed ?? locale, store.set];
}

/**
 * One interface string, in the reader's language.
 *
 * It says which language it is in. Parts of the application are still English whatever the
 * reader chose, and some are marked `lang="en"` while `<html lang>` follows the reader; a word
 * of this interface drawn inside one of those regions has to name its own language, or the
 * browser would apply the region's rules to it — case mapping under `uppercase` first among them.
 */
export function T({ k }: { k: MessageKey }) {
  const [locale] = useLocale();
  return <span lang={locale}>{translate(locale, k)}</span>;
}

/**
 * One sentence of the forensic report, in the reader's language (R16-T3).
 *
 * The English is the key and stays where it was written — in the report, or in the shared
 * vocabulary of `../analysis` — so this leaf is handed exactly the sentence the English report
 * prints, and looks it up; see `translateCopy`. A sentence with no Turkish entry is drawn in
 * English and says so with `lang="en"`, which is also what the report's tests look for.
 *
 * `{name}` slots in a template are filled from `values`, and what fills them is never looked
 * up: a host, a status, a rule id or a figure is drawn exactly as the record holds it, in
 * whichever order the reader's language puts the words around it.
 */
export function Copy({
  children,
  text,
  values,
}: {
  children?: string;
  text?: string;
  values?: Record<string, ReactNode>;
}) {
  const [locale] = useLocale();
  const english = text ?? children ?? "";
  const copy = translateCopy(locale, english);

  return (
    <span lang={copy.lang}>
      {copySegments(copy.text).map((segment, index) =>
        typeof segment === "string" ? (
          segment
        ) : (
          <Fragment key={index}>{values?.[segment.slot] ?? `{${segment.slot}}`}</Fragment>
        ),
      )}
    </span>
  );
}

/**
 * `<Copy>` for a `title` (R16-T5): the dashboard's hover text, in the reader's language.
 *
 * A server component cannot know the reader's language, and an attribute cannot hold a client
 * island, so the element that carries the title is drawn here. Slot values are written in as the
 * record holds them, exactly as `<Copy>` draws them.
 */
export function CopyTitle({
  text,
  values,
  as: Tag = "span",
  className,
  children,
}: {
  text: string;
  values?: Record<string, string | number>;
  as?: "span" | "div";
  className?: string;
  children: ReactNode;
}) {
  const [locale] = useLocale();
  const title = copySegments(translateCopy(locale, text).text)
    .map((segment) =>
      typeof segment === "string" ? segment : String(values?.[segment.slot] ?? `{${segment.slot}}`),
    )
    .join("");
  return (
    <Tag className={className} title={title}>
      {children}
    </Tag>
  );
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
  // A value shown as the API spelled it is not a word of the reader's language, so it claims
  // none and takes the language of whatever it is drawn inside.
  return label === value ? <span>{label}</span> : <span lang={locale} title={value}>{label}</span>;
}

/**
 * One `<option>` naming an interface string. An `<option>` may hold only text, so the word is
 * computed here and handed to it as a string rather than rendered through `<T>`.
 */
export function MessageOption({ value, k }: { value: string; k: MessageKey }) {
  const [locale] = useLocale();
  return <option value={value}>{translate(locale, k)}</option>;
}

/** One `<option>` naming a value the API owns. `value` is what the form posts, untouched. */
export function CanonicalOption({ domain, value }: { domain: CanonicalDomain; value: string }) {
  const [locale] = useLocale();
  return <option value={value}>{translateCanonical(locale, domain, value)}</option>;
}

/*
 * Where the switch sits. `rail` is the admin navigation it was written for; `bar` is the
 * workspace's header and the sign-in pages, on the same graphite ground but beside other
 * controls rather than above them; `document` is the report's light masthead.
 */
const SELECTOR_STYLES = {
  rail: {
    label: "flex items-center justify-between gap-2 px-3 text-[12px] text-muted",
    word: "",
    select: "rounded-md border border-line bg-ink px-2 py-1 text-[12px] text-bone",
  },
  bar: {
    label: "flex items-center gap-2 text-[12px] text-muted",
    // The word is spoken to a screen reader on a phone, where the header has no room for it.
    word: "sr-only sm:not-sr-only",
    select: "rounded-md border border-line bg-ink px-2 py-1 text-[12px] text-bone",
  },
  document: {
    label: "flex items-center gap-2 text-sm",
    word: "sr-only",
    select: "rounded-md border border-black/20 bg-paper px-2 py-1 text-sm text-doc",
  },
} as const;

/**
 * The language switch.
 *
 * A native `<select>`, so it is keyboard- and screen-reader-complete without any code here.
 * Changing it writes `localStorage` and repaints; it does not navigate, does not fetch and does
 * not touch the address bar. It also keeps `<html lang>` in step with what is painted, which is
 * the one piece of the document outside React's tree that a language change has to reach.
 */
export function LanguageSelector({
  variant = "rail",
}: {
  variant?: keyof typeof SELECTOR_STYLES;
}) {
  const [locale, setLocale] = useLocale();
  const styles = SELECTOR_STYLES[variant];

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  return (
    <label className={styles.label}>
      <span className={styles.word}>{translate(locale, "locale.selector")}</span>
      <select
        value={locale}
        onChange={(event) => {
          if (isLocale(event.target.value)) {
            setLocale(event.target.value);
          }
        }}
        className={styles.select}
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
