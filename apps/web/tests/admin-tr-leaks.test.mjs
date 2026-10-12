// R16 admin localization — the administrative console in two languages.
//
// Every admin route is rendered through the real page module, fed the fixtures in
// `support/admin-fixtures.mjs` through a replaced `fetch`, once with the language fixed to English
// and once to Turkish — including the empty, error and feedback states a route can show. The
// structural audit then holds the Turkish console to the rule the customer pages are held to in
// `report-tr-leaks.test.mjs`: every word it shares with the English is a value from the record or a
// word classified on purpose. Casing and tooltips are checked over the same documents.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";

register("./support/tsx-loader.mjs", import.meta.url);

const { createElement } = await import("react");
const { renderToStaticMarkup } = await import("react-dom/server");
const { default: Accounts } = await import("../app/admin/page.tsx");
const { default: Account } = await import("../app/admin/users/[id]/page.tsx");
const { default: Jobs } = await import("../app/admin/jobs/page.tsx");
const { default: Queue } = await import("../app/admin/review-queue/page.tsx");
const { default: Case } = await import("../app/admin/analyses/[id]/page.tsx");
const { default: Analytics } = await import("../app/admin/analytics/page.tsx");
const { default: Audit } = await import("../app/admin/audit/page.tsx");
const { default: ApiKeys } = await import("../app/admin/api-keys/page.tsx");
const { FixedLocale } = await import("../app/i18n/client.tsx");
// `CreateKey` on the API keys page is a client component that calls `useRouter()`; outside a
// Next.js request it needs the router context a real request would mount. Nothing navigates here.
const { AppRouterContext } = await import("next/dist/shared/lib/app-router-context.shared-runtime.js");
const ROUTER = { push() {}, replace() {}, refresh() {}, back() {}, forward() {}, prefetch() {} };
const { CANONICAL, MESSAGES, WEB_MESSAGE_PATTERNS, translateCanonical } = await import("../app/i18n/core.ts");
const { AdminSidebar } = await import("../app/admin/components/AdminSidebar.tsx");
const { PathnameContext } = await import("next/dist/shared/lib/hooks-client-context.shared-runtime.js");
const F = await import("./support/admin-fixtures.mjs");

process.env.API_INTERNAL_URL = "http://api.invalid";

const LOCALES = ["en", "tr"];
const page = (query = {}, params = {}) => ({
  searchParams: Promise.resolve(query),
  params: Promise.resolve(params),
});
const refused = { status: 503, body: {} };

/** Each admin route, in each state worth reading, as `[api overrides, render]`. */
const STATES = {
  "accounts": [{}, () => Accounts(page())],
  "accounts:updated": [{}, () => Accounts(page({ updated: F.OTHER }))],
  "accounts:created": [{}, () => Accounts(page({ created: F.OTHER }))],
  "accounts:error": [{}, () => Accounts(page({ error: "Give the account a password." }))],
  "accounts:unavailable": [{ "/api/v1/admin/users": refused }, () => Accounts(page())],
  "account": [{}, () => Account(page({}, { id: F.OTHER }))],
  "account:reset": [{}, () => Account(page({ reset: "1" }, { id: F.OTHER }))],
  "account:updated": [{}, () => Account(page({ updated: "1" }, { id: F.OTHER }))],
  "account:self": [{}, () => Account(page({}, { id: F.ADMIN.id }))],
  "jobs": [{}, () => Jobs()],
  "jobs:empty": [{ "/api/v1/admin/jobs": [] }, () => Jobs()],
  "jobs:unavailable": [{ "/api/v1/admin/jobs": refused }, () => Jobs()],
  "queue": [{}, () => Queue(page())],
  "queue:filtered": [{}, () => Queue(page({ feedback_assessment: "DISAGREE", signal_error: "true", unreviewed_only: "false", offset: "50" }))],
  "queue:empty": [{ "/api/v1/admin/analyses/queue": { ...F.QUEUE, items: [], total: 0 } }, () => Queue(page())],
  "queue:unavailable": [{ "/api/v1/admin/analyses/queue": refused }, () => Queue(page())],
  "case": [{}, () => Case(page({ saved: "1", gt_saved: "1" }, { id: F.ANALYSIS.id }))],
  "case:unreviewed": [
    { [`/api/v1/admin/analyses/${F.ANALYSIS.id}/review`]: F.UNREVIEWED, [`/api/v1/admin/ground-truth/${F.SHA}`]: { status: 404, body: {} } },
    () => Case(page({}, { id: F.ANALYSIS.id })),
  ],
  "case:legacyReview": [{ [`/api/v1/admin/analyses/${F.ANALYSIS.id}/review`]: F.LEGACY_REVIEW }, () => Case(page({}, { id: F.ANALYSIS.id }))],
  "case:promoteManual": [{}, () => Case(page({ promote: "manual" }, { id: F.ANALYSIS.id }))],
  "case:promoteFeedback": [
    { [`/api/v1/admin/dataset-governance/${F.SHA}`]: { status: 404, body: {} } },
    () => Case(page({ promote: "feedback", feedback: F.OTHER }, { id: F.ANALYSIS.id })),
  ],
  "case:promoteMissingFeedback": [{}, () => Case(page({ promote: "feedback", feedback: "nobody" }, { id: F.ANALYSIS.id }))],
  "case:errors": [
    { [`/api/v1/admin/analyses/${F.ANALYSIS.id}/feedback`]: refused, [`/api/v1/admin/ground-truth/${F.SHA}`]: refused },
    () => Case(page({ error: "Choose the review status to leave this case in.", gt_error: "Choose what the media is." }, { id: F.ANALYSIS.id })),
  ],
  "analytics": [{}, () => Analytics()],
  "analytics:unavailable": [{ "/api/v1/admin/analytics": refused }, () => Analytics()],
  "audit": [{}, () => Audit()],
  "audit:empty": [{ "/api/v1/admin/audit": [] }, () => Audit()],
  "audit:unreadable": [{ "/api/v1/admin/audit": [...F.AUDIT, F.REVIEW_UPDATED_EVENT] }, () => Audit()],
  "audit:unavailable": [{ "/api/v1/admin/audit": refused }, () => Audit()],
  "apiKeys": [{}, () => ApiKeys(page())],
  "apiKeys:confirm": [{}, () => ApiKeys(page({ revoking: F.API_KEYS[0].id }))],
  "apiKeys:revoked": [{}, () => ApiKeys(page({ revoked: F.API_KEYS[1].id }))],
  "apiKeys:error": [{}, () => ApiKeys(page({ error: "The revocation named no key." }))],
  "apiKeys:empty": [{ "/api/v1/admin/api-keys": [] }, () => ApiKeys(page())],
  // Runtime-valued sentences (Architect fix): the words are Turkish, the value stays raw.
  "apiKeys:nameTooLong": [{}, () => ApiKeys(page({ error: "A name may be at most 255 characters." }))],
  "case:reviewRefused": [{}, () => Case(page({ error: "The review was refused (HTTP 403)." }, { id: F.ANALYSIS.id }))],
  "case:groundTruthRefused": [{}, () => Case(page({ gt_error: "The Ground Truth was refused (HTTP 409)." }, { id: F.ANALYSIS.id }))],
};

/** The promotion outcomes the case review can report, read from the query string. */
for (const [name, query] of Object.entries({
  "case:promotionSaved": { promo_saved: "governance,ground_truth,review" },
  "case:promotionPartial": { promo_saved: "governance", promo_failed: "ground_truth", promo_error: "Choose what the media is." },
  "case:promotionReviewFailed": { promo_saved: "governance,ground_truth", promo_failed: "review", promo_error: "The review is temporarily unavailable." },
  "case:promotionStepRefused": { promo_saved: "governance", promo_failed: "ground_truth", promo_error: "Ground Truth was refused (HTTP 409)." },
  "case:promotionUnreachable": { promo_saved: "", promo_failed: "governance", promo_error: "Dataset governance: the API could not be reached." },
  "case:promotionNested": { promo_saved: "governance,ground_truth", promo_failed: "review", promo_error: "Review: The review is temporarily unavailable." },
  "case:promotionStale": { promo_saved: "governance,ground_truth", promo_failed: "review", promo_error: "The review changed to REVIEWED after this form was opened; it was not updated. Check it and choose again." },
  "case:promotionField": { promo_saved: "", promo_error: "redistributable must be yes, no or not recorded." },
})) {
  STATES[name] = [{}, () => Case(page(query, { id: F.ANALYSIS.id }))];
}

const PAGES = {};
for (const [name, [overrides, render]] of Object.entries(STATES)) {
  PAGES[name] = {};
  for (const locale of LOCALES) {
    F.stubApi(overrides);
    PAGES[name][locale] = renderToStaticMarkup(
      createElement(AppRouterContext.Provider, { value: ROUTER }, createElement(FixedLocale, { locale }, await render())),
    );
  }
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

const unescape = (value) =>
  value.replaceAll("&quot;", '"').replaceAll("&#x27;", "'").replaceAll("&lt;", "<").replaceAll("&gt;", ">").replaceAll("&amp;", "&");

/* ------------------------------------------------------------------ *
 * The dictionary
 * ------------------------------------------------------------------ */

const ADMIN_KEYS = Object.keys(MESSAGES.en).filter((key) => key.startsWith("admin."));

test("every admin string has a Turkish entry, and Turkish defines no key English lacks", () => {
  assert.ok(ADMIN_KEYS.length > 250, `only ${ADMIN_KEYS.length} admin keys`);
  assert.deepEqual(ADMIN_KEYS.filter((key) => !Object.hasOwn(MESSAGES.tr, key)), []);
  assert.deepEqual(Object.keys(MESSAGES.tr).filter((key) => !Object.hasOwn(MESSAGES.en, key)), []);
});

test("a Turkish admin string carries exactly the slots of its English", () => {
  const slots = (text) => [...text.matchAll(/\{([A-Za-z]+)\}/g)].map((match) => match[1]).sort();
  for (const key of ADMIN_KEYS) {
    assert.deepEqual(slots(MESSAGES.tr[key]), slots(MESSAGES.en[key]), key);
  }
});

test("a Turkish admin string is Turkish: only `Ground Truth` and technical words may equal the English", () => {
  const same = ADMIN_KEYS.filter((key) => MESSAGES.tr[key] === MESSAGES.en[key]);
  assert.deepEqual(same.sort(), ["admin.case.gt.title", "admin.case.promotion.step.groundTruth"]);
});

/* ------------------------------------------------------------------ *
 * The audit log's action badge (Architect ruling 2)
 * ------------------------------------------------------------------ */

test("every audit action the API writes is named in Turkish, and keeps its spelling as data-canonical", () => {
  const { tr, en } = PAGES.audit;
  for (const action of F.AUDIT_ACTIONS) {
    const name = CANONICAL.auditAction.tr[action];
    assert.ok(name && name !== action, action);
    assert.ok(tr.includes(`<span lang="tr" data-canonical="${action}">${name}</span>`), action);
    assert.ok(!visible(tr).includes(action), `${action} is painted in Turkish`);
    // English keeps the API's spelling, as the log always printed it.
    assert.ok(en.includes(`<span>${action}</span>`), action);
  }
  assert.equal(CANONICAL.auditAction.tr.REVIEW_UPDATED, "İnceleme güncellendi");
  assert.equal(CANONICAL.auditAction.tr.USER_UPDATED, "Kullanıcı güncellendi");
  assert.equal(CANONICAL.auditAction.tr.API_KEY_CREATED, "API anahtarı oluşturuldu");
});

test("an audit action this build does not know is drawn exactly as recorded, in both languages", () => {
  for (const locale of LOCALES) {
    const html = PAGES.audit[locale];
    assert.ok(html.includes(`<span>${F.UNKNOWN_ACTION}</span>`), locale);
    assert.ok(!html.includes(`data-canonical="${F.UNKNOWN_ACTION}"`), locale);
    // Its unknown target type is drawn raw as well.
    assert.ok(html.includes("<span>WORKSPACE</span>"), locale);
  }
});

test("the diff's field names stay raw and English-cased; ids, timestamps and request ids stay raw", () => {
  const { tr } = PAGES.audit;
  assert.ok(tr.includes('<dt lang="en" class="font-mono text-[11px] tracking-[0.06em] text-muted uppercase">is_active</dt>'));
  for (const event of F.AUDIT) {
    assert.ok(tr.includes(event.created_at), event.id);
    assert.ok(tr.includes(event.target_id), event.id);
    if (event.request_id) assert.ok(tr.includes(event.request_id), event.id);
  }
  assert.ok(visible(tr).includes("istek req-a1"));
});

/* ------------------------------------------------------------------ *
 * Canonical values stay raw
 * ------------------------------------------------------------------ */

test("canonical and technical values are identical in both languages", () => {
  const raw = [
    F.ANALYSIS.id,
    F.SHA,
    "r9-v5.0.0",
    F.ANALYSIS.risk_rule_id,
    F.ANALYSIS.risk_calibration_id,
    F.ANALYSIS.created_at,
    "nvidia",
    "svd",
    "fn-2026.09",
    "MANIPULATION_DETECTED",
    "NO_CALIBRATED_MANIPULATION_SIGNAL",
    "UNDECIDED",
    "r7-v4.0.0/BOGUS",
    F.JOBS[1].id,
    F.JOBS[1].request_id,
    F.JOBS[3].error_message,
    F.API_KEYS[0].id,
    "lineage-r16p-008",
  ].filter(Boolean);
  const pages = Object.values(PAGES);
  for (const value of raw) {
    const inEnglish = pages.filter(({ en }) => visible(en).includes(value)).length;
    const inTurkish = pages.filter(({ tr }) => visible(tr).includes(value)).length;
    assert.ok(inEnglish > 0, `${value} is never shown`);
    assert.equal(inTurkish, inEnglish, `${value} is shown on ${inEnglish} English pages and ${inTurkish} Turkish ones`);
  }
});

test("a known job status is Turkish; the stale chip and its tooltip are Turkish", () => {
  const { tr } = PAGES.jobs;
  for (const status of ["queued", "processing", "completed", "failed"]) {
    assert.ok(tr.includes(`data-canonical="${status}">${CANONICAL.jobStatus.tr[status]}</span>`), status);
  }
  assert.ok(visible(tr).includes("Gecikmiş"));
  assert.ok(tr.includes(`title="${MESSAGES.tr["admin.jobs.staleTitle"].replaceAll("'", "&#x27;")}"`));
  assert.ok(!tr.includes("The worker&#x27;s claim"));
});

test("form posts and query values are unchanged by the language", () => {
  const values = (html) => [...html.matchAll(/\s(?:value|name|action|href)="([^"]*)"/g)].map((match) => match[1]);
  for (const [name, { en, tr }] of Object.entries(PAGES)) {
    assert.deepEqual(values(tr), values(en), name);
  }
});

/* ------------------------------------------------------------------ *
 * The structural audit
 * ------------------------------------------------------------------ */

/*
 * A word the Turkish console shares with the English, on purpose — the classes
 * `report-tr-leaks.test.mjs` defines, for the words this surface adds.
 */
const CLASSIFIED = {
  // technical — products, providers, standards, formats
  InspectRoot: "technical",
  API: "technical",
  URL: "technical",
  "SHA-256": "technical",
  C2PA: "technical",
  JavaScript: "technical",
  hash: "technical",
  sha256: "technical", // the review queue's label for a media hash, beside the hash itself
  Content: "technical", // "Content Credentials", the C2PA standard's own name
  Credentials: "technical",
  worker: "technical", // the detection worker, named as the operators and the logs name it
  Worker: "technical",
  manifest: "shared",
  English: "technical", // the language selector names each language in itself
  // technical — the rulesets' own verdict and level names, written raw in both languages
  MANIPULATION_DETECTED: "technical",
  NO_CALIBRATED_MANIPULATION_SIGNAL: "technical",
  INCONCLUSIVE: "technical",
  UNDECIDED: "technical",
  LOW: "technical",
  MEDIUM: "technical",
  HIGH: "technical",
  UNKNOWN: "technical",
  // technical — the domain term the Architect kept in English (Ground Truth)
  Ground: "technical",
  Truth: "technical",
  // shared — Turkish spells these the same way
  Risk: "shared",
  risk: "shared",
  Test: "shared",
  test: "shared", // "Kontrollü test"
  not: "shared", // Turkish "not" (a note), as in "bir inceleyicinin not ettiği"
  model: "shared", // "model değerlendirme hataları"
  form: "shared", // "bu form açıldıktan sonra"
  HTTP: "technical", // "(HTTP 403)": the status code's own protocol name, beside the raw code
  holdout: "shared",
  transaction: "shared", // glossed in Turkish beside it: "işlem (transaction)"
};

const WORD = /(?<![\p{L}\p{N}'’])\p{L}[\p{L}\p{N}_-]*/gu;
const words = (text) => new Set((text.match(WORD) ?? []).filter((word) => /^[A-Za-z_-]+[A-Za-z0-9_.-]*$/.test(word)));
const RULESET = /^[a-z]\d+-v\d/; // `r9-v5`, `p7-v1` — the version is cut at its first dot

/** Every word of every value the fixtures hold, except a word the reader's language names. */
function recordWords() {
  const named = new Set(Object.values(CANONICAL).flatMap((domain) => Object.keys(domain.tr)));
  // `null` is how the analytics table writes a version the row does not hold.
  const out = new Set(["example", "com", "req", "fn", "null"]);
  const walk = (value) => {
    if (typeof value === "string" && !named.has(value)) for (const word of value.match(WORD) ?? []) out.add(word);
    else if (Array.isArray(value)) value.forEach(walk);
    else if (value && typeof value === "object") {
      for (const [key, inner] of Object.entries(value)) {
        // A key that is drawn — a diff's field name, an analytics category — is a record word too.
        if (!named.has(key)) for (const word of key.match(WORD) ?? []) out.add(word);
        walk(inner);
      }
    }
  };
  for (const fixture of [F.ADMIN, F.ACCOUNTS, F.JOBS, F.QUEUE, F.ANALYTICS, F.AUDIT, F.API_KEYS, F.REVIEW, F.GROUND_TRUTH, F.GOVERNANCE, F.FEEDBACK, F.ANALYSIS]) {
    walk(fixture);
  }
  out.add(F.UNKNOWN_ACTION);
  out.add("WORKSPACE");
  // A diff's values are drawn as recorded, including the ones a domain names elsewhere (`USER`,
  // `ADMIN`) and the booleans, which the page writes out as `true` / `false`.
  for (const event of F.AUDIT) {
    for (const change of Object.values(event.changes)) for (const side of [change.old, change.new]) out.add(String(side));
  }
  // An id shortened to its first eight characters, as the success notices print it.
  for (const id of [...F.ACCOUNTS.map((account) => account.id), ...F.API_KEYS.map((key) => key.id)]) out.add(id.slice(0, 8));
  return out;
}

test("every English word in the Turkish console is a record value or classified (structural audit)", () => {
  const record = recordWords();
  const unclassified = {};
  for (const [name, { en, tr }] of Object.entries(PAGES)) {
    const english = words(visible(en));
    for (const word of words(visible(tr))) {
      if (!english.has(word) || record.has(word) || Object.hasOwn(CLASSIFIED, word) || RULESET.test(word)) continue;
      (unclassified[word] ??= []).push(name);
    }
  }
  assert.deepEqual(unclassified, {}, "English in the Turkish console that is neither record nor classified");
});

test("no presentation sentence of the English console is painted on the Turkish one", () => {
  // Every admin string with at least four words, as English prints it, and the leaks the
  // Architect named. None may appear in the Turkish text of any state.
  const sentences = ADMIN_KEYS.map((key) => MESSAGES.en[key])
    .filter((text) => !/\{/.test(text) && text.split(" ").length >= 4);
  const leaks = [];
  for (const [name, { tr }] of Object.entries(PAGES)) {
    const text = visible(tr);
    for (const sentence of sentences) if (text.includes(sentence)) leaks.push(`${name}: ${sentence}`);
    for (const leak of ["Administration", "Refresh", "Account updated", "Audit log", "Detection jobs", "never used"]) {
      if (text.includes(leak)) leaks.push(`${name}: ${leak}`);
    }
  }
  assert.deepEqual(leaks, []);
});

/* ------------------------------------------------------------------ *
 * Casing (Architect ruling 1: no CLAİM / CHECKPOİNT)
 * ------------------------------------------------------------------ */

/**
 * The page's text as the browser paints it under `text-transform: uppercase`, inside a document
 * whose `<html lang>` is the reader's language: each run carries the nearest `lang` above it.
 */
function uppercased(html, documentLang) {
  const runs = [];
  const stack = [{ upper: false, lang: documentLang }];
  const VOID = new Set(["br", "img", "input", "meta", "link", "hr", "source", "col", "wbr"]);
  const body = html.replace(/<(style|script)[^>]*>[\s\S]*?<\/\1>/g, "");
  for (const [, close, tag, attrs, text] of body.matchAll(/<(\/?)([a-zA-Z0-9]+)([^>]*)>|([^<]+)/g)) {
    if (text !== undefined) {
      const top = stack[stack.length - 1];
      if (top.upper && text.trim()) runs.push({ text: unescape(text).toLocaleUpperCase(top.lang), lang: top.lang, source: unescape(text) });
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

test("no English word is cased by Turkish rules anywhere in the Turkish console", () => {
  // A word the Turkish page shares with the English one is not Turkish (a raw value, a technical
  // name); under `uppercase` it must be inside a `lang="en"` run, or its `i` becomes `İ`.
  const broken = [];
  for (const [name, { en, tr }] of Object.entries(PAGES)) {
    const english = words(visible(en));
    for (const run of uppercased(tr, "tr")) {
      if (run.lang !== "tr") continue;
      for (const word of run.source.match(WORD) ?? []) {
        if (!english.has(word) || CLASSIFIED[word] === "shared") continue;
        if (word.toLocaleUpperCase("tr") !== word.toLocaleUpperCase("en")) {
          broken.push(`${name}: ${word} → ${word.toLocaleUpperCase("tr")}`);
        }
      }
    }
  }
  assert.deepEqual(broken, []);
});

test("the casing failures R16-T8 found cannot recur: no `İ` inside an ASCII identifier", () => {
  for (const [name, { tr }] of Object.entries(PAGES)) {
    const painted = uppercased(tr, "tr").map((run) => run.text).join("\n");
    for (const broken of ["CLAİM", "CHECKPOİNT", "İS_ACTİVE", "MİSSİNG", "UNRECOGNİSED"]) {
      assert.ok(!painted.includes(broken), `${name}: ${broken}`);
    }
    assert.ok(!/[A-Z_]+İ[A-Z_]*_[A-Z_İ]+/.test(painted), `${name}: an identifier cased by Turkish rules`);
  }
  // The legend is Turkish and cased as Turkish.
  assert.ok(uppercased(PAGES.jobs.tr, "tr").some((run) => run.text === "YÖNETİM"));
});

/* ------------------------------------------------------------------ *
 * Tooltips
 * ------------------------------------------------------------------ */

test("no Turkish tooltip carries English presentation or a known API word", () => {
  const known = new Set(
    Object.values(CANONICAL).flatMap((domain) =>
      Object.entries(domain.tr).filter(([value, name]) => value !== "" && value !== name).map(([value]) => value),
    ),
  );
  const record = recordWords();
  const leaks = [];
  for (const [name, { en, tr }] of Object.entries(PAGES)) {
    const englishTitles = new Set([...en.matchAll(/\stitle="([^"]*)"/g)].map(([, raw]) => unescape(raw)));
    for (const [, raw] of tr.matchAll(/\stitle="([^"]*)"/g)) {
      const title = unescape(raw);
      // A title that is a record value (an id, an address, a hash) is the same in both languages.
      if (title === "" || [...(title.match(WORD) ?? [])].every((word) => record.has(word))) continue;
      if (englishTitles.has(title)) leaks.push(`${name}: English title "${title}"`);
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
});

/* ------------------------------------------------------------------ *
 * English regression
 * ------------------------------------------------------------------ */

test("the English console still reads as it did", () => {
  // Tags become spaces in `visible`, so a full stop after a link reads " ." — closed up here.
  const text = (name) => visible(PAGES[name].en).replace(/ ([.,?:])/g, "$1");
  const expected = {
    accounts: ["Administration", "Accounts", "Create an account", "At least 12 characters.", "Your own account — another administrator must change it.", "Deactivated"],
    "accounts:updated": ["Account updated · b0000000."],
    "accounts:created": ["Account created. The password is not recoverable from anywhere — tell the person what it is, or set a new one from their account page."],
    jobs: ["Detection jobs", "job c0000001-", "request req-0001", "Stale"],
    queue: ["Review queue", "Owner feedback", "Has AGREE", "Has FAILED or TIMEOUT", "All, including REVIEWED", "UNREVIEWED", "AGREE 0", "FAILED 1", "1–3 of 3"],
    case: ["Case review", "Automated assessment", "Decision coverage", "2/2 complete", "Authenticity and provenance", "Reviewed", "Disagrees with automated assessment", "Ground Truth", "Known to the owner", "Claims: Genuine", "first written 2026-10-05T09:00:00Z"],
    "case:promotionPartial": ["Partial success. Saved: dataset governance. Failed: Ground Truth — Choose what the media is. Not attempted: review status. Correct the form and submit it again; a step already saved is not written twice."],
    analytics: ["Operational summary", "Analyses submitted", "Detector health", "Failure rate", "MISSING", "unrecorded", "Decisions (r9-v5.0.0)"],
    audit: ["Audit log", "Administrator", "USER_UPDATED", "user b0000000-", "request req-a1", "not recorded"],
    apiKeys: ["API keys", "Issue a key", "Issue key", "never used", "Revoked"],
    "apiKeys:confirm": ["Revoke Acme production? This cannot be undone.", "Confirm", "Cancel"],
  };
  for (const [name, phrases] of Object.entries(expected)) {
    for (const phrase of phrases) assert.ok(text(name).includes(phrase), `${name}: ${phrase}`);
  }
  assert.ok(PAGES.apiKeys.en.includes('placeholder="Who is this key for?"'));
  assert.ok(PAGES.apiKeys.tr.includes('placeholder="Bu anahtar kimin için?"'));
});

/* ------------------------------------------------------------------ *
 * Architect fixes: runtime-valued sentences and the rail's accessible name
 * ------------------------------------------------------------------ */

test("a sentence with a runtime value is Turkish around the value, and the value stays raw", () => {
  const cases = {
    "The review was refused (HTTP 403).": "İnceleme reddedildi (HTTP 403).",
    "The Ground Truth was refused (HTTP 409).": "Ground Truth reddedildi (HTTP 409).",
    "A name may be at most 255 characters.": "Ad en fazla 255 karakter olabilir.",
    "redistributable must be yes, no or not recorded.": "redistributable alanı evet, hayır ya da kaydedilmemiş olmalıdır.",
    "Dataset governance was refused (HTTP 422).": "Veri kümesi yönetişimi reddedildi (HTTP 422).",
    "Review: the API could not be reached.": "İnceleme: API'ye ulaşılamadı.",
    // A nested sentence is looked up on its own; the API's own text after a step stays raw.
    "Review: The review is temporarily unavailable.": "İnceleme: İnceleme geçici olarak kullanılamıyor.",
    "Ground Truth: Unsupported label.": "Ground Truth: Unsupported label.",
    "Dataset governance: Unsupported split.": "Veri kümesi yönetişimi: Unsupported split.",
    // A known value inside is named; an unknown one is drawn as it arrived.
    "The review changed to REVIEWED after this form was opened; it was not updated. Check it and choose again.":
      "İnceleme, bu form açıldıktan sonra incelendi durumuna geçti; güncellenmedi. Kontrol edip yeniden seçin.",
  };
  for (const [english, turkish] of Object.entries(cases)) {
    assert.equal(translateCanonical("tr", "webMessage", english), turkish, english);
    assert.equal(translateCanonical("en", "webMessage", english), english, english);
  }
  // Nothing a pattern does not name is claimed: other steps, other codes, prose, the API's text.
  for (const raw of [
    "The feedback was refused (HTTP 500).",
    "The review was refused (HTTP 4O3).",
    "Workspace: the API could not be reached.",
    "The review changed to ARCHIVED after this form was opened; it was not updated. Check it and choose again.",
    "Redistributable must be yes, no or not recorded.",
    "An account with that email already exists.",
    "constructor",
  ]) {
    assert.equal(translateCanonical("tr", "webMessage", raw), raw, raw);
  }
});

test("each runtime-valued state is painted in Turkish with its value raw", () => {
  const expected = {
    "apiKeys:nameTooLong": "Ad en fazla 255 karakter olabilir.",
    "case:reviewRefused": "İnceleme reddedildi (HTTP 403).",
    "case:groundTruthRefused": "Ground Truth reddedildi (HTTP 409).",
    "case:promotionStepRefused": "Ground Truth reddedildi (HTTP 409).",
    "case:promotionUnreachable": "Veri kümesi yönetişimi: API'ye ulaşılamadı.",
    "case:promotionNested": "İnceleme: İnceleme geçici olarak kullanılamıyor.",
    "case:promotionStale": "incelendi durumuna geçti",
    "case:promotionField": "redistributable alanı evet, hayır ya da kaydedilmemiş olmalıdır.",
  };
  for (const [name, turkish] of Object.entries(expected)) {
    assert.ok(visible(PAGES[name].tr).includes(turkish), `${name}: ${turkish}`);
    for (const english of ["was refused", "may be at most", "could not be reached", "must be yes", "changed to"]) {
      assert.ok(!visible(PAGES[name].tr).includes(english), `${name}: ${english}`);
    }
  }
  // English is unchanged, and the record's sentence stays on the element.
  assert.ok(visible(PAGES["case:reviewRefused"].en).includes("The review was refused (HTTP 403)."));
  assert.ok(PAGES["case:reviewRefused"].tr.includes('data-canonical="The review was refused (HTTP 403)."'));
});

test("every runtime-valued pattern is still written, word for word, by the code that sends it", () => {
  const source = ["promotion.ts", "promote-ground-truth/route.ts", "update-review/route.ts", "update-ground-truth/route.ts", "create-api-key/route.ts"]
    .map((path) => readFileSync(new URL(`../app/admin/${path}`, import.meta.url), "utf8"))
    .join("\n");
  for (const { en } of WEB_MESSAGE_PATTERNS) {
    for (const fragment of en.split(/\{[A-Za-z]+\}/).filter((part) => part.trim().length >= 3)) {
      assert.ok(source.includes(fragment), `no longer written: "${fragment}" of "${en}"`);
    }
  }
  for (const step of Object.keys(CANONICAL.promotionStep.en)) {
    assert.ok(source.includes(`"${step}"`), `step name no longer written: ${step}`);
  }
});

test("the admin rail's screen-reader name is in the reader's language", () => {
  for (const [locale, name] of [["en", "Administration"], ["tr", "Yönetim"]]) {
    // `AdminNavLink` reads the current path; outside a request it needs the context a request mounts.
    const html = renderToStaticMarkup(
      createElement(PathnameContext.Provider, { value: "/admin/jobs" }, createElement(FixedLocale, { locale }, createElement(AdminSidebar))),
    );
    assert.ok(html.startsWith(`<nav aria-label="${name}"`), `${locale}: ${html.slice(0, 80)}`);
  }
});
