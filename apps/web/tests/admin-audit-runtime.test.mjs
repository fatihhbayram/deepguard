// R16 admin localization — the audit log's functional incompatibility, proven rather than fixed.
//
// CONFIRMED FUNCTIONAL BUG. `app/api/admin_analyses.py` (`_review_changes`) writes every
// REVIEW_CREATED / REVIEW_UPDATED event with `changes: {"note_changed": <bool>, …}`: a bare boolean
// beside the `{old, new}` pairs. `app/admin/audit.ts` (`parseChanges`) accepts only `{old, new}`
// pairs, so that one event makes `parseEvent` return null, and `fetchAuditEvents` then refuses the
// whole list — every row, not just the review's — with "The audit log could not be read."
//
// Localization does not fix this and is not allowed to: the backend and the parser are unchanged
// by this task, and these tests pin the failure as it stands so the separate minimum-change fix has
// a regression to turn green. What localization does is name the resulting error in the reader's
// language, which is the last test here.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";

register("./support/tsx-loader.mjs", import.meta.url);

const { createElement } = await import("react");
const { renderToStaticMarkup } = await import("react-dom/server");
const { fetchAuditEvents, parseChanges, parseEvent } = await import("../app/admin/audit.ts");
const { default: AdminAudit } = await import("../app/admin/audit/page.tsx");
const { FixedLocale } = await import("../app/i18n/client.tsx");
const { AUDIT, REVIEW_UPDATED_EVENT, stubApi } = await import("./support/admin-fixtures.mjs");

process.env.API_INTERNAL_URL = "http://api.invalid";

const UNREADABLE_EN = "The audit log could not be read.";
const UNREADABLE_TR = "Denetim kaydı okunamadı.";

test("the backend writes `note_changed` as a bare boolean, not an {old, new} pair", () => {
  // Read, not imported: the API is not part of this test run and is not changed by this task.
  const source = readFileSync(new URL("../../api/app/api/admin_analyses.py", import.meta.url), "utf8");
  assert.match(source, /changes: dict = \{"note_changed": note_moved\}/);
  assert.match(source, /AUDIT_ACTION_REVIEW_UPDATED/);
});

test("parseChanges rejects the review event's `changes` (returns null)", () => {
  assert.equal(parseChanges({ note_changed: false }), null);
  assert.equal(parseChanges({ note_changed: true }), null);
  assert.equal(parseChanges(REVIEW_UPDATED_EVENT.changes), null);
  // The same parser accepts every other event shape the API writes.
  assert.deepEqual(parseChanges({ is_active: { old: true, new: false } }), { is_active: { old: true, new: false } });
});

test("parseEvent rejects the whole review event", () => {
  assert.equal(parseEvent(REVIEW_UPDATED_EVENT), null);
  assert.notEqual(parseEvent(AUDIT[0]), null);
});

test("one review event makes fetchAuditEvents refuse the entire log", async () => {
  stubApi({ "/api/v1/admin/audit": AUDIT });
  const healthy = await fetchAuditEvents();
  assert.equal(healthy.ok, true);
  assert.equal(healthy.events.length, AUDIT.length);

  stubApi({ "/api/v1/admin/audit": [...AUDIT, REVIEW_UPDATED_EVENT] });
  const poisoned = await fetchAuditEvents();
  assert.deepEqual(poisoned, { ok: false, unauthenticated: false, error: UNREADABLE_EN });
});

async function auditPage(locale, events) {
  stubApi({ "/api/v1/admin/audit": events });
  return renderToStaticMarkup(createElement(FixedLocale, { locale }, await AdminAudit()));
}

test("the resulting error is named in each language, and no row survives it", async () => {
  const en = await auditPage("en", [...AUDIT, REVIEW_UPDATED_EVENT]);
  const tr = await auditPage("tr", [...AUDIT, REVIEW_UPDATED_EVENT]);

  assert.ok(en.includes(UNREADABLE_EN));
  assert.ok(tr.includes(`<span lang="tr" data-canonical="${UNREADABLE_EN}">${UNREADABLE_TR}</span>`));
  // The English sentence stays on the element as `data-canonical`, never in the painted text.
  assert.ok(!tr.replace(/<[^>]*>/g, " ").includes(UNREADABLE_EN));
  for (const html of [en, tr]) {
    assert.ok(!html.includes("<table"), "no event row is drawn once the log is refused");
  }
});

test("the unavailable state is named in each language as well", async () => {
  stubApi({ "/api/v1/admin/audit": { status: 503, body: {} } });
  const en = renderToStaticMarkup(createElement(FixedLocale, { locale: "en" }, await AdminAudit()));
  stubApi({ "/api/v1/admin/audit": { status: 503, body: {} } });
  const tr = renderToStaticMarkup(createElement(FixedLocale, { locale: "tr" }, await AdminAudit()));
  assert.ok(en.includes("The audit log is temporarily unavailable."));
  assert.ok(tr.includes(">Denetim kaydı geçici olarak kullanılamıyor.<"));
});
