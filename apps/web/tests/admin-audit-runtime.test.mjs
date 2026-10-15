// The audit log reads review events (fix for the CONFIRMED FUNCTIONAL BUG found in the R16 admin
// localization remediation).
//
// `app/api/admin_analyses.py` (`_review_changes`) writes every REVIEW_CREATED / REVIEW_UPDATED
// event with `changes: {"note_changed": <bool>, …}`: a bare boolean beside the `{old, new}` pairs.
// `parseChanges` used to accept only pairs, so one review event made the whole log unreadable.
// The backend is unchanged; the console now accepts `note_changed` — that key, as a boolean, and
// nothing else in that shape — as a flag beside the diff. It is a statement about the note, not a
// transition, so it is never drawn as `false → true`. Every other malformed entry is still refused.

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
const { AUDIT, MALFORMED_EVENT, REVIEW_CREATED_EVENT, REVIEW_UPDATED_EVENT, stubApi } = await import(
  "./support/admin-fixtures.mjs"
);

process.env.API_INTERNAL_URL = "http://api.invalid";

const UNREADABLE_EN = "The audit log could not be read.";
const UNREADABLE_TR = "Denetim kaydı okunamadı.";

const plain = (value) => JSON.parse(JSON.stringify(value));

test("the backend still writes `note_changed` as a bare boolean (unchanged by this fix)", () => {
  // Read, not imported: the API is not part of this test run and is not changed by this task.
  const source = readFileSync(new URL("../../api/app/api/admin_analyses.py", import.meta.url), "utf8");
  assert.match(source, /changes: dict = \{"note_changed": note_moved\}/);
});

test("`note_changed: true|false` is a flag, not a diff", () => {
  for (const value of [true, false]) {
    const parsed = parseChanges({ note_changed: value });
    assert.notEqual(parsed, null, String(value));
    assert.deepEqual(plain(parsed), { changes: {}, flags: { note_changed: value } });
    // No `{old, new}` pair is invented for it.
    assert.equal(Object.hasOwn(parsed.changes, "note_changed"), false);
  }
});

test("a review event parses whole: its flag and its field changes side by side", () => {
  const event = parseEvent(REVIEW_UPDATED_EVENT);
  assert.notEqual(event, null);
  assert.deepEqual(plain(event.changes), { status: { old: "UNREVIEWED", new: "REVIEWED" } });
  assert.deepEqual(plain(event.flags), { note_changed: false });
  assert.deepEqual(plain(parseEvent(REVIEW_CREATED_EVENT).flags), { note_changed: true });
});

test("a normal {old, new} event is unchanged, with no flags", () => {
  const parsed = parseChanges({ is_active: { old: true, new: false }, role: { old: null, new: "ADMIN" } });
  assert.deepEqual(plain(parsed), {
    changes: { is_active: { old: true, new: false }, role: { old: null, new: "ADMIN" } },
    flags: {},
  });
  assert.deepEqual(plain(parseEvent(AUDIT[1]).changes), plain(AUDIT[1].changes));
});

test("every other malformed entry is still refused", () => {
  for (const changes of [
    { is_active: 0 },
    { is_active: [] },
    { is_active: null },
    { is_active: true }, // a boolean under any key but `note_changed`
    { is_active: { old: true } }, // half a pair
    { note_changed: 1 },
    { note_changed: "true" },
    { note_changed: null },
    { note_changed: [true] },
    { Note_Changed: true },
  ]) {
    assert.equal(parseChanges(changes), null, JSON.stringify(changes));
  }
  for (const changes of [null, [], 0, "note_changed", true]) {
    assert.equal(parseChanges(changes), null, JSON.stringify(changes));
  }
  assert.equal(parseEvent(MALFORMED_EVENT), null);
});

test("a well-formed pair is read as a pair under any key, `note_changed` included, as before", () => {
  // Backward compatible: the special case widens what is accepted and narrows nothing.
  assert.deepEqual(plain(parseChanges({ note_changed: { old: false, new: true } })), {
    changes: { note_changed: { old: false, new: true } },
    flags: {},
  });
});

test("prototype-named keys stay own data and cannot reach Object.prototype", () => {
  // As JSON.parse delivers them: `__proto__` and `constructor` as own keys of the payload.
  const payload = JSON.parse(
    '{"__proto__": {"old": 1, "new": 2}, "constructor": {"old": "a", "new": "b"}, "note_changed": true}',
  );
  const parsed = parseChanges(payload);
  assert.notEqual(parsed, null);
  assert.deepEqual(Object.keys(parsed.changes).sort(), ["__proto__", "constructor"]);
  assert.deepEqual({ ...parsed.changes.__proto__ }, { old: 1, new: 2 });
  assert.equal(Object.getPrototypeOf(parsed.changes), null);
  assert.equal(Object.getPrototypeOf(parsed.flags), null);
  assert.equal({}.old, undefined, "Object.prototype was not polluted");
  // A flag only ever comes from its own exact key — never from an inherited or look-alike one.
  for (const key of ["constructor", "__proto__", "toString", "hasOwnProperty"]) {
    assert.equal(parseChanges(JSON.parse(`{"${key}": true}`)), null, key);
  }
  assert.equal(parseChanges({}).flags.note_changed, undefined);
});

test("a mixed list — review events among ordinary ones — reads in full and in order", async () => {
  const list = [REVIEW_CREATED_EVENT, ...AUDIT, REVIEW_UPDATED_EVENT];
  stubApi({ "/api/v1/admin/audit": list });
  const result = await fetchAuditEvents();
  assert.equal(result.ok, true);
  assert.deepEqual(
    result.events.map((event) => event.id),
    list.map((event) => event.id),
  );
});

test("one malformed event still makes the log unreadable, as before", async () => {
  stubApi({ "/api/v1/admin/audit": [...AUDIT, REVIEW_UPDATED_EVENT, MALFORMED_EVENT] });
  assert.deepEqual(await fetchAuditEvents(), { ok: false, unauthenticated: false, error: UNREADABLE_EN });
});

async function auditPage(locale, events) {
  stubApi({ "/api/v1/admin/audit": events });
  return renderToStaticMarkup(createElement(FixedLocale, { locale }, await AdminAudit()));
}

const visible = (html) => html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");

test("the page draws review events: the flag as a sentence, the pairs as diffs", async () => {
  const list = [REVIEW_CREATED_EVENT, ...AUDIT, REVIEW_UPDATED_EVENT];
  const tr = await auditPage("tr", list);
  const en = await auditPage("en", list);

  for (const [html, to] of [[tr, "yeni değer"], [en, "to"]]) {
    assert.ok(html.includes("<table"), "the rows are drawn");
    assert.ok(!visible(html).includes(UNREADABLE_EN) && !visible(html).includes(UNREADABLE_TR));
    // The status pair is drawn as before.
    assert.ok(visible(html).includes(`UNREVIEWED → ${to} REVIEWED`));
    // The flag's row holds a sentence and no transition: no arrow, no `true`/`false`.
    const rows = [...html.matchAll(/note_changed<\/dt>(.*?)<\/div>/g)].map((match) => match[1]);
    assert.equal(rows.length, 2);
    for (const row of rows) assert.ok(!row.includes("→") && !/\b(true|false)\b/.test(visible(row)), row);
  }
  assert.ok(tr.includes('<span lang="tr">Not değiştirildi</span>'));
  assert.ok(tr.includes('<span lang="tr">Not değiştirilmedi</span>'));
  assert.ok(en.includes('<span lang="en">Note changed</span>'));
  assert.ok(en.includes('<span lang="en">Note not changed</span>'));
  // The key is drawn raw and English-cased, like every field name in the diff.
  assert.ok(tr.includes('<dt lang="en" class="font-mono text-[11px] tracking-[0.06em] text-muted uppercase">note_changed</dt>'));
  // Every row of the ordinary events is still there.
  for (const event of AUDIT) assert.ok(tr.includes(event.created_at) && tr.includes(event.target_id), event.id);
});

test("a malformed event still shows the localized error, with no rows", async () => {
  const en = await auditPage("en", [...AUDIT, MALFORMED_EVENT]);
  const tr = await auditPage("tr", [...AUDIT, MALFORMED_EVENT]);
  assert.ok(en.includes(UNREADABLE_EN));
  assert.ok(tr.includes(`<span lang="tr" data-canonical="${UNREADABLE_EN}">${UNREADABLE_TR}</span>`));
  for (const html of [en, tr]) assert.ok(!html.includes("<table"));
});

test("the unavailable state is named in each language as well", async () => {
  const en = await auditPage("en", { status: 503, body: {} });
  const tr = await auditPage("tr", { status: 503, body: {} });
  assert.ok(en.includes("The audit log is temporarily unavailable."));
  assert.ok(tr.includes(">Denetim kaydı geçici olarak kullanılamıyor.<"));
});
