# Admin console UI guidance

The rules the `/admin` surface is built to, written down because seven screens were built one
at a time and each one restated the last one's primitives from memory. This file is the memory
now. It covers the administrative surface only — `/app` is a different instrument and the
evidence report is a different material again (see the comment at the top of `globals.css`).

## 1. What this surface is

An operations console for one operator. Dense, legible under long viewing, and built out of
server-rendered HTML forms. **It is not a dashboard product**: nothing here animates, nothing
floats, and no number on it is computed in the browser.

## 2. Tokens

Only the tokens declared in `apps/web/app/globals.css` are used. No raw hex, no `slate-*`, no
new colour introduced at a call site.

| Token | Role |
| --- | --- |
| `ink` | the ground the console sits on |
| `ink-2` | the raised plate: every card, every table body |
| `bone` | foreground — values, headings, what the reader came to read |
| `muted` | labels, captions, secondary timestamps |
| `line` | the visible border of a plate |
| `hair` | the hairline *inside* a plate: row separators, card-header rules |
| `chip` | the neutral badge ground |
| `rule` | the hover weight a `line` border moves to |
| `accent` | **the system speaking only** — focus, the eyebrow legend, the current nav item |

`accent` is never used to mean a forensic outcome. Colour on evidence belongs to `RISK_STYLES`
and to the semantic badge tones below.

Semantic tones, and the only four that exist:

- **neutral** `bg-chip text-bone` — a normal, working state (`Active`, `completed`)
- **muted** `bg-chip text-muted` — a state with no opinion in it (`queued`, `UNREVIEWED`)
- **warning** `amber-500/10` / `amber-200` — attention, not failure (`Stale`, `NEEDS_FOLLOW_UP`)
- **negative** `rose-500/10` / `rose-200` — ended or failed (`Deactivated`, `REVOKED`, `failed`)
- **positive** `emerald-500/10` / `emerald-200` — an affirmative human act (`REVIEWED`)

## 3. Spacing

A 4px scale, used tightly. Tailwind's `1` is 4px, so the console's whole vocabulary is
`1 / 1.5 / 2 / 3 / 4 / 5 / 6 / 8`, and nothing between.

- Page gutter: `px-4 sm:px-6 lg:px-8`, vertical `py-8`.
- Card padding: `p-5` for a card holding a form or prose; `p-4` for a card that is a list of
  figures. Never `p-8` — a console is not a marketing page.
- Card internals: `mt-2` under a heading, `mt-4` before the first control.
- Between stacked cards: `space-y-5`; between cards in a grid, `gap-4` with `items-start` so a
  short card stops where its content stops instead of stretching to its neighbour.
- Table cells: `px-5 py-3` for headers, `px-5 py-4` for rows.
- Sidebar rail: `w-60`, items `py-1.5 pl-3 pr-2`.

## 4. Depth

**Borders only.** `border border-line rounded-md` is the whole of how a plate is separated from
its ground. No `shadow-*`, no gradient, no blur, no glass, no glow. A drop shadow on a dark
ground reads as smudge, and the console has to stay legible at low brightness on a laptop in an
office.

Radii are sharp: `rounded-md` for plates, inputs and buttons; `rounded-sm` for badges;
`rounded-full` only for the 6px status dots and the analytics bar tracks.

## 5. Type

- Page title: `text-2xl font-semibold tracking-[-0.02em]`, `sm:text-[28px]`.
- Card title: `text-[15px] font-semibold`.
- Body and prose: `text-[13px] leading-relaxed`, capped at `max-w-[74ch]`.
- Labels, column headings, eyebrow: `text-[11px] tracking-[0.08em] uppercase` (`0.16em` on the
  accented eyebrow).
- **Every machine value is `font-mono`**: ids, addresses, timestamps, hashes, counts in a table.
  Prose is never mono, and a name is not a machine value. This is the rule that lets an operator
  tell at a glance which strings they can safely compare character by character.

## 6. Domain wording is never softened

`FAILED`, `REVOKED`, `UNREVIEWED`, `NEEDS_FOLLOW_UP`, `UNDECIDED` are printed as the API spells
them. A badge may colour a word; it may not translate it, title-case it, or replace it with a
friendlier one. A status this application does not recognise is printed as it arrived rather
than mapped to a guess — the day the vocabulary grows, that is what says so.

Timestamps are shown as the API stored them. Nothing is reformatted into the reader's locale,
because the record does not hold a local rendering and a browser with a wrong clock must not be
able to change what a page says.

## 7. Tables

Every table scrolls horizontally rather than reflowing. The wrapper is
`overflow-x-auto` and the table carries a `min-w-[…]` wide enough for its columns.

This is deliberate and it replaces the per-screen `sm:` grid collapse the earlier pages used. A
row that restacks into a card on a narrow screen loses its column alignment, which is the whole
reason an operator is scanning a table; a row that scrolls keeps it. The console is a desktop
instrument that must not *break* on a phone, not a phone product.

## 8. Controls

Plain HTML forms, posting to route handlers. A client component is written only when the server
cannot do the job at all — today that is exactly two: `create-key.tsx` (the response carries a
secret that cannot travel through a redirect's query string) and `AdminMobileNav` (a drawer has
an open state and the server has none).

`AdminMobileNav` holds the open boolean and nothing else. Navigation items, the current path
and anything derived from the session are rendered on the server and handed to it as children.
A client component that knew the route table would be a second copy of it.

A control that cannot be honoured is **not drawn**, not drawn disabled. A disabled button
invites the reader to work out how to enable it; a sentence saying why there is nothing there
answers them.

## 9. Figures

Two shapes, and they are not interchangeable.

**The headline strip** is for the two or three numbers somebody opens a screen to read. Tiles on
one row, separated by hairlines rather than each being its own bordered card — `grid gap-px` over
a `bg-line` ground, each tile `bg-ink-2`. The primary figure is `text-[26px]`, the rest
`text-[20px]`, all `font-mono tabular-nums text-bone`. No tile is ever a computed number: it is a
count the API reported, read straight out of the payload.

**A distribution** is a list of categories with their counts. One line each, on a
`[label | track | value]` grid so every bar starts and ends at the same two x-positions and the
figures align right. The track is `h-1` in `bg-plate` with an `bg-accent/70` fill; the fill's
width is a count over the largest count *in that card*, never over a total — a share of a total is
a percentage, and a percentage is a claim the API has not made. The count is always written out,
so a card where one category dwarfs the rest stays readable when every other bar is a sliver. The
fill is `aria-hidden`: the reading is the figure.

Do not shade the whole row by magnitude. It was tried; a partial wash reads as a highlighted row
and a full one reads as a selected row, and neither is a meaning these cards have.

## 10. Hierarchy

Not every plate carries the same weight, and the type says which is which:

- A **section** title is `text-[15px] font-semibold text-bone`.
- A **figure card** title is `text-[11px] tracking-[0.12em] uppercase text-muted` — a label on a
  list of numbers, not a heading competing with the page title.
- The headline strip has no title at all; its tiles are labelled individually.

## 11. The shell

`app/admin/layout.tsx` owns the guard, the sidebar and the `<main>` wrapper. Pages return their
content and own neither their page gutter nor a navigation strip of their own — the cross-links
that used to sit at the foot of every screen are the sidebar now.

The current item is marked by `AdminNavLink`: a `bg-plate` ground, the foreground lifted from
`muted` to `bone`, and a 2px `accent` rule down the left edge, plus `aria-current="page"`. Not an
outline and not a `chip` ground — both read as a button the reader is meant to press for some
further effect, and what the marker says is only "you are here". Hover is `bg-plate/60`.

Navigation is grouped by what the operator is doing, not by what the API is:

- **Operations** — Accounts, Detection jobs, Operational summary
- **Governance** — Audit log
- **Access** — API keys

Case review (`/admin/analyses/[id]`) and the account detail (`/admin/users/[id]`) are not in the
sidebar and must not be: neither has a listing to land on, and the way into both is a row that
already names the record.
