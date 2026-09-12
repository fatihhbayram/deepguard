/**
 * The raised plate every administrative screen is built out of: a card with a border, an
 * optional heading and optional prose, and a body.
 *
 * Extracted for the same reason `AdminPageHeader` was — `rounded-lg border border-line bg-ink-2`
 * was written out by hand on every card of all seven screens, and the padding, the radius and
 * the heading size had each drifted at least once. The radius is `rounded-md` here, which is the
 * console's radius as `docs/ui-guidance.md` states it; `rounded-lg` at the call sites was the
 * drift, not the intention.
 *
 * **`bleed` is the table case.** A card wrapping a form wants `p-5`; a card wrapping a table
 * wants no padding at all, because the cells carry their own and a padded plate would inset the
 * row separators from the card's edge. Rather than two components, the body's padding is a flag —
 * and when it is set, the heading keeps its padding and is separated from the body by a hairline,
 * so a titled table still reads as one plate.
 *
 * **`scroll` is the responsiveness rule**, stated once. A body that is a table is wrapped in
 * `overflow-x-auto` so a narrow viewport scrolls the table instead of breaking around it. The
 * table itself carries the `min-w-[…]` that makes the scroll happen; this component cannot know
 * how wide a given table's columns need to be.
 *
 * `children` is optional, for the one card that is only a heading and a paragraph: the "this is
 * your own account" panel on `/admin/users/[id]`, which exists to say why there is no control
 * rather than to hold one. Without this the card would close with the body's bottom padding
 * under nothing.
 *
 * **`contain: paint` is load-bearing, not decoration.** Without it a card whose body is a
 * horizontally scrolling table leaks that table's width into the *document's* scrollable area:
 * the card itself clips correctly and nothing is drawn outside it, but `<html>` becomes 1008px
 * wide on a 390px phone and the whole page swipes sideways into 600px of empty ground. It was
 * reproduced on `/admin/audit` and `/admin` at 390px and 820px in a real browser, and
 * `overflow: clip` on this box does not fix it,
 * because the leak is in the scrollable overflow region rather than in the paint. `contain:
 * paint` is the declaration that actually says what this box means: nothing inside it paints
 * outside it and nothing inside it contributes overflow to anything outside it. `overflow-hidden`
 * stays beside it because that is what rounds the corners off a square-cornered table.
 *
 * `tone="accent"` exists for exactly one card — the human review on `/admin/analyses/[id]`,
 * which is the only plate in this surface that has to be visibly a different kind of statement
 * from the one above it. It is a hairline in `accent` and a barely-there wash, not a highlight.
 */

export function AdminSection({
  title,
  description,
  actions,
  bleed = false,
  scroll = false,
  tone = "default",
  children,
}: {
  title?: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  bleed?: boolean;
  scroll?: boolean;
  tone?: "default" | "accent";
  children?: React.ReactNode;
}) {
  const plate =
    tone === "accent"
      ? "border-accent/30 bg-accent/[0.03]"
      : "border-line bg-ink-2";

  const hasHeading = title !== undefined || description !== undefined || actions !== undefined;

  const body =
    children === undefined ? null : scroll ? (
      <div className="overflow-x-auto">{children}</div>
    ) : (
      children
    );

  return (
    <section className={`overflow-hidden [contain:paint] rounded-md border ${plate}`}>
      {hasHeading && (
        <div
          className={`flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-start sm:justify-between sm:gap-6 ${
            bleed ? "border-b border-hair" : ""
          }`}
        >
          <div className="min-w-0">
            {title !== undefined && (
              <h3 className="text-[15px] font-semibold text-bone">{title}</h3>
            )}
            {description !== undefined && (
              <div className="mt-2 max-w-[74ch] space-y-2 text-[13px] leading-relaxed text-muted">
                {description}
              </div>
            )}
          </div>
          {actions !== undefined && (
            <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>
          )}
        </div>
      )}

      {/* A heading and a padded body both carry `px-5 py-4`, so the body's top padding is
          dropped when a heading already spent it — otherwise every titled card would open with
          a 32px gap nobody asked for. */}
      {body === null ? null : bleed ? (
        body
      ) : (
        <div className={hasHeading ? "px-5 pb-5" : "p-5"}>{body}</div>
      )}
    </section>
  );
}
