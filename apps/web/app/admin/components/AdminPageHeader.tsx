/**
 * The top of every administrative screen: eyebrow, title, prose, and a place for the one or two
 * things this page can do.
 *
 * Seven screens grew a private `Legend` and `Heading` each, and every one of them carried a
 * comment counting the copies and saying the Rule of Three had been passed. This is the
 * extraction those comments were waiting for. It is a presentation component and nothing else:
 * it takes no session, reads nothing, and decides nothing about what a reader may see.
 *
 * `description` is prose and is capped at `74ch` here rather than at each call site, which is
 * what kept drifting — some pages capped at `68ch`, some at `74ch`, and the two were never a
 * decision anybody made.
 *
 * `actions` is a slot rather than a list of buttons, because the three things that sit there are
 * not the same kind of thing: a link that reloads the page, a link that leaves for another
 * surface, and — on the accounts screen — a disclosure that opens a form. Wrapping them in a
 * common shape would mean inventing one.
 *
 * **The actions sit on the title's own line, not above it.** They were first placed in a row
 * that ran from the eyebrow to the far edge, which on a 1440px screen left `Refresh` hanging
 * eleven hundred pixels from the thing it refreshes and vertically level with a 11px label —
 * it read as a control belonging to the page frame rather than to this page. Aligned to the
 * baseline of the title it reads as part of the same statement, and the description then runs
 * its full measure underneath both.
 *
 * It is a grid rather than two nested rows so that the source order and the visual order can
 * differ. The actions come last in the markup — after the title and after the prose that
 * explains it, which is the order somebody reading the page aloud wants — and are lifted into
 * the title's row on `sm` and up by explicit placement. Stacked in a row-per-line at phone
 * widths they therefore fall below the description instead of wedging between the heading and
 * the sentence that explains it, which is where a flex reordering of the same markup put them.
 *
 * The heading is an `<h2>`. The product's name in the workspace header is the `<h1>`, and this
 * surface has no header of its own, so the page title is the second level for the same reason it
 * was before this component existed.
 */

export function AdminPageHeader({
  legend = "Administration",
  title,
  description,
  actions,
}: {
  legend?: string;
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="grid border-b border-hair pb-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-x-8">
      <p className="text-[11px] font-medium tracking-[0.16em] text-accent uppercase sm:col-start-1 sm:row-start-1">
        {legend}
      </p>

      <h2 className="mt-1.5 min-w-0 text-2xl font-semibold tracking-[-0.02em] text-bone sm:col-start-1 sm:row-start-2 sm:text-[28px]">
        {title}
      </h2>

      {description !== undefined && (
        <div className="mt-2.5 max-w-[74ch] space-y-2.5 text-[13px] leading-relaxed text-muted sm:col-start-1 sm:row-start-3">
          {description}
        </div>
      )}

      {/* Last in the markup, and on the title's line from `sm` up. `self-baseline` sits it on
          the heading's baseline rather than centring it against a 28px cap height, which is what
          makes it read as attached to the title instead of floating beside it. */}
      {actions !== undefined && (
        <div className="mt-4 flex shrink-0 flex-wrap items-center gap-2 sm:col-start-2 sm:row-start-2 sm:mt-0 sm:self-baseline">
          {actions}
        </div>
      )}
    </div>
  );
}
