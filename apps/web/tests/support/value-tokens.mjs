// Record values in the order a rendered document's text holds them, wherever they sit (R16-T5).
//
// `renderToStaticMarkup` merges a `<Copy>` sentence and the values in its slots into one text
// node, so comparing exact text nodes cannot see a host, a rule id or a time drawn inside a
// sentence. This scans the text for every value as a whole token instead — longest first, and
// never as part of a longer word — which finds them inside sentences as well as on their own.
//
// A bare word (`provenance`, `SUCCESS`) cannot be told from the same word in prose, and a value
// under three characters (`0`, `12`) cannot be told from a figure in prose; standing alone each
// is a text node of its own, which an exact-node comparison already holds.

/** `values` as the markup spells them (HTML-escaped). */
export function valueTokens(html, values) {
  const text = html
    .split(/<[^>]*>/)
    .filter((node) => node.length > 0)
    .join("\u0000");
  const pattern = [...values]
    .filter((value) => value.length >= 3 && !/^\p{L}+$/u.test(value))
    .sort((a, b) => b.length - a.length)
    .map((value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  if (pattern === "") return [];
  return [...text.matchAll(new RegExp(`(?<![\\p{L}\\p{N}_.])(?:${pattern})(?![\\p{L}\\p{N}_])`, "gu"))].map(
    (match) => match[0],
  );
}
