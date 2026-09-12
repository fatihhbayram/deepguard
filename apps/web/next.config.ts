import type { NextConfig } from "next";

// The transport ceiling for a request body reaching this server, in bytes.
//
// Not a media limit. The API owns that question and answers it alone: `MAX_UPLOAD_BYTES` in
// `apps/api/app/media.py` is 100 MiB, and it is the only place a file is measured against a
// product limit. This number exists one layer below that, to make sure a submission the API
// is willing to consider can physically arrive — a 100 MiB file plus the multipart framing
// around it, with 10 MiB of headroom so the envelope is never what fails.
//
// It is deliberately larger than the media limit rather than equal to it. A ceiling set at
// exactly 100 MiB would reject a 100 MiB file, because the boundary markers, the headers for
// each part and the trailing delimiter all ride in the same body — the request is always
// bigger than the file inside it. Sizing this to the file alone would move the effective
// media limit below the documented one and put it in a second place, which is the failure
// this constant exists to avoid.
// **Known limitation: this ceiling is a memory bound, not just a size check.**
// `request.formData()` in the upload route buffers the whole body before it can read a single
// field, so a submission in flight is held in this container's heap — and the container is
// capped at 1 GiB (see `deploy.resources.limits` on the `web` service). The design is
// therefore bounded but memory-sensitive, because multipart bodies are buffered in the web
// process. That is what makes this number load-bearing rather than cosmetic, and why it is
// bounded at all rather than simply raised out of the way.
//
// No safe concurrency figure is stated here on purpose: it would depend on the runtime's own
// allocation behaviour and on everything else the process is holding at the time, and a
// number in a comment would be read as a guarantee. Raising this ceiling, or admitting more
// concurrent uploads, needs the upload to stop being buffered first — streaming the body
// through to the API, or handing the browser a pre-signed destination and keeping the bytes
// out of this process entirely. Both are re-architecture, deliberately out of scope for R7,
// and neither is required for the 100 MiB the product actually promises.
const UPLOAD_TRANSPORT_CEILING_BYTES = 110 * 1024 * 1024;

const nextConfig: NextConfig = {
  // Where the reports used to be (R8-T1).
  //
  // `/report/[id]` was the report's address for the whole of R7, and those addresses are in
  // the places a forensic report ends up: a case file, a mail thread, somebody's bookmarks.
  // Moving the route under `/app` without this would turn every one of them into a 404 —
  // silently, and only for the readers who kept a link rather than navigating from the
  // dashboard.
  //
  // Stated here rather than as a server component at the old path that calls `redirect()`.
  // Both work; this one is checked before the filesystem, so the answer is one 308 with no
  // React render behind it, and it leaves nothing under `/report` for a later reader to
  // mistake for a live route.
  //
  // `permanent: true` — 308, not 301. The status preserves the request method, and it tells
  // a client that kept the old address to stop asking for it. The move is not provisional.
  redirects() {
    return [
      {
        source: "/report/:id",
        destination: "/app/report/:id",
        permanent: true,
      },
    ];
  },
  experimental: {
    // Next.js caps the request body it will carry through middleware, and this application
    // runs middleware on every path (`middleware.ts` matches everything but static assets),
    // so the cap applies to the upload route. The default is 10 MiB, which silently made the
    // effective upload limit a tenth of the documented one (R7-CLOSURE-FIX-2).
    //
    // `proxyClientMaxBodySize`, never `middlewareClientMaxBodySize`. The latter is the older
    // name for the same setting, is deprecated in Next 16, and refuses to start when both are
    // present. It is also the name Next's own over-limit warning still links to, so following
    // that message leads to the deprecated option — the link is stale, not authoritative.
    //
    // Stated in bytes rather than as `"110mb"`. The string form is parsed by a library that
    // reads `mb` as 2^20, which is what is wanted here but is not what the suffix says, and a
    // limit that has to be read twice to be believed is worth writing as arithmetic.
    proxyClientMaxBodySize: UPLOAD_TRANSPORT_CEILING_BYTES,
  },
};

export default nextConfig;
