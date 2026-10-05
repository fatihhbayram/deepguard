// `next/headers` for a render outside a Next.js request: a signed-in session cookie and no
// request id. The values are placeholders; the API they would be sent to is `fetch`, which the
// test replaces as well.

export async function cookies() {
  return { get: (name) => ({ name, value: "test-session" }) };
}

export async function headers() {
  return new Headers();
}
