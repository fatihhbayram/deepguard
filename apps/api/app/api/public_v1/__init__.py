"""The external B2B API.

Separate from `app.api.analyses` because it answers to a different caller. That one serves
the dashboard, which runs inside the trust boundary, authenticates nobody and sees every
analysis; this one serves customers, who authenticate with an API key and must see only
what that key submitted.

Kept under `/api/public/v1` rather than versioned alongside the internal routes, so the
two contracts can move independently: the dashboard's response models change whenever the
dashboard needs them to, and a public contract that shared them would change underneath
paying integrations every time it did.
"""
