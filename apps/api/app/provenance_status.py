"""The two axes a provenance reading is reported on, and the only mapping onto them.

C2PA answers a question no detector asks: who signed these bytes, and does that signature
hold. A manipulation verdict answers whether the picture was altered. The two are separate
sources of evidence about the same file, and R9-T6 exists because a single `status` field
was being asked to carry both a fact about the media and a fact about our reading of it —
"this file carries no credentials" and "we could not find out whether it does" arrived as
the same word, and a reader had no way to tell a finding from a gap.

So the state is split in two, and the axes answer different questions:

- `status` — what provenance was found. `PROVENANCE_PRESENT` when a manifest is in the
  file, `UNVERIFIED` when none was found or none could be looked for. Never a third word:
  there is deliberately no `VERIFIED_AUTHENTIC` in this vocabulary, because nothing this
  system reads could produce one. A detector score is not provenance, and a manifest that
  validates says the signer stands behind the bytes as signed, not that what was signed is
  an unaltered record of something that happened;
- `availability` — whether provenance could be evaluated at all. `AVAILABLE` when the read
  ran to completion, `UNAVAILABLE` when it did not.

Reading the two together is what keeps three distinct facts distinct, and the pairing is
the whole point of the split:

    UNVERIFIED / AVAILABLE           we looked, and this file carries no credentials
    PROVENANCE_PRESENT / AVAILABLE   we looked, and a manifest is here
    UNVERIFIED / UNAVAILABLE         we could not look, and nothing is known either way

Three rules this module holds to, all of them the reason it is a module and not a ternary
at a call site:

**Absence is not evidence.** `UNVERIFIED / AVAILABLE` is the ordinary state of almost
every file on the internet. It is not a weak `PROVENANCE_PRESENT` and carries no suspicion
whatsoever.

**Presence is not proof.** `PROVENANCE_PRESENT` means a manifest exists, and it means only
that. Whether its signature validates, who issued it and what the SDK made of it stay
where they were — in the raw C2PA evidence beside this state, reported in the SDK's own
words. An invalid signature still reports as `PROVENANCE_PRESENT` here, because the
manifest is genuinely present; collapsing an invalid one into `UNVERIFIED` would hide a
failed signature behind a word that reads like a routine absence.

**A failed read is not an absence.** `FAILED` and `TIMEOUT` mean the question was never
answered. They are `UNAVAILABLE`, never the `AVAILABLE` state that asserts we looked.

Nothing here takes a manipulation verdict, a detector score or an analysis as an argument.
That is the orthogonality guarantee stated as a signature rather than promised in prose:
there is no input by which a verdict could reach this mapping, and `app.risk_engine` takes
no provenance input by which this state could reach a verdict. The two-way independence is
structural, and `tests/test_provenance_status.py` pins it in both directions.
"""

from dataclasses import dataclass

from app.db.models import SIGNAL_STATUS_SUCCESS

# What provenance was found. Two words, and the vocabulary is closed at two on purpose:
# see the module docstring on why there is no `VERIFIED_AUTHENTIC` for anything to reach.
PROVENANCE_UNVERIFIED = "UNVERIFIED"
PROVENANCE_PRESENT = "PROVENANCE_PRESENT"
PROVENANCE_STATUSES = (PROVENANCE_UNVERIFIED, PROVENANCE_PRESENT)

# Whether provenance could be evaluated at all — a fact about the reading, not the media.
PROVENANCE_AVAILABLE = "AVAILABLE"
PROVENANCE_UNAVAILABLE = "UNAVAILABLE"
PROVENANCE_AVAILABILITIES = (PROVENANCE_AVAILABLE, PROVENANCE_UNAVAILABLE)


@dataclass(frozen=True)
class ProvenanceState:
    """One provenance reading on both axes at once.

    Frozen and returned as a pair rather than as two calls, because the two fields are only
    meaningful together: `UNVERIFIED` alone does not say whether anybody looked, and a caller
    that carried one axis and dropped the other would reintroduce the exact collapse this
    module was added to end.
    """

    status: str
    availability: str


def provenance_state(
    signal_status: str | None, manifest_exists: bool | None
) -> ProvenanceState:
    """Map one persisted C2PA signal onto the two axes. The complete mapping is here.

    `signal_status` is the stored signal status (`SUCCESS`, `FAILED`, `TIMEOUT`) and
    `manifest_exists` is the reader's answer to whether the file carries a manifest, which
    only a successful read has. Both are the persisted values, passed through untouched.

    Four inputs, three outcomes:

    - `SUCCESS` with `manifest_exists` true  -> `PROVENANCE_PRESENT` / `AVAILABLE`
    - `SUCCESS` with `manifest_exists` false -> `UNVERIFIED` / `AVAILABLE`
    - `SUCCESS` with `manifest_exists` null  -> `UNVERIFIED` / `UNAVAILABLE`
    - anything else                          -> `UNVERIFIED` / `UNAVAILABLE`

    The third line is the one worth stating outright. A successful read whose stored
    metadata does not say whether a manifest was found is a reading we cannot interpret,
    and the honest report of it is that provenance could not be evaluated — not that the
    file carries nothing, which is an assertion the record does not support.

    The last line is a deliberate default rather than an oversight. `FAILED` and `TIMEOUT`
    land there by name, and so does any status this function has never heard of: a status
    added later and not yet mapped must degrade to "could not be evaluated", never to a
    finding about the media.
    """
    if signal_status == SIGNAL_STATUS_SUCCESS and manifest_exists is not None:
        return ProvenanceState(
            status=PROVENANCE_PRESENT if manifest_exists else PROVENANCE_UNVERIFIED,
            availability=PROVENANCE_AVAILABLE,
        )

    return ProvenanceState(
        status=PROVENANCE_UNVERIFIED, availability=PROVENANCE_UNAVAILABLE
    )
