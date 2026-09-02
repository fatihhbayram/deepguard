"""The two endpoints a B2B customer integrates against: submit media, poll for the result.

Every route here is behind an API key, applied to the router rather than to each function,
so a route added later is authenticated by construction instead of by whoever remembers to
decorate it.

Isolation is the other half. A key sees the analyses that key submitted and nothing else:
the dashboard's uploads are owned by nobody, another customer's are owned by another key,
and both are simply absent from the filtered read — so they come back as `404`, the same
answer an id that never existed gets. Distinguishing them would confirm to one customer
that another customer's analysis id is real.

Nothing here recomputes anything. The status, the risk decision and the signals are read
off the rows the worker committed; a read path that evaluated a rule of its own could
answer differently from the record it is supposed to be reporting.

Submission is capped per key (P9-T3): one customer may have `MAX_ACTIVE_ANALYSES` analyses
in flight at a time, so no single integration can fill a queue that everyone shares. The
cap is enforced in PostgreSQL, inside the transaction that creates the analysis, and only
on submission — reads are not throttled.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.analyses import (
    ActiveAnalysisLimitReached,
    AnalysisSummary,
    accept_upload,
    analysis_evidence_select,
    analysis_payloads,
)
from app.api.url_analyses import UrlSubmission, accept_url
from app.auth import ApiKeyPrincipal, require_api_key
from app.db.models import Analysis
from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/public/v1",
    tags=["public"],
    # On the router, not on the routes. Authentication is the property of this whole
    # surface, and mounting it per-function would make an unauthenticated public endpoint
    # one forgotten line away.
    dependencies=[Depends(require_api_key)],
)

# How much work one API key may have in flight at once. A concurrency limit, not a quota:
# it bounds how many analyses a customer can have outstanding at any moment, and says
# nothing about how many they may run in a day. Finished analyses leave the count on their
# own, so a key that submits five and waits is never worse off than one that submits five
# and polls.
#
# Five because the queue is served by one worker against real detector calls, and a single
# customer filling it would starve every other one. It is deliberately a plain constant:
# per-key limits are a tier model, and DeepGuard has no tiers, no plans and no billing.
MAX_ACTIVE_ANALYSES = 5


class QueuedAnalysis(BaseModel):
    """The answer to a submission: what was created, and that it has not run yet.

    Two fields, and deliberately no more. The id is what the customer polls with, and
    `status` is always `queued` here — an accepted upload has been stored, probed and put
    on the queue, and no detector has looked at it. The storage keys, the content hash and
    the probed media facts the internal route reports are all real, and all internal.
    """

    id: uuid.UUID
    status: str


class PublicSignal(BaseModel):
    """One detector's answer about the media, as that detector left it.

    Kept as a list of independent signals rather than merged into a single number
    (AGENTS.md rule 11): the providers measure different things on different scales, and an
    average over them would be a figure no detector reported.

    `status` separates the states that must never be read alike — a provider that answered,
    one that failed, and one that timed out. `score` is the provider's own figure on the
    provider's own scale and is null for every signal that has none, which is most of them:
    only NVIDIA's synthetic-video detector reports a file-level number at all. Null is not
    zero and not a low risk; it means this detector produced no such figure.
    """

    provider: str
    signal_type: str
    status: str
    provider_version: str | None
    score: float | None


class PublicAnalysis(BaseModel):
    """One analysis as its submitting customer gets it.

    Its own model rather than the dashboard's `AnalysisSummary`, which is wider: that one
    carries the probed container facts and the per-clip evidence the internal UI renders,
    and it changes whenever that UI needs it to. This is a contract someone integrates
    against, so it is narrow and it moves only on purpose.
    """

    id: uuid.UUID
    # `queued`, `completed` or `failed`. `failed` means DeepGuard could not complete the
    # analysis — never that the media was found to be fake.
    status: str
    created_at: datetime

    # The decision the risk engine committed, and the trace that makes it readable later:
    # which ruleset was in force, which single rule fired, and which calibration its
    # thresholds came from.
    #
    # All four are null while an analysis is queued or being worked on — no decision has
    # been taken. That is not the same as `risk_level = "UNKNOWN"`, which is a decision: the
    # engine ran, a rule fired, and the answer is that the evidence supports no
    # classification. A client must not fold the two together.
    risk_level: str | None
    risk_rules_version: str | None
    risk_rule_id: str | None
    risk_calibration_id: str | None

    # Every signal the analysis actually carries, in a stable order. Empty while nothing has
    # run yet, and short of the full set for an analysis that predates a detector or whose
    # provider never answered — the absence of a signal is itself a fact, and nothing is
    # substituted in to fill the gap.
    signals: list[PublicSignal]


def public_signals(summary: AnalysisSummary) -> list[PublicSignal]:
    """Project the stored signals onto the public shape, dropping the ones that are absent.

    Only the fields named here cross the boundary. The internal models also carry each
    provider's stored metadata document, which on a failure holds the provider exception's
    class name — internal diagnostic detail that has no business in a customer's response.

    Ordered by signal type so the list reads the same way on every poll. A client that
    indexes into it by position is doing the wrong thing either way, but an order that
    shifted between two polls of the same unchanged analysis would look like the analysis
    had changed.
    """
    signals = [
        summary.synthetic_video,
        summary.audio_authenticity,
        summary.active_speaker,
        summary.face_manipulation,
        summary.provenance,
    ]

    return sorted(
        (
            PublicSignal(
                provider=signal.provider,
                signal_type=signal.signal_type,
                status=signal.status,
                provider_version=signal.provider_version,
                # Two of them carry a score at all — the synthetic-video model and the
                # face-manipulation classifier — and each is that provider's own figure on
                # its own scale. The rest have no such field, and `None` here says so rather
                # than inventing one. The two scores are never comparable with each other and
                # nothing here relates either to `risk_level`: only the first is calibrated,
                # and it is the only one a rule reads (`app.risk_engine`).
                score=getattr(signal, "score", None),
            )
            for signal in signals
            if signal is not None
        ),
        key=lambda signal: (signal.signal_type, signal.provider),
    )


def limit_reached(
    api_key_id: uuid.UUID, limit: ActiveAnalysisLimitReached
) -> HTTPException:
    """The `429` a key at its concurrency ceiling gets, whichever door it submitted through.

    One copy for both submission routes (P10-T2): a URL submission occupies a slot exactly
    as an upload does, and a second wording of the refusal would let the two answers drift.

    The limit itself is worth telling the caller — it is their own outstanding work, and a
    client that cannot see the ceiling can only guess at it. The count behind it is not: it
    is the same number the caller can already get by polling their own analyses, and echoing
    it here would be one more field to keep true.

    No `Retry-After`. It would have to be a guess — what frees a slot is a detector
    finishing, and nothing here knows when that will be — and a fabricated number is worse
    than none, because a client would trust it.
    """
    logger.info(
        "API key %s is at its concurrent analysis limit (%s active).",
        api_key_id,
        limit.active,
    )

    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"At most {limit.limit} analyses may be in progress at once. "
            "Wait for one to finish."
        ),
    )


def public_analysis(summary: AnalysisSummary) -> PublicAnalysis:
    """The stored record, narrowed to what the public contract promises."""
    return PublicAnalysis(
        id=summary.id,
        status=summary.status,
        created_at=summary.created_at,
        # Read straight off the analysis row, exactly as the worker committed them.
        risk_level=summary.risk_level,
        risk_rules_version=summary.risk_rules_version,
        risk_rule_id=summary.risk_rule_id,
        risk_calibration_id=summary.risk_calibration_id,
        signals=public_signals(summary),
    )


@router.post(
    "/analyses", response_model=QueuedAnalysis, status_code=status.HTTP_202_ACCEPTED
)
async def submit_analysis(
    file: UploadFile,
    principal: ApiKeyPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> QueuedAnalysis:
    """Accept a customer's media and queue it, owned by the key that submitted it.

    The pipeline is the internal one, called rather than reimplemented: the same size
    limit, the same container validation, the same forensic original in MinIO and the same
    queued job. A customer's upload is therefore admitted or refused on exactly the terms a
    dashboard upload is, and a change to those terms cannot apply to one route and miss the
    other.

    The one difference is ownership. The authenticated key is written onto the analysis in
    the transaction that creates it, which is what every read below filters on.

    `require_api_key` is declared here as well as on the router. The router's copy is what
    guards the route; this one is how the handler gets the principal, and FastAPI resolves
    the shared dependency once per request rather than authenticating twice.

    A key may have `MAX_ACTIVE_ANALYSES` analyses outstanding at once, and the sixth is
    refused with `429`. The limit is enforced inside the transaction that writes the
    analysis, under a lock on the key's own row, so concurrent submissions from one key
    cannot slip past it together — see `persist_analysis`. It is a temporary condition and
    says so: the caller's own work finishing is what clears it.

    Failures come out of the pipeline already shaped — `415` for a media type that is not
    accepted, `413` for an oversized body, `422` for bytes that are not usable video, `503`
    when storage or the media processor is down. None of them carries a stack trace, a
    storage key or a statement; those stay in the server log.
    """
    try:
        accepted = await accept_upload(
            file,
            session,
            api_key_id=principal.id,
            max_active_analyses=MAX_ACTIVE_ANALYSES,
        )
    except ActiveAnalysisLimitReached as limit:
        raise limit_reached(principal.id, limit) from None

    return QueuedAnalysis(id=accepted.analysis.id, status=accepted.analysis.status)


@router.post(
    "/analyses/url", response_model=QueuedAnalysis, status_code=status.HTTP_202_ACCEPTED
)
async def submit_url_analysis(
    submission: UrlSubmission,
    principal: ApiKeyPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> QueuedAnalysis:
    """Accept a customer's media URL, download it, and queue it owned by their key.

    The same submission as the route above, with the media arriving by a different door.
    `accept_url` downloads the URL and hands the file to the very pipeline an upload goes
    through, so the size limit, the container validation, the forensic original, the queued
    job, the ownership stamp and the concurrency limit are not restated here — they are the
    same code, reached with the same two arguments.

    Authentication is the router's, as it is for every route on this surface, so this
    endpoint is behind an API key by construction rather than by this function remembering.

    The request waits for the download. A URL that turns out to be a live stream, oversized
    media, an address this server will not fetch or simply unavailable is refused with a
    client error that says which of those it was, and nothing about the extractor, the
    network or the address behind it — see `client_error` in `app.api.url_analyses`.
    """
    try:
        accepted = await accept_url(
            submission.url,
            session,
            api_key_id=principal.id,
            max_active_analyses=MAX_ACTIVE_ANALYSES,
        )
    except ActiveAnalysisLimitReached as limit:
        raise limit_reached(principal.id, limit) from None

    return QueuedAnalysis(id=accepted.analysis.id, status=accepted.analysis.status)


@router.get("/analyses/{analysis_id}", response_model=PublicAnalysis)
def read_analysis(
    analysis_id: uuid.UUID,
    principal: ApiKeyPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
) -> PublicAnalysis:
    """Return one of the caller's own analyses: its status, its risk decision, its signals.

    Ownership is part of the query, not a check on the result. Fetching the row and then
    comparing its owner would work, but it would put the isolation one early `return` away
    from being bypassed; as a `WHERE` clause, a row belonging to another key is never read
    in the first place. Dashboard analyses carry a null owner and no key's id equals null,
    so they are excluded by the same clause without needing a rule of their own.

    Anything the filter excludes is a `404` with the same body an id that was never issued
    gets. That is the point: a customer who guesses another customer's analysis id learns
    nothing from the response about whether it exists.

    Nothing here recomputes the risk level. It is read from the analysis row along with the
    ruleset, rule and calibration it was decided under.

    A malformed id is rejected as a `422` by path validation before any statement runs.
    """
    try:
        rows = session.execute(
            analysis_evidence_select().where(
                Analysis.id == analysis_id,
                Analysis.api_key_id == principal.id,
            )
        ).all()

        payloads = analysis_payloads(session, rows)
    except SQLAlchemyError:
        # Statements, connection strings and driver errors stay in the server log.
        logger.exception("Reading public analysis %s failed.", analysis_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="analyses are temporarily unavailable",
        ) from None

    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="analysis not found",
        )

    # The id is a primary key, so the narrowed select cannot return a second row.
    return public_analysis(payloads[0])
