"""Submitting media by URL, into the pipeline an uploaded file already goes through.

A URL is a second door, not a second kind of analysis. The media behind it is downloaded to
a temporary file and that file is handed to `accept_upload` — the same size limit, the same
container validation, the same forensic original in MinIO, the same queued job and the same
row shape. Nothing downstream learns that a URL was involved: the worker, the detectors, the
risk engine and both read paths see an analysis, and the database schema is untouched.

Its own module rather than another section of `app/api/analyses.py`, for a concrete reason:
`app.downloader` imports the upload ceiling from that module, so a route there that imported
the downloader back would close an import cycle that breaks on whichever module Python
happens to load first. The dependencies only run one way from here — this module knows about
both, and neither knows about it.

The internal route is here and the public one stays in `app/api/public_v1/analyses.py`, both
calling `accept_url`. Keeping the public route beside its sibling is what keeps the API-key
dependency on that router covering it: a public route defined here would be authenticated
only by whoever remembered to say so.
"""

import logging
import uuid
from contextlib import ExitStack

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.datastructures import Headers

from app import downloader
from app.api.analyses import (
    AcceptedUpload,
    CreatedAnalysis,
    accept_upload,
    created_analysis,
)
from app.db.session import get_session
from app.media import MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])

# What a downloaded file's extension is allowed to mean. The downloader asks the source for
# one already-muxed file and prefers MP4, but a site is free to serve something else, so the
# extension it wrote is mapped here rather than assumed — and anything not in this table is
# refused before the pipeline is entered.
#
# Deliberately the same two types an upload may declare. A URL that could smuggle in a
# container the upload endpoint refuses would make the door, rather than the media policy,
# decide what DeepGuard analyses; the test suite pins the two sets together.
URL_MEDIA_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime"}

# The longest URL that will be considered. Nothing about the downloader needs a bound — this
# is about not carrying an arbitrarily large string through validation, logging and the
# request body for something no real media URL comes close to.
MAX_URL_LENGTH = 2048


class UrlSubmission(BaseModel):
    """A submitted media URL, and nothing else.

    The URL is validated as a URL by `app.downloader`, not here. That check is inseparable
    from the SSRF defence — the scheme, the host and every address it resolves to — and a
    second, looser opinion at the request boundary would be a rule that could disagree with
    the one that matters. All this model establishes is that a non-empty, bounded string
    arrived.
    """

    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)


def client_error(error: downloader.DownloadError) -> HTTPException:
    """The safe, client-facing answer to a download that did not produce media.

    Every branch says what the caller can act on and nothing else. The downloader's own
    exception text can name a host, an extractor or a resolved address, and none of that
    belongs in a response — it is logged where it is raised.

    A refused URL and a blocked one deliberately share an answer. `BlockedAddress` means the
    URL resolved to something inside the deployment's network, and an error that said so
    would turn this endpoint into a scanner: a caller could learn which internal names exist
    by watching which of the two messages came back. What both get instead is the rule they
    broke.
    """
    if isinstance(error, downloader.MediaTooLarge):
        return HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"The media at this URL exceeds the {MAX_UPLOAD_BYTES} byte limit.",
        )

    if isinstance(error, downloader.LiveStreamRejected):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Live streams cannot be analysed.",
        )

    if isinstance(error, (downloader.UnsupportedUrl, downloader.BlockedAddress)):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The URL must be an http or https link to a downloadable video.",
        )

    # `DownloadUnavailable`, and any later `DownloadError` that has no branch of its own:
    # the URL was acceptable and the media did not arrive. Falling through to this rather
    # than raising on an unrecognized subclass is on purpose — a new failure mode should
    # reach the caller as a refusal, not as a 500 with a traceback behind it.
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="The media at this URL could not be downloaded.",
    )


async def accept_url(
    url: str,
    session: Session,
    *,
    api_key_id: uuid.UUID | None = None,
    max_active_analyses: int | None = None,
) -> AcceptedUpload:
    """Download the media behind a URL and put it through the upload pipeline.

    The download is synchronous — the request waits for it, and the analysis is queued only
    once the bytes are actually in hand. It runs in a worker thread rather than on the event
    loop: yt-dlp blocks, a large download blocks for a long time, and a blocked loop would
    stall every other request in this process, including the polling the dashboard does.
    That is a scheduling detail and not a change of shape; nothing about this call returns
    before the file exists.

    The downloaded file is wrapped as an `UploadFile` rather than given a pipeline of its
    own. `accept_upload` wants something with a content type, a filename and bytes to read,
    which is exactly what this is, and the wrapping is what keeps a URL submission and a
    dashboard upload from being two implementations of one policy.

    Cleanup is the `ExitStack`'s, on every path out: a successful analysis, a refusal from
    inside the pipeline, or an exception from anywhere below. The temporary directory is the
    downloader's own and it removes it — what is added here is that the file handle closes
    first, and that both happen whatever the pipeline did with the file.

    `api_key_id` and `max_active_analyses` are passed straight through, so a public URL
    submission is owned and throttled exactly as a public upload is.
    """
    with ExitStack() as cleanup:
        try:
            # `enter_context` in the thread, so the whole download happens off the loop.
            # The SSRF guard the downloader installs is thread-local and is taken and
            # released inside this call, so it applies to every resolution the download
            # makes and to nothing else this process is doing.
            media = await run_in_threadpool(cleanup.enter_context, downloader.download(url))
        except downloader.DownloadError as error:
            logger.info("Refused a URL submission: %s", error)
            raise client_error(error) from None

        content_type = URL_MEDIA_TYPES.get(media.path.suffix.lower())
        if content_type is None:
            logger.info("Downloaded media had an unusable extension: %r.", media.path.suffix)
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="The media at this URL is not an MP4 or MOV video.",
            )

        handle = cleanup.enter_context(media.path.open("rb"))

        return await accept_upload(
            UploadFile(
                file=handle,
                size=media.size_bytes,
                # The downloader's own name for the file it wrote, from a fixed template.
                # Nothing the source publisher controls reaches this.
                filename=media.filename,
                headers=Headers({"content-type": content_type}),
            ),
            session,
            api_key_id=api_key_id,
            max_active_analyses=max_active_analyses,
        )


@router.post(
    "/analyses/url", response_model=CreatedAnalysis, status_code=status.HTTP_202_ACCEPTED
)
async def create_url_analysis(
    submission: UrlSubmission, session: Session = Depends(get_session)
) -> CreatedAnalysis:
    """Accept a dashboard URL submission and report what the request established about it.

    The internal route, so no `api_key_id` and no limit: the analysis it commits is owned by
    nobody, which is what keeps it out of every public read, and the dashboard is not
    throttled. The response is the upload route's, because what was established is the same
    — the media, its identity and where it was put.
    """
    return created_analysis(await accept_url(submission.url, session))
