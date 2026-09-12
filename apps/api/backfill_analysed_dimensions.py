"""Measure the artifacts historical analyses were detected against, where they survive.

    docker compose exec api python backfill_analysed_dimensions.py [--commit]

`media_files.analyzed_width`/`analyzed_height` were added after these rows were written, so
every analysis from before the change carries null in both. This script fills that gap the
only way it may be filled — by probing the artifact itself — and leaves it alone wherever
the artifact is gone.

## Why a script rather than a migration

The migration that adds these columns backfills nothing, deliberately. Filling them would
mean downloading and probing every stored derivative inside a DDL transaction, on a
database nobody can write to while it runs, with a failure mode of "the schema change timed
out because object storage was slow". The measurement is also not schema: it can be run
again, it can be run in parts, and it can be abandoned halfway without leaving the schema
inconsistent. So the columns arrive null and this fills what it can, separately and
interruptibly.

## What is deterministic, and what is not

Rows fall into three groups, and only the first two can be answered honestly:

  * **Bypassed media** (`was_normalized` false, `derivative_storage_key` equal to
    `original_storage_key`). No transcode ever happened, so the artifact a detector read is
    the forensic original, and `width`/`height` are what ffprobe measured off exactly those
    bytes at upload. Copying them across is not an inference — it is the same measurement of
    the same file — so these are filled from the existing columns without a download.

  * **Normalized media whose derivative still exists.** The derivative is fetched and
    probed, and what ffprobe says it is, it is. This is the group the download cost is spent
    on.

  * **Everything else** — a normalized row whose derivative has been removed, or a job that
    never produced one. These stay null. There is no way to recover the shape of a file that
    no longer exists, and the rotation metadata on the original is not one: it would mean
    reimplementing ffmpeg's display-matrix and padding behaviour, guessing at the ffmpeg
    version that ran, and writing the result into a column that is supposed to mean
    "measured". A null here reads as "not recorded", which is true; a computed value would
    read as a measurement, which would not be.

`display_rotation` is never backfilled, in any group. It is read from the original during
`probe_media`, and re-probing every stored original to fill a field that explains a
divergence — rather than producing one — is not worth the download. It stays null, meaning
"nobody looked", which is a different fact from the `0` a probe records when it looks and
finds nothing.

## Safety

A dry run by default: it reports what it would do and writes nothing. `--commit` is what
actually updates rows. Nothing is ever overwritten — only rows with both columns null are
considered — so the script is safe to re-run, and re-running it is how a partially completed
pass is finished.
"""

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

from sqlalchemy import or_, select

from app.db.models import MediaFile
from app.db.session import SessionLocal
from app.media import MediaProbeError, MediaProbeUnavailable, probe_dimensions
from app.storage import fetch_object

logger = logging.getLogger("backfill")


def candidates(session) -> list[MediaFile]:
    """Every row that has never recorded an analysed shape, oldest first.

    Both columns null is the condition, not either: a row holding one without the other
    would be a defect worth seeing rather than something to quietly complete.
    """
    return list(
        session.execute(
            select(MediaFile)
            .where(MediaFile.analyzed_width.is_(None))
            .where(MediaFile.analyzed_height.is_(None))
            .order_by(MediaFile.id)
        ).scalars()
    )


def bypassed(row: MediaFile) -> bool:
    """Whether the original itself was the artifact handed to the detectors.

    The storage keys agreeing is the evidence, not `was_normalized` alone. A row that was
    never going to be transcoded has the original's key written at upload; a row still
    owing a derivative has null there, and nothing was ever analysed for it at all.
    """
    return (
        not row.was_normalized
        and row.derivative_storage_key is not None
        and row.derivative_storage_key == row.original_storage_key
    )


def measured(storage_key: str) -> tuple[int, int] | None:
    """Download one stored artifact and measure it, or return None if that is not possible.

    The temp file is removed on every path. A missing object and an unreadable one are the
    same outcome here — the shape cannot be established — and both leave the row null
    rather than failing the pass: one deleted derivative must not stop the other hundred
    from being measured.
    """
    handle = tempfile.NamedTemporaryFile(prefix="deepguard-backfill-", suffix=".mp4", delete=False)
    handle.close()
    path = Path(handle.name)

    try:
        fetch_object(storage_key, path)
        return asyncio.run(probe_dimensions(path))
    except (MediaProbeError, MediaProbeUnavailable) as error:
        logger.warning("%s could not be probed: %s", storage_key, error)
        return None
    except Exception as error:  # noqa: BLE001 — object storage raises its own hierarchy
        logger.warning("%s could not be fetched: %s", storage_key, error)
        return None
    finally:
        path.unlink(missing_ok=True)


def run(commit: bool) -> int:
    """Fill what can be filled, and report the three groups separately.

    The counts are the point of the dry run: they say how much of the history is
    recoverable before anything is written, and how much of it is honestly unanswerable.
    """
    filled_from_original = 0
    filled_from_derivative = 0
    left_null = 0

    with SessionLocal() as session:
        rows = candidates(session)
        logger.info("%s rows carry no analysed dimensions.", len(rows))

        for row in rows:
            if bypassed(row):
                # The same measurement of the same bytes, already in the table.
                row.analyzed_width, row.analyzed_height = row.width, row.height
                filled_from_original += 1
                continue

            if row.derivative_storage_key is None:
                # No artifact was ever produced for this analysis.
                left_null += 1
                continue

            dimensions = measured(row.derivative_storage_key)
            if dimensions is None:
                left_null += 1
                continue

            row.analyzed_width, row.analyzed_height = dimensions
            filled_from_derivative += 1

        if commit:
            session.commit()
        else:
            session.rollback()

    logger.info("Filled from the original (no transcode): %s", filled_from_original)
    logger.info("Filled by probing the derivative:        %s", filled_from_derivative)
    logger.info("Left null (artifact unavailable):        %s", left_null)
    if not commit:
        logger.info("Dry run — nothing was written. Re-run with --commit to apply.")

    return filled_from_original + filled_from_derivative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Write the measurements. Without this the script only reports what it would do.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    run(arguments.commit)


if __name__ == "__main__":
    main()
