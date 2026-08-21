"""Reader for C2PA provenance embedded in a local media file.

C2PA (Content Credentials) is signed provenance the producing tool writes into the
asset itself: who claims to have made it, with what, and a signature over the bytes.
Reading it is a forensic evidence source in its own right, independent of any detector.

Two things this module deliberately does not do:

- it does not interpret. Absent credentials are not evidence of manipulation, an invalid
  signature is not proof of a deepfake, and neither is turned into a risk level here
  (rule 11). What C2PA reports is passed through as C2PA words it;
- it does not fetch. A manifest stored away from the asset is named by a URL the upload
  itself carries, so retrieving it would let arbitrary uploaded media steer a request out
  of the worker. Remote manifests and OCSP revocation lookups are therefore switched off
  and the remote URL is recorded as evidence instead.

Validation runs against the SDK's own trust list, so media signed by a recognized issuer
reports as trusted and media signed by anyone else does not.
"""

from dataclasses import dataclass
from pathlib import Path

import c2pa

# Everything this module reads is local. Both settings below default to on and would
# otherwise reach the network on behalf of an untrusted file.
READER_SETTINGS = {
    "verify": {
        "remote_manifest_fetch": False,
        "ocsp_fetch": False,
    }
}

# The prefix of the SDK's own "this manifest lives elsewhere" error. The URL follows it.
_REMOTE_MANIFEST_PREFIX = "must fetch remote manifests from url "


class C2paExtractionError(Exception):
    """Base class for every failure raised by this module."""


class C2paLocalFileError(C2paExtractionError):
    """The local media file could not be opened or read."""


class C2paUnsupportedFormat(C2paExtractionError):
    """The SDK cannot inspect this container at all, so nothing is known either way.

    Distinct from a supported file that simply carries no credentials: there the answer
    is a fact about the media, here it is a limit of the reader.
    """


@dataclass(frozen=True)
class C2paEvidence:
    """What C2PA reports about one media file, in C2PA's own terms.

    `manifest_exists` is the first question and the only one always answered. When it is
    false every manifest field below is empty, and that is an ordinary result rather than
    a failure — most media carries no credentials at all.

    `validation_state` is the SDK's overall verdict, one of its own words: `Trusted` when
    the signature verifies and the signer is on the trust list, `Valid` when it verifies
    but the signer is not, `Invalid` when validation failed. `validation_failures` carries
    the provider's failure codes for the active manifest verbatim — `signingCredential.
    untrusted` next to a `Valid` state is the ordinary shape of a self-signed asset.

    `manifest_json` is the SDK's full rendering of the manifest store, kept as the
    provider's own evidence artifact; the named fields are a summary of it, never a
    substitute. `sdk_version` is the c2pa-rs version that produced this reading.
    """

    manifest_exists: bool
    sdk_version: str
    validation_state: str | None = None
    validation_failures: tuple[str, ...] = ()
    is_embedded: bool = False
    remote_manifest_url: str | None = None
    active_manifest_label: str | None = None
    claim_generator: str | None = None
    signature_issuer: str | None = None
    signature_time: str | None = None
    assertion_labels: tuple[str, ...] = ()
    manifest_json: str | None = None


def _check_readable(file_path: Path) -> None:
    """Fail on our own unreadable file before the SDK turns it into a C2PA error."""
    try:
        open(file_path, "rb").close()
    except OSError as error:
        raise C2paLocalFileError(f"Cannot read media file '{file_path}': {error}") from error


def _remote_manifest_url(error: Exception) -> str | None:
    """Recover the URL the SDK refused to fetch, or None if this is another failure.

    The refusal arrives as an untyped error carrying the URL in its text, and the URL is
    evidence: it says provenance was claimed but stored away from the asset. Matching on
    the text is best effort by design — if the SDK rewords the error the file is reported
    as a plain extraction failure, which is better than the field lying.
    """
    message = str(error)
    _, separator, url = message.partition(_REMOTE_MANIFEST_PREFIX)
    return url.strip() if separator and url.strip() else None


def _validation_failures(reader: c2pa.Reader) -> tuple[str, ...]:
    """The active manifest's failure codes, in the order the SDK reported them."""
    results = reader.get_validation_results() or {}
    active = results.get("activeManifest") or {}
    return tuple(entry["code"] for entry in active.get("failure", []) if "code" in entry)


def _first_claim_generator(manifest: dict) -> str | None:
    """The name of the tool that made the claim, as the manifest itself names it."""
    generators = manifest.get("claim_generator_info") or []
    if not generators:
        return None
    return generators[0].get("name")


def _evidence(reader: c2pa.Reader) -> C2paEvidence:
    manifest = reader.get_active_manifest() or {}
    signature = manifest.get("signature_info") or {}

    return C2paEvidence(
        manifest_exists=True,
        sdk_version=c2pa.sdk_version(),
        validation_state=reader.get_validation_state(),
        validation_failures=_validation_failures(reader),
        is_embedded=reader.is_embedded(),
        remote_manifest_url=reader.get_remote_url(),
        active_manifest_label=manifest.get("label"),
        claim_generator=_first_claim_generator(manifest),
        signature_issuer=signature.get("issuer"),
        # Present only when the signature carries a trusted timestamp; a signature
        # without one says nothing about when it was made.
        signature_time=signature.get("time"),
        assertion_labels=tuple(
            assertion["label"]
            for assertion in manifest.get("assertions", [])
            if "label" in assertion
        ),
        manifest_json=reader.json(),
    )


def extract_c2pa_evidence(file_path: Path) -> C2paEvidence:
    """Read the C2PA provenance of one local media file.

    Media with no credentials returns `manifest_exists=False`, not an error. So does
    media whose manifest lives at a remote URL, with that URL recorded and unvisited.

    Raises `C2paLocalFileError` if our own file cannot be read, `C2paUnsupportedFormat`
    if the SDK cannot inspect the container, and `C2paExtractionError` for anything else
    the SDK reports.
    """
    _check_readable(file_path)

    try:
        with (
            c2pa.Settings.from_dict(READER_SETTINGS) as settings,
            c2pa.Context(settings=settings) as context,
            c2pa.Reader(file_path, context=context) as reader,
        ):
            return _evidence(reader)
    except c2pa.C2paError.ManifestNotFound:
        return C2paEvidence(manifest_exists=False, sdk_version=c2pa.sdk_version())
    except c2pa.C2paError.NotSupported as error:
        raise C2paUnsupportedFormat(f"C2PA cannot inspect '{file_path}': {error}") from error
    except c2pa.C2paError as error:
        remote_url = _remote_manifest_url(error)
        if remote_url is not None:
            return C2paEvidence(
                manifest_exists=False,
                sdk_version=c2pa.sdk_version(),
                remote_manifest_url=remote_url,
            )
        raise C2paExtractionError(f"C2PA extraction failed for '{file_path}': {error}") from error
