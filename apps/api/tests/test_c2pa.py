"""Tests for the C2PA evidence reader.

Nothing here is a recorded fixture of somebody else's signature. A throwaway certificate
chain is generated per session and used to sign the committed media with the same SDK the
reader uses, so a valid manifest, a broken one and media with no manifest at all are each
produced for real and read back through the real validation path.

The signer is unknown to the SDK's trust list, which is exactly what an ordinary
self-signed asset looks like: `Valid` state, `signingCredential.untrusted` alongside it.
"""

import datetime
import json
import shutil
from pathlib import Path

import c2pa
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from app import c2pa_extractor
from app.c2pa_extractor import extract_c2pa_evidence

SOURCE_MEDIA = Path(__file__).parent / "fixtures" / "still.mp4"

SIGNER_ORGANIZATION = "DeepGuard Test Signing Cert"
SIGNER_COMMON_NAME = "DeepGuard Test Signer"
CLAIM_GENERATOR = "deepguard-test"

MANIFEST_DEFINITION = {
    "claim_generator_info": [{"name": CLAIM_GENERATOR, "version": "0.1"}],
    "title": "still.mp4",
    "assertions": [
        {
            "label": "c2pa.actions",
            "data": {
                "actions": [
                    {
                        "action": "c2pa.created",
                        "digitalSourceType": (
                            "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
                        ),
                    }
                ]
            },
        }
    ],
}


def _key_usage(**allowed) -> x509.KeyUsage:
    usages = dict(
        digital_signature=False,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=False,
        crl_sign=False,
        encipher_only=False,
        decipher_only=False,
    )
    usages.update(allowed)
    return x509.KeyUsage(**usages)


def _name(common_name: str, organization: str) -> x509.Name:
    # C2PA requires the signing certificate to identify an organization; a chain carrying
    # only a common name is rejected as an invalid credential rather than an untrusted one.
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


@pytest.fixture(scope="session")
def signer() -> c2pa.Signer:
    """A signer whose certificate meets the C2PA end-entity profile.

    The root is self-signed and stays out of the chain the manifest carries — C2PA
    rejects a chain containing its own trust anchor.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=365)

    root_key = ec.generate_private_key(ec.SECP256R1())
    root_name = _name("DeepGuard Test Root", "DeepGuard Test Root CA")
    root = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(_key_usage(key_cert_sign=True, crl_sign=True), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False
        )
        .sign(root_key, hashes.SHA256())
    )

    signing_key = ec.generate_private_key(ec.SECP256R1())
    certificate = (
        x509.CertificateBuilder()
        .subject_name(_name(SIGNER_COMMON_NAME, SIGNER_ORGANIZATION))
        .issuer_name(root.subject)
        .public_key(signing_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=True
        )
        .add_extension(
            _key_usage(digital_signature=True, content_commitment=True), critical=True
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(signing_key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()),
            critical=False,
        )
        .sign(root_key, hashes.SHA256())
    )

    return c2pa.Signer.from_info(
        c2pa.C2paSignerInfo(
            alg=b"es256",
            sign_cert=certificate.public_bytes(serialization.Encoding.PEM),
            private_key=signing_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
            ta_url=None,
        )
    )


@pytest.fixture
def unsigned(tmp_path) -> Path:
    """The committed media, untouched: real MP4, no credentials of any kind."""
    destination = tmp_path / "unsigned.mp4"
    shutil.copy(SOURCE_MEDIA, destination)
    return destination


@pytest.fixture
def signed(tmp_path, unsigned, signer) -> Path:
    destination = tmp_path / "signed.mp4"
    c2pa.Builder(dict(MANIFEST_DEFINITION)).sign_file(unsigned, destination, signer)
    return destination


@pytest.fixture
def tampered(tmp_path, signed) -> Path:
    """The signed media with its picture data overwritten, manifest left intact.

    Zeroing bytes just inside `mdat` edits what the BMFF hash assertion covers without
    disturbing the box structure, so the file still parses and the manifest still reads.
    """
    data = bytearray(signed.read_bytes())
    payload_start = data.find(b"mdat") + len(b"mdat")
    assert payload_start > len(b"mdat"), "fixture has no mdat box to edit"
    data[payload_start : payload_start + 32] = bytes(32)

    destination = tmp_path / "tampered.mp4"
    destination.write_bytes(bytes(data))
    return destination


def test_media_without_credentials_is_not_a_failure(unsigned):
    evidence = extract_c2pa_evidence(unsigned)

    assert evidence.manifest_exists is False
    assert evidence.validation_state is None
    assert evidence.validation_failures == ()
    assert evidence.manifest_json is None
    assert evidence.assertion_labels == ()


def test_signed_media_validates(signed):
    evidence = extract_c2pa_evidence(signed)

    assert evidence.manifest_exists is True
    assert evidence.validation_state == "Valid"
    assert evidence.is_embedded is True
    assert evidence.remote_manifest_url is None


def test_an_unrecognized_signer_is_reported_as_untrusted(signed):
    """Untrusted is not invalid: the signature verifies, the issuer is just unknown."""
    evidence = extract_c2pa_evidence(signed)

    assert evidence.validation_failures == ("signingCredential.untrusted",)


def test_the_manifest_summary_reports_what_the_manifest_says(signed):
    evidence = extract_c2pa_evidence(signed)

    assert evidence.claim_generator == CLAIM_GENERATOR
    assert evidence.signature_issuer == SIGNER_ORGANIZATION
    assert evidence.active_manifest_label.startswith("urn:c2pa:")
    # The SDK upgrades `c2pa.actions` to its versioned label and adds the hash assertion
    # it computes itself, so the labels are the manifest's own, not the ones asked for.
    assert "c2pa.actions.v2" in evidence.assertion_labels


def test_a_signature_without_a_timestamp_claims_no_time(signed):
    evidence = extract_c2pa_evidence(signed)

    assert evidence.signature_time is None


def test_the_provider_manifest_is_kept_verbatim(signed):
    evidence = extract_c2pa_evidence(signed)

    store = json.loads(evidence.manifest_json)
    assert store["validation_state"] == evidence.validation_state
    assert store["active_manifest"] == evidence.active_manifest_label


def test_the_reader_version_travels_with_the_evidence(signed, unsigned):
    assert extract_c2pa_evidence(signed).sdk_version == c2pa.sdk_version()
    assert extract_c2pa_evidence(unsigned).sdk_version == c2pa.sdk_version()


def test_edited_media_fails_validation_but_keeps_its_manifest(tampered):
    evidence = extract_c2pa_evidence(tampered)

    assert evidence.manifest_exists is True
    assert evidence.validation_state == "Invalid"
    assert "assertion.bmffHash.mismatch" in evidence.validation_failures
    assert evidence.claim_generator == CLAIM_GENERATOR


def test_a_remote_manifest_is_recorded_and_never_fetched(tmp_path, unsigned, signer):
    """The URL comes out of the file; visiting it would be the file steering the worker."""
    remote_url = "http://127.0.0.1:9/deepguard-must-not-fetch.c2pa"
    destination = tmp_path / "remote.mp4"

    builder = c2pa.Builder(dict(MANIFEST_DEFINITION))
    builder.set_remote_url(remote_url)
    builder.set_no_embed()
    builder.sign_file(unsigned, destination, signer)

    evidence = extract_c2pa_evidence(destination)

    assert evidence.manifest_exists is False
    assert evidence.remote_manifest_url == remote_url
    assert evidence.validation_state is None


def test_a_missing_file_is_a_local_failure(tmp_path):
    with pytest.raises(c2pa_extractor.C2paLocalFileError):
        extract_c2pa_evidence(tmp_path / "absent.mp4")


def test_a_container_c2pa_cannot_read_is_not_reported_as_absent_credentials(tmp_path):
    not_media = tmp_path / "notes.txt"
    not_media.write_text("no container here")

    with pytest.raises(c2pa_extractor.C2paUnsupportedFormat):
        extract_c2pa_evidence(not_media)
