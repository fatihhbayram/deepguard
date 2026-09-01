"""The configurable bounds on external operations (R1-T3).

Two things are worth pinning here and nothing else is. The first is that an unconfigured
deployment keeps the exact figures that used to be literals in the modules — the move to
`app.limits` was meant to change nothing about how DeepGuard runs, and a drifted default
would be a behavioural change hiding inside a refactor. The second is that a value nobody
can act on is refused rather than quietly replaced with the default, because an operator
who set a bound and is not getting it has no way to find that out from the outside.
"""

import pytest

from app import limits


# The figures that were hardcoded before this task, restated here from the modules they
# lived in rather than imported from `app.limits` — importing them would make this test
# agree with itself no matter what either side said.
PREVIOUS_LITERALS = {
    limits.ffprobe_timeout_seconds: 10.0,
    limits.normalization_timeout_seconds: 900.0,
    limits.audio_extraction_timeout_seconds: 300.0,
    limits.nvidia_svd_timeout_seconds: 600.0,
    limits.nvidia_asd_timeout_seconds: 600.0,
    limits.download_socket_timeout_seconds: 30.0,
}


@pytest.fixture(autouse=True)
def unconfigured(monkeypatch):
    """Start every test from an environment that configures none of these.

    The suite is run by people whose shells may well have some of these set, and a test
    asserting a default against an environment that overrode it would fail for a reason
    that has nothing to do with the code.
    """
    for variable in (
        limits.FFPROBE_TIMEOUT_VARIABLE,
        limits.NORMALIZATION_TIMEOUT_VARIABLE,
        limits.AUDIO_EXTRACTION_TIMEOUT_VARIABLE,
        limits.NVIDIA_SVD_TIMEOUT_VARIABLE,
        limits.NVIDIA_ASD_TIMEOUT_VARIABLE,
        limits.DOWNLOAD_SOCKET_TIMEOUT_VARIABLE,
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize(
    "accessor, expected",
    PREVIOUS_LITERALS.items(),
    ids=lambda value: getattr(value, "__name__", value),
)
def test_an_unconfigured_deployment_keeps_the_figure_it_always_had(accessor, expected):
    assert accessor() == expected


def test_every_bound_has_a_test_for_its_default():
    # The parametrized test above only covers what is in the mapping, so a bound added to
    # `ACCESSORS` without a default recorded here would be silently untested.
    assert set(limits.ACCESSORS) == set(PREVIOUS_LITERALS)


def test_a_configured_bound_is_the_one_that_is_used(monkeypatch):
    monkeypatch.setenv(limits.NORMALIZATION_TIMEOUT_VARIABLE, "45.5")

    assert limits.normalization_timeout_seconds() == 45.5


def test_the_environment_is_read_per_call_rather_than_at_import(monkeypatch):
    # The property the whole module depends on: `analyze_video` resolves its deadline when
    # it is called, so a value bound at import would freeze whatever the environment said
    # when the process started.
    monkeypatch.setenv(limits.NVIDIA_SVD_TIMEOUT_VARIABLE, "11")
    assert limits.nvidia_svd_timeout_seconds() == 11.0

    monkeypatch.setenv(limits.NVIDIA_SVD_TIMEOUT_VARIABLE, "12")
    assert limits.nvidia_svd_timeout_seconds() == 12.0


def test_an_empty_value_is_the_same_as_an_unset_one(monkeypatch):
    # What a compose file passing an unset host variable through produces. Treating it as a
    # configuration error would fail every deployment that lists the variable without
    # setting it — which is exactly what `docker-compose.yml` does.
    monkeypatch.setenv(limits.FFPROBE_TIMEOUT_VARIABLE, "")

    assert limits.ffprobe_timeout_seconds() == 10.0


@pytest.mark.parametrize("configured", ["15m", "soon", "1,5", "nan"])
def test_a_value_that_is_not_seconds_is_refused(monkeypatch, configured):
    monkeypatch.setenv(limits.FFPROBE_TIMEOUT_VARIABLE, configured)

    with pytest.raises(limits.InvalidTimeout):
        limits.ffprobe_timeout_seconds()


@pytest.mark.parametrize("configured", ["0", "-1", "-0.5"])
def test_zero_and_negative_bounds_are_refused(monkeypatch, configured):
    # `0` is refused rather than read as "no limit". A deployment that wants no limit is not
    # something this service offers, and the convention is one that gets typed by accident.
    monkeypatch.setenv(limits.NORMALIZATION_TIMEOUT_VARIABLE, configured)

    with pytest.raises(limits.InvalidTimeout):
        limits.normalization_timeout_seconds()


def test_validate_names_the_variable_it_could_not_read(monkeypatch):
    monkeypatch.setenv(limits.NVIDIA_ASD_TIMEOUT_VARIABLE, "ten minutes")

    with pytest.raises(limits.InvalidTimeout) as raised:
        limits.validate()

    # An operator has to be able to find the typo from the log line alone.
    assert limits.NVIDIA_ASD_TIMEOUT_VARIABLE in str(raised.value)


def test_validate_resolves_every_bound_when_they_are_all_usable():
    resolved = limits.validate()

    assert set(resolved) == {accessor.__name__ for accessor in limits.ACCESSORS}
    assert all(seconds > 0 for seconds in resolved.values())


def test_a_malformed_value_never_quotes_more_than_what_was_configured(monkeypatch):
    # The message is logged and, on a bad configuration, is the reason the worker exits. It
    # carries a variable name and the value that was typed into it — neither of which is a
    # credential — and nothing else from the environment.
    monkeypatch.setenv(limits.DOWNLOAD_SOCKET_TIMEOUT_VARIABLE, "thirty")

    with pytest.raises(limits.InvalidTimeout) as raised:
        limits.download_socket_timeout_seconds()

    assert str(raised.value) == (
        f"{limits.DOWNLOAD_SOCKET_TIMEOUT_VARIABLE} is not a number of seconds: 'thirty'"
    )
