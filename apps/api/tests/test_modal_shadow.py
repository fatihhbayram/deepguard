"""Executing a shadow workload on Modal's GPU, and everything that must survive it (R6-T2).

R6-T1's suite proved the four negatives shadow mode rests on: an observation is not evidence,
no reader can see it, the risk engine cannot consume it, and production never waits for it.
This module does not re-prove them — it proves that moving the workload onto somebody else's
GPU does not weaken any of them, which is a different and mostly adversarial question.

So most of what is tested here is failure. A remote backend introduces failure modes a local
stub does not have — a vendor SDK that is not installed, credentials that are absent, a
function that was never deployed, a network that goes away mid-call, a container killed by
its own timeout, a result that is not the shape it promised — and every one of them has to
end the same way: one `shadow_runs` row marked failed, a worker still polling, and an
analysis that never knew.

The Modal SDK is faked throughout rather than reached. A test that called Modal would be a
test of somebody's control plane, would cost GPU-seconds, and would be unable to produce the
failures above on demand — which are exactly the cases worth pinning. That the real thing
works is a manual verification, recorded in the task's report; that it fails *safely* is what
runs on every commit.
"""

import json
import time
import tokenize
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app import modal_client, shadow
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    SHADOW_RUN_STATUS_COMPLETED,
    SHADOW_RUN_STATUS_FAILED,
    SHADOW_RUN_STATUS_PROCESSING,
    SHADOW_RUN_STATUS_QUEUED,
    Analysis,
    AnalysisSignal,
    MediaFile,
    ShadowRun,
)
from app.db.session import SessionLocal, engine

API_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = API_ROOT.parent / "web"

# The names a customer-facing module would have to contain to be able to know that a shadow
# observation came from a rented GPU. `modal` on its own is deliberately included: the point
# is that nothing outside the worker's own path so much as imports the SDK.
MODAL_NAMES = ("modal", "Modal", "modal_client", "modal_shadow_app")

# A well-formed remote result, as `app.modal_shadow_app.run_shadow_stub` returns one.
def remote_result(analysis_id: uuid.UUID) -> dict:
    return {
        "observed": True,
        "analysis_id": str(analysis_id),
        "backend": "modal",
        "workload_version": "modal-stub-1",
        "gpu_requested": "L4",
        "gpu_attached": "NVIDIA L4, 550.127.05, 23034 MiB",
        "remote_seconds": 0.021,
    }


# --- configuration -----------------------------------------------------------------------


@pytest.fixture
def modal_configured(monkeypatch):
    """Modal switched on the way a deployment switches it on: a flag and two tokens."""
    monkeypatch.setenv(modal_client.MODAL_SHADOW_VARIABLE, "true")
    monkeypatch.setenv(modal_client.TOKEN_ID_VARIABLE, "ak-test")
    monkeypatch.setenv(modal_client.TOKEN_SECRET_VARIABLE, "as-test")


@pytest.fixture
def modal_absent(monkeypatch):
    """A deployment that never configured Modal — the R6-T1 stack, unchanged."""
    for variable in (
        modal_client.MODAL_SHADOW_VARIABLE,
        modal_client.TOKEN_ID_VARIABLE,
        modal_client.TOKEN_SECRET_VARIABLE,
    ):
        monkeypatch.delenv(variable, raising=False)


def test_modal_is_used_only_when_it_is_asked_for_and_configured(modal_configured):
    assert modal_client.configured() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "  "])
def test_the_modal_flag_means_what_it_says(monkeypatch, modal_configured, value):
    # The same trap `DEEPGUARD_SHADOW_MODE` avoids: a compose file that carries `=false` for a
    # backend that bills by the GPU-second must not be read as "on".
    monkeypatch.setenv(modal_client.MODAL_SHADOW_VARIABLE, value)

    assert modal_client.enabled() is False
    assert modal_client.configured() is False


@pytest.mark.parametrize(
    "variable", [modal_client.TOKEN_ID_VARIABLE, modal_client.TOKEN_SECRET_VARIABLE]
)
def test_a_missing_credential_leaves_modal_unconfigured(monkeypatch, modal_configured, variable):
    monkeypatch.delenv(variable, raising=False)

    assert modal_client.configured() is False


@pytest.mark.parametrize(
    "variable", [modal_client.TOKEN_ID_VARIABLE, modal_client.TOKEN_SECRET_VARIABLE]
)
def test_an_empty_credential_is_a_missing_one(monkeypatch, modal_configured, variable):
    # What a compose file passing an unset host variable through actually produces. A client
    # authenticating with an empty token would fail on the wire; this fails before the import.
    monkeypatch.setenv(variable, "")

    assert modal_client.configured() is False


def test_credentials_without_the_flag_do_not_enable_modal(monkeypatch, modal_configured):
    monkeypatch.delenv(modal_client.MODAL_SHADOW_VARIABLE, raising=False)

    assert modal_client.credentials_present() is True
    assert modal_client.configured() is False


def test_an_unconfigured_deployment_queues_the_local_stub(modal_absent):
    assert shadow.workload() == shadow.STUB_WORKLOAD


def test_a_configured_deployment_queues_the_modal_stub(modal_configured):
    assert shadow.workload() == shadow.MODAL_STUB_WORKLOAD


def test_the_sdk_is_never_reached_without_configuration(modal_absent, monkeypatch):
    """The load-bearing half of "opt-in": unconfigured means Modal is not even imported.

    A `spawn_stub` that imported the SDK first and checked configuration afterwards would
    authenticate from whatever `~/.modal.toml` the machine happens to hold — which is exactly
    how a developer's laptop ends up billing a workspace nobody meant to use.
    """
    def explode():
        raise AssertionError("the Modal SDK was imported without configuration")

    monkeypatch.setattr(modal_client, "_load_modal", explode)

    with pytest.raises(modal_client.ModalNotConfigured):
        modal_client.spawn_stub(uuid.uuid4())


def test_a_broken_sdk_is_one_failed_run_rather_than_a_broken_import(modal_configured, monkeypatch):
    """`_load_modal` reports, never propagates somebody else's ImportError.

    The import is inside a function precisely so this is possible. At module scope, a Modal
    package that failed to import would stop the worker process from starting — a production
    outage caused by an experiment's dependency.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            raise ImportError("no modal here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.delitem(__import__("sys").modules, "modal", raising=False)

    with pytest.raises(modal_client.ModalUnavailable):
        modal_client.spawn_stub(uuid.uuid4())


# --- the remote result -------------------------------------------------------------------


def test_a_well_formed_result_is_taken_as_it_came():
    analysis_id = uuid.uuid4()
    result = remote_result(analysis_id)

    validated = modal_client._validated(result)

    # Uninterpreted: the document is carried through whole, which is what makes it possible
    # for a later workload to report fields this release has never heard of.
    assert validated.evidence == result
    assert validated.provider_version == "modal-stub-1"


@pytest.mark.parametrize("returned", [None, "ok", 3, ["observed"], b"{}"])
def test_a_result_that_is_not_a_document_is_refused(returned):
    with pytest.raises(modal_client.ModalExecutionError):
        modal_client._validated(returned)


@pytest.mark.parametrize("version", [None, "", "   ", 1, {"v": 1}])
def test_a_result_that_cannot_say_which_workload_produced_it_is_refused(version):
    # An observation whose deployment identity is unknown cannot be told apart from the next
    # workload's, which makes the corpus these rows exist for unreadable.
    result = remote_result(uuid.uuid4())
    if version is None:
        result.pop(modal_client.VERSION_KEY)
    else:
        result[modal_client.VERSION_KEY] = version

    with pytest.raises(modal_client.ModalExecutionError):
        modal_client._validated(result)


def test_a_result_postgresql_could_not_store_is_refused():
    # Caught here rather than by `JSONB` at commit time, where it would already have failed
    # the transaction that was writing the observation.
    result = remote_result(uuid.uuid4())
    result["device"] = {1, 2}

    with pytest.raises(modal_client.ModalExecutionError):
        modal_client._validated(result)


def test_the_stub_result_survives_a_round_trip_through_json():
    # It is written to a JSONB column and read back by an offline calibration much later.
    result = remote_result(uuid.uuid4())

    assert json.loads(json.dumps(result)) == result


# --- starting and collecting -------------------------------------------------------------


class FakeCall:
    """A spawned Modal call whose behaviour each test dictates.

    `results` is what successive collects do: an exception instance is raised, anything else
    is returned. That is enough to express a cold start (several "not yet"s, then a value), a
    remote failure, and a network that went away.
    """

    def __init__(self, results, forever=False):
        self.results = list(results)
        self.forever = forever
        self.polls = 0
        self.cancelled = False
        self.last = None

    def get(self, timeout=None):
        # The one thing every collect must do. A blocking wait here is the defect this whole
        # design exists to prevent, and it is cheap to assert on every single call.
        assert timeout == 0, f"collect waited {timeout}s instead of asking once"

        self.polls += 1
        if not self.results:
            if not self.forever:
                raise AssertionError("polled more often than the test allowed for")
            # A remote call that simply never finishes, which is the case the local deadline
            # exists for.
            self.results = [self.last]
        outcome = self.results.pop(0)
        self.last = outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def cancel(self):
        self.cancelled = True


class FakeFunction:
    def __init__(self, call):
        self.call = call
        self.spawned_with = None

    def spawn(self, *args):
        self.spawned_with = args
        return self.call


@pytest.fixture
def fake_modal(monkeypatch):
    """Modal's SDK, faked at the two seams `app.modal_client` actually uses.

    The real `modal.exception` classes are kept — `collect` branches on them, and a fake
    hierarchy would let the ordering of those `except` clauses drift without a test noticing.
    """
    modal = pytest.importorskip("modal")
    import modal.exception  # noqa: F401

    def install(call):
        function = FakeFunction(call)
        monkeypatch.setattr(modal_client, "_load_modal", lambda: modal)
        monkeypatch.setattr(modal_client, "_deployed_function", lambda _modal: function)
        return function

    return install


def test_spawning_returns_without_waiting_for_the_gpu(modal_configured, fake_modal):
    """The property the worker depends on: starting a remote call does not wait for it.

    `FakeCall` has no results at all here, so any attempt to read one would fail the test.
    Spawning must not read one.
    """
    analysis_id = uuid.uuid4()
    call = FakeCall([])
    function = fake_modal(call)

    pending = modal_client.spawn_stub(analysis_id)

    assert call.polls == 0
    # The analysis id is what the remote half is given, as a string: a UUID does not survive
    # somebody else's serializer, and the remote half echoes it into the observation.
    assert function.spawned_with == (str(analysis_id),)
    assert pending.analysis_id == analysis_id
    assert pending.deadline > time.monotonic()


@pytest.mark.parametrize(
    "not_yet",
    [
        # What Modal actually raises: `modal/_functions.py` does not import `TimeoutError`
        # from `modal.exception`, so `poll_function`'s bare `raise TimeoutError()` is the
        # builtin — which is an `OSError` subclass, and was therefore being read as a lost
        # connection until a runtime test caught it. This parameter is the regression.
        "builtin",
        # The documented type, which the SDK's hierarchy says this condition is.
        "modal",
    ],
)
def test_a_call_that_has_not_answered_yet_is_reported_as_such(
    modal_configured, fake_modal, not_yet
):
    """A cold start, from the collector's side: "not yet" is a normal answer, not a failure."""
    import modal.exception

    unfinished = (
        TimeoutError() if not_yet == "builtin" else modal.exception.TimeoutError()
    )
    call = FakeCall([unfinished] * 4 + [remote_result(uuid.uuid4())])
    fake_modal(call)
    pending = modal_client.spawn_stub(uuid.uuid4())

    for _ in range(4):
        assert modal_client.collect(pending) is None

    result = modal_client.collect(pending)
    assert result.provider_version == "modal-stub-1"
    assert call.polls == 5
    assert call.cancelled is False


def test_a_call_that_outlives_the_local_deadline_is_refused(
    modal_configured, fake_modal, monkeypatch
):
    """The deadline bounds the *asking*, and it is noticed where time is noticed.

    Cancelling is the caller's job here rather than this function's — `app.shadow` cancels on
    every path that ends a run — so this asserts the refusal, and the shadow tests below
    assert the cancellation.
    """
    import modal.exception

    monkeypatch.setattr(modal_client, "shadow_modal_timeout_seconds", lambda: 0.01)

    call = FakeCall([TimeoutError()], forever=True)
    fake_modal(call)
    pending = modal_client.spawn_stub(uuid.uuid4())

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            modal_client.collect(pending)
        except modal_client.ModalTimeout:
            break
        time.sleep(0.01)
    else:
        pytest.fail("the local deadline was never enforced")


def test_a_container_killed_by_its_own_timeout_is_an_execution_failure(
    modal_configured, fake_modal
):
    # Modal's remote ceiling, not ours. Distinguished from "no output yet" even though both
    # are `modal.exception.TimeoutError` subclasses — which is why the `except` clauses are
    # ordered the way they are, and why this test exists at all.
    import modal.exception

    fake_modal(FakeCall([modal.exception.FunctionTimeoutError("killed")]))
    pending = modal_client.spawn_stub(uuid.uuid4())

    with pytest.raises(modal_client.ModalExecutionError):
        modal_client.collect(pending)


def test_an_expired_output_is_an_execution_failure(modal_configured, fake_modal):
    import modal.exception

    fake_modal(FakeCall([modal.exception.OutputExpiredError("gone")]))
    pending = modal_client.spawn_stub(uuid.uuid4())

    with pytest.raises(modal_client.ModalExecutionError):
        modal_client.collect(pending)


def test_a_workload_that_raised_remotely_is_an_execution_failure(
    modal_configured, fake_modal
):
    import modal.exception

    fake_modal(FakeCall([modal.exception.RemoteError("the workload raised")]))
    pending = modal_client.spawn_stub(uuid.uuid4())

    with pytest.raises(modal_client.ModalExecutionError):
        modal_client.collect(pending)


def test_a_connection_that_went_away_is_reported_as_unavailable(
    modal_configured, fake_modal
):
    # Unavailable rather than an execution failure: the workload may well still be running,
    # we simply cannot hear it. The distinction is written into `error_message` and is the
    # kind of thing that matters when these rows are read back to decide whether Modal is
    # worth keeping.
    fake_modal(FakeCall([ConnectionResetError("connection reset")]))
    pending = modal_client.spawn_stub(uuid.uuid4())

    with pytest.raises(modal_client.ModalUnavailable):
        modal_client.collect(pending)

    # And the distinction is real rather than accidental: an unfinished call is an
    # `OSError` subclass too, and must not land here.
    assert issubclass(TimeoutError, OSError)


def test_a_spawn_that_modal_refuses_is_reported_rather_than_raised_raw(
    modal_configured, monkeypatch
):
    modal = pytest.importorskip("modal")

    class RefusingFunction:
        def spawn(self, *_args):
            raise modal.exception.ResourceExhaustedError("no capacity")

    monkeypatch.setattr(modal_client, "_load_modal", lambda: modal)
    monkeypatch.setattr(modal_client, "_deployed_function", lambda _m: RefusingFunction())

    with pytest.raises(modal_client.ModalUnavailable):
        modal_client.spawn_stub(uuid.uuid4())


def test_a_function_that_was_never_deployed_is_reported_rather_than_raised_raw(
    modal_configured, monkeypatch
):
    modal = pytest.importorskip("modal")

    def missing(_app_name, _name, **_kwargs):
        raise modal.exception.NotFoundError("no such function")

    monkeypatch.setattr(modal_client, "_load_modal", lambda: modal)
    monkeypatch.setattr(modal.Function, "from_name", staticmethod(missing))

    with pytest.raises(modal_client.ModalUnavailable):
        modal_client.spawn_stub(uuid.uuid4())


def test_cancellation_never_becomes_the_failure(modal_configured, fake_modal):
    """Best-effort by nature: the reason we are cancelling may be that Modal is unreachable."""
    class UncancellableCall(FakeCall):
        def cancel(self):
            raise RuntimeError("modal is unreachable")

    fake_modal(UncancellableCall([]))
    pending = modal_client.spawn_stub(uuid.uuid4())

    modal_client.cancel(pending)


# --- the deployed half -------------------------------------------------------------------


def test_the_local_and_remote_halves_agree_on_the_names_they_meet_by():
    """`app.modal_client` restates these instead of importing them; this checks the copy.

    Importing them would drag `import modal` to module scope in the local half, which is the
    one thing that file is arranged to avoid. So the duplication is deliberate and this is the
    test that keeps it honest.
    """
    pytest.importorskip("modal")
    from app import modal_shadow_app

    assert modal_client.APP_NAME == modal_shadow_app.APP_NAME
    assert modal_client.FUNCTION_NAME == modal_shadow_app.FUNCTION_NAME


def test_the_remote_half_reports_the_version_the_local_half_insists_on():
    pytest.importorskip("modal")
    from app import modal_shadow_app

    observed = modal_shadow_app.run_shadow_stub.local("an-analysis")

    assert observed[modal_client.VERSION_KEY] == modal_shadow_app.WORKLOAD_VERSION
    # It runs locally here, where there is no GPU, and says so rather than failing: a stub
    # that raised when `nvidia-smi` was absent would fail runs that did execute remotely.
    assert observed["analysis_id"] == "an-analysis"
    assert observed["gpu_requested"] == modal_shadow_app.GPU
    modal_client._validated(observed)


def test_the_remote_half_imports_nothing_from_this_application():
    """`modal deploy` imports that file outside the service, where `app.config` would refuse.

    `app/__init__.py` validates the full production credential set at import. A single
    `from app...` in the deployed half would therefore mean a deploy that only works from a
    machine configured to run DeepGuard — and a container that needs those credentials to
    start.
    """
    # Tokenized rather than searched raw, for the reason `executable_source` gives: this
    # module explains at length *why* it imports nothing from `app`, and a substring search
    # would fail on the explanation and teach the next reader to delete it.
    code = executable_source(API_ROOT / "app" / "modal_shadow_app.py")

    assert "from app" not in code
    assert "import app" not in code


def test_the_remote_half_asks_for_a_gpu_r6_approved():
    pytest.importorskip("modal")
    from app import modal_shadow_app

    # R6 names L4 and A10 as the cost-effective NVIDIA starting points. Pinned so that
    # "shadow runs are cheap" stays a fact about the deployment rather than about the day it
    # was written.
    assert modal_shadow_app.GPU in {"L4", "A10", "A10G"}


# --- execution against the real queue ----------------------------------------------------


@pytest.fixture(scope="module")
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture(autouse=True)
def no_pending_run():
    """Leave the module's in-flight registry empty, before and after every test.

    `app.shadow._pending` is process state — the one piece of this design that is not a row —
    so a test that left an entry behind would change what the next test's `process_one` does
    before it reaches a line of its own.
    """
    shadow._pending.clear()
    yield
    shadow._pending.clear()


@pytest.fixture
def analysed(database):
    """One completed analysis with its media, removed again afterwards.

    Deliberately without a production signal, unlike `tests/test_shadow_mode.py`'s equivalent.
    That module asserts a *decision* survives shadow mode and needs evidence to have reached
    one; this module asserts a shadow row is written and no signal appears beside it, and an
    empty `analysis_signals` makes the second of those an equality rather than a diff.
    """
    digest = uuid.uuid4().hex

    with SessionLocal() as db:
        analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED, risk_level="MEDIUM")
        db.add(analysis)
        db.flush()

        db.add(
            MediaFile(
                analysis_id=analysis.id,
                original_filename="modal-shadow.mp4",
                content_type="video/mp4",
                size_bytes=4096,
                original_sha256=digest,
                original_storage_key=f"originals/{digest}",
                format_name="mov,mp4,m4a,3gp,3g2,mj2",
                codec_name="h264",
                width=1920,
                height=1080,
                duration=12.34,
                frame_rate=30.0,
                pix_fmt="yuv420p",
                constant_frame_rate=True,
                was_normalized=False,
                derivative_storage_key=f"originals/{digest}",
                derivative_sha256=None,
            )
        )
        db.commit()
        analysis_id = analysis.id

    yield analysis_id

    # Media and shadow runs go with it through ON DELETE CASCADE.
    with SessionLocal() as db:
        db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.commit()


@pytest.fixture
def shadow_mode(monkeypatch):
    monkeypatch.setenv(shadow.SHADOW_MODE_VARIABLE, "true")


@pytest.fixture
def remote(monkeypatch):
    """Modal faked at `app.shadow`'s seam: a spawn that records, a collect that is scripted.

    Scripted per *collect*, because the number of collects is now part of the behaviour under
    test — a run that answers on the third one is a cold start, and the two polls before it
    are two polls where the worker was free.
    """
    class Remote:
        def __init__(self):
            self.spawned = []
            self.cancelled = []
            self.collects = 0
            self.script = []

        def spawn_stub(self, analysis_id):
            self.spawned.append(analysis_id)
            return modal_client.ModalCall(
                call=object(), analysis_id=analysis_id, deadline=time.monotonic() + 60
            )

        def collect(self, pending):
            self.collects += 1
            outcome = self.script.pop(0) if self.script else None
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is None:
                return None
            return modal_client.ModalResult("modal-stub-1", outcome)

        def cancel(self, pending):
            self.cancelled.append(pending.analysis_id)

    fake = Remote()
    monkeypatch.setattr(modal_client, "spawn_stub", fake.spawn_stub)
    monkeypatch.setattr(modal_client, "collect", fake.collect)
    monkeypatch.setattr(modal_client, "cancel", fake.cancel)
    return fake


def read_run(run_id: uuid.UUID) -> ShadowRun:
    """The shadow run as the database holds it now, on a session of its own."""
    with SessionLocal() as db:
        return db.query(ShadowRun).filter(ShadowRun.id == run_id).one()


def runs_for(analysis_id: uuid.UUID) -> list[ShadowRun]:
    with SessionLocal() as db:
        return db.query(ShadowRun).filter(ShadowRun.analysis_id == analysis_id).all()


def analysis_row(analysis_id: uuid.UUID) -> tuple:
    """Everything about the analysis a shadow run is forbidden to change."""
    with SessionLocal() as db:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).one()
        return (
            analysis.status,
            analysis.risk_level,
            analysis.risk_rules_version,
            analysis.risk_rule_id,
            analysis.risk_calibration_id,
        )


def drive(limit: int = 10) -> int:
    """Poll shadow execution the way the worker loop does, until nothing is in flight.

    Returns how many steps it took. A local stub takes one; a Modal run takes at least two —
    one to start it and one to collect it — and that difference is the point of the design.
    """
    steps = 0
    while steps < limit:
        with SessionLocal() as db:
            shadow.process_one(db)
        steps += 1
        if not shadow._pending:
            return steps

    raise AssertionError("shadow execution never settled")


@pytest.mark.integration
def test_a_configured_deployment_queues_and_runs_the_workload_on_modal(
    analysed, shadow_mode, modal_configured, remote
):
    """The end-to-end shape, with only the SDK faked: queue, claim, spawn, collect, persist.

    Note what is asserted about the row: the workload name it was queued under, the version
    the *remote* half reported rather than any constant this process holds, and the document
    exactly as it came back.
    """
    before = analysis_row(analysed)
    remote.script = [None, None, remote_result(analysed)]

    with SessionLocal() as db:
        assert shadow.enqueue(db, analysed) is True

    queued = runs_for(analysed)
    assert [run.workload for run in queued] == [shadow.MODAL_STUB_WORKLOAD]
    assert queued[0].status == SHADOW_RUN_STATUS_QUEUED

    # One step to start it, then a step per collect: two that answer "not yet" and one that
    # answers.
    assert drive() == 4

    run = read_run(queued[0].id)
    assert run.status == SHADOW_RUN_STATUS_COMPLETED
    assert run.provider_version == "modal-stub-1"
    assert run.evidence["backend"] == "modal"
    assert run.evidence["gpu_attached"]
    assert run.lease_expires_at is None
    assert run.error_message is None

    assert remote.spawned == [analysed]
    assert remote.cancelled == []
    assert analysis_row(analysed) == before


@pytest.mark.integration
def test_a_run_in_flight_never_holds_the_worker_and_never_claims_a_second(
    analysed, shadow_mode, modal_configured, remote
):
    """The guarantee this design exists for, asserted from the worker's side.

    Before the spawn/collect split, `process_one` waited inside Modal's SDK for the whole
    remote execution, and a production job submitted meanwhile was claimed only once that
    returned — measured at 4.1 s behind a warm call, and a cold start or a hung Modal would
    have been far worse.

    Two things are checked, and they are different claims. That a poll with a call in flight
    is *fast* is the non-blocking property. That it claims nothing new is what keeps the
    lease and the registry honest while it waits.
    """
    with SessionLocal() as db:
        shadow.enqueue(db, analysed)

    with SessionLocal() as db:
        shadow.process_one(db)

    assert len(shadow._pending) == 1

    # Ten polls with the remote call outstanding. Each one has to come straight back.
    remote.script = [None] * 10
    started = time.monotonic()
    for _ in range(10):
        with SessionLocal() as db:
            shadow.process_one(db)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"ten idle polls took {elapsed:.2f}s with a call in flight"
    assert remote.collects == 10
    # Still exactly one call, and exactly one row: a poll that found work outstanding did not
    # go looking for more.
    assert remote.spawned == [analysed]
    assert len(runs_for(analysed)) == 1
    assert read_run(runs_for(analysed)[0].id).status == SHADOW_RUN_STATUS_PROCESSING


@pytest.mark.integration
def test_the_lease_is_renewed_on_every_poll_a_call_is_outstanding(
    analysed, shadow_mode, modal_configured, remote
):
    """A cold start longer than `SHADOW_LEASE_SECONDS` is the case this exists for.

    The lease is pushed *into the past* between polls — which is what a slow remote call does
    to it — and the next poll both renews it and finds nothing to recover. Without the
    renewal, `recover_stale_runs` would fail the row underneath a run that was going perfectly
    well and its observation would be thrown away.
    """
    with SessionLocal() as db:
        shadow.enqueue(db, analysed)
    with SessionLocal() as db:
        shadow.process_one(db)

    run_id = runs_for(analysed)[0].id

    with SessionLocal() as db:
        db.execute(
            ShadowRun.__table__.update()
            .where(ShadowRun.id == run_id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        )
        db.commit()

    remote.script = [None]
    with SessionLocal() as db:
        shadow.process_one(db)

    renewed = read_run(run_id)
    assert renewed.status == SHADOW_RUN_STATUS_PROCESSING
    assert renewed.lease_expires_at > datetime.now(timezone.utc)

    with SessionLocal() as db:
        assert shadow.recover_stale_runs(db) == 0

    # And it still finishes normally afterwards.
    remote.script = [remote_result(analysed)]
    with SessionLocal() as db:
        shadow.process_one(db)

    assert read_run(run_id).status == SHADOW_RUN_STATUS_COMPLETED


@pytest.mark.integration
def test_a_worker_that_lost_its_claim_stops_asking_and_writes_nothing(
    analysed, shadow_mode, modal_configured, remote
):
    """Renewing the lease is also the ownership check, and this is the half that matters.

    The row is failed underneath the run — a stale sweep, or another worker — and the next
    poll notices, cancels the remote call and forgets it. The observation the GPU was about
    to produce is never written, which is what stops two workers from both believing they own
    one row's result.
    """
    with SessionLocal() as db:
        shadow.enqueue(db, analysed)
    with SessionLocal() as db:
        shadow.process_one(db)

    run_id = runs_for(analysed)[0].id

    with SessionLocal() as db:
        db.execute(
            ShadowRun.__table__.update()
            .where(ShadowRun.id == run_id)
            .values(status=SHADOW_RUN_STATUS_FAILED, error_message="StaleShadowLease")
        )
        db.commit()

    remote.script = [remote_result(analysed)]
    with SessionLocal() as db:
        shadow.process_one(db)

    run = read_run(run_id)
    # The recovery's own reason, not overwritten by the loser's, and no evidence at all.
    assert run.status == SHADOW_RUN_STATUS_FAILED
    assert run.error_message == "StaleShadowLease"
    assert run.evidence is None

    # The call was let go rather than left burning GPU time for an answer nobody will read.
    assert remote.cancelled == [analysed]
    assert shadow._pending == {}


@pytest.mark.integration
def test_the_observation_is_written_to_the_shadow_row_and_to_no_other(
    analysed, shadow_mode, modal_configured, remote
):
    """Isolation, at the level of rows, for the remote backend specifically.

    R6-T1 proved no other table gains anything from a shadow run. This proves the same of a
    run whose evidence came off a rented GPU — the case where an operator might reasonably
    expect a "real" result to be treated differently.
    """
    remote.script = [remote_result(analysed)]

    with SessionLocal() as db:
        shadow.enqueue(db, analysed)
    drive()

    with SessionLocal() as db:
        signals = (
            db.query(AnalysisSignal)
            .filter(AnalysisSignal.analysis_id == analysed)
            .all()
        )

    # The forensic record gained nothing. The observation is in `shadow_runs` and there is no
    # signal, no provider and no score anywhere a report or the risk engine would read.
    assert signals == []
    assert runs_for(analysed)[0].status == SHADOW_RUN_STATUS_COMPLETED


@pytest.mark.integration
@pytest.mark.parametrize(
    "error",
    [
        modal_client.ModalUnavailable("modal is down"),
        modal_client.ModalTimeout("the cold start never ended"),
        modal_client.ModalExecutionError("the workload returned nonsense"),
    ],
)
def test_every_way_a_collect_can_fail_fails_only_the_shadow_run(
    analysed, shadow_mode, modal_configured, remote, error
):
    """The graceful-failure guarantee, once per failure mode a collect has.

    Nothing propagates, so the loop that claims production jobs never learns Modal exists, and
    the remote call is cancelled on the way out.
    """
    before = analysis_row(analysed)
    remote.script = [None, error]

    with SessionLocal() as db:
        shadow.enqueue(db, analysed)
    drive()

    run = runs_for(analysed)[0]
    assert run.status == SHADOW_RUN_STATUS_FAILED
    # The class name and not the message — a Modal error can quote an endpoint or a token id.
    assert run.error_message == type(error).__name__
    assert run.evidence is None
    assert run.lease_expires_at is None
    assert remote.cancelled == [analysed]

    assert analysis_row(analysed) == before


@pytest.mark.integration
def test_a_spawn_that_fails_fails_only_the_shadow_run(
    analysed, shadow_mode, modal_configured, monkeypatch
):
    """The other half: Modal refusing to start the work at all.

    Distinguished from a failed collect because nothing is in flight afterwards — there is no
    call to cancel and nothing for a later poll to come back to.
    """
    before = analysis_row(analysed)

    def refuse(_analysis_id):
        raise modal_client.ModalUnavailable("modal is down")

    monkeypatch.setattr(modal_client, "spawn_stub", refuse)

    with SessionLocal() as db:
        shadow.enqueue(db, analysed)
    with SessionLocal() as db:
        assert shadow.process_one(db) is True

    run = runs_for(analysed)[0]
    assert run.status == SHADOW_RUN_STATUS_FAILED
    assert run.error_message == "ModalUnavailable"
    assert run.evidence is None
    assert shadow._pending == {}
    assert analysis_row(analysed) == before


@pytest.mark.integration
def test_a_modal_run_queued_before_modal_was_turned_off_fails_rather_than_running_locally(
    analysed, shadow_mode, modal_configured, monkeypatch
):
    """Configuration removed between enqueue and execution.

    The alternative — falling back to the local stub — would put an observation the local
    stub made into a row labelled `modal-stub`, and the corpus comparing the two backends
    would be quietly wrong in the direction that makes Modal look identical to running
    locally.
    """
    with SessionLocal() as db:
        assert shadow.enqueue(db, analysed) is True

    monkeypatch.delenv(modal_client.MODAL_SHADOW_VARIABLE, raising=False)

    with SessionLocal() as db:
        assert shadow.process_one(db) is True

    run = runs_for(analysed)[0]
    assert run.workload == shadow.MODAL_STUB_WORKLOAD
    assert run.status == SHADOW_RUN_STATUS_FAILED
    assert run.error_message == "ModalNotConfigured"
    assert run.evidence is None


@pytest.mark.integration
def test_a_run_naming_a_workload_this_worker_does_not_have_fails_by_name(
    analysed, shadow_mode
):
    with SessionLocal() as db:
        db.add(
            ShadowRun(
                analysis_id=analysed,
                workload="a-workload-from-the-future",
                status=SHADOW_RUN_STATUS_QUEUED,
            )
        )
        db.commit()

    with SessionLocal() as db:
        assert shadow.process_one(db) is True

    run = runs_for(analysed)[0]
    assert run.status == SHADOW_RUN_STATUS_FAILED
    assert run.error_message == "UnknownShadowWorkload"


@pytest.mark.integration
def test_a_worker_shutting_down_lets_go_of_the_call_it_was_waiting_on(
    analysed, shadow_mode, modal_configured, remote
):
    """The row is left to its lease, exactly as if the process had died — but the GPU is not.

    Cancelling on the way out is the only part of this that is new: a container still running
    for an answer nobody will collect is GPU time billed for nothing.
    """
    with SessionLocal() as db:
        shadow.enqueue(db, analysed)
    with SessionLocal() as db:
        shadow.process_one(db)

    shadow.abandon_pending()

    assert remote.cancelled == [analysed]
    assert shadow._pending == {}
    # Still `processing`, holding the lease that `recover_stale_runs` will eventually fail.
    assert runs_for(analysed)[0].status == SHADOW_RUN_STATUS_PROCESSING


# --- nothing outside the worker knows Modal exists ---------------------------------------


def executable_source(path: Path) -> str:
    """The module with comments and string literals dropped — see `tests/test_shadow_mode.py`.

    Restated here rather than imported, for the reason that file gives about its own copies:
    a structural assertion that depends on another test module's helper fails for reasons
    that have nothing to do with what it is asserting.
    """
    dropped = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE}

    with path.open("rb") as handle:
        return " ".join(
            token.string
            for token in tokenize.tokenize(handle.readline)
            if token.type not in dropped
        )


@pytest.mark.parametrize(
    "source",
    [
        *sorted((API_ROOT / "app" / "api").rglob("*.py")),
        API_ROOT / "app" / "risk_engine.py",
        API_ROOT / "app" / "main.py",
        API_ROOT / "app" / "detection.py",
    ],
    ids=lambda path: path.name,
)
def test_no_customer_facing_module_knows_about_modal(source):
    """The execution backend is invisible from everywhere a customer is answered.

    Stronger than "the response has no Modal fields": there is no import to add a field
    through. A shadow observation is unreachable from these modules whether it was produced
    on this machine or on a rented GPU, and this is the assertion that keeps that true of the
    second case as it was of the first.
    """
    code = executable_source(source)

    for name in MODAL_NAMES:
        assert name not in code, f"{source.name} names {name!r}"


@pytest.mark.skipif(not WEB_ROOT.exists(), reason="the web application is not present")
def test_the_web_application_knows_nothing_about_modal():
    sources = [
        path
        for pattern in ("*.ts", "*.tsx")
        for path in (WEB_ROOT / "app").rglob(pattern)
    ]

    assert sources, "no web sources were found to check"

    for path in sources:
        text = path.read_text()
        for name in ("modal_client", "modal_shadow_app", "gpu_attached", "modal-stub"):
            assert name not in text, f"{path.name} names {name!r}"


def test_the_production_analysis_path_never_imports_modal():
    """`app.worker` runs the analysis and the experiments, and only the latter reaches Modal.

    Asserted as an absence of a direct import: the worker reaches shadow execution through
    `app.shadow`, which reaches Modal through `app.modal_client`, and every one of those hops
    is somewhere a failure is already swallowed. A worker importing the SDK itself would be a
    worker whose startup could fail on an experiment's dependency.
    """
    code = executable_source(API_ROOT / "app" / "worker.py")

    assert "modal" not in code
