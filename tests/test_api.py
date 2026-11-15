"""Test suite for the GTFS publication API."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app import store as store_module
from app.main import app
from app.store import Store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_zip(files: dict[str, str]) -> bytes:
    """Deterministic ZIP: fixed member timestamps, so repeated calls with the
    same files produce identical bytes (several tests re-generate the upload
    and compare it against the downloaded artifact)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, text)
    return buffer.getvalue()


BASE_FILES = {
    "stops.txt": "stop_id,stop_name\nS1,Central\nS2,People Square\nS3,Science Hall\n",
    "trips.txt": "route_id,trip_id\nR1,T1\nR1,T2\n",
    "stop_times.txt": (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,06:10:00,S2,2\n"
        "T1,24:30:00,S3,3\n"
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    ),
}


def zip_bytes(files: dict[str, str] | None = None) -> bytes:
    return make_zip(files if files is not None else BASE_FILES)


def upload(client: TestClient, data: bytes | None = None, name: str = "feed.zip"):
    return client.post(
        "/api/v1/versions",
        files={"file": (name, data if data is not None else zip_bytes(), "application/zip")},
    )


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture()
def client(data_dir: Path):
    store = Store(data_dir)
    store.init()
    main_module.set_store(store)
    with TestClient(app) as test_client:
        yield test_client
    main_module._store = None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_publish_and_query_by_current_and_version(client: TestClient):
    response = upload(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"]["version_id"] == 1
    assert body["previous_version"] is None
    assert body["version"]["stops_count"] == 3
    assert body["version"]["trips_count"] == 2
    assert body["version"]["stop_times_count"] == 5

    current = client.get("/api/v1/trips/T1/stops")
    assert current.status_code == 200
    payload = current.json()
    assert payload["version_id"] == 1
    assert payload["trip_id"] == "T1"
    assert [s["stop_sequence"] for s in payload["stops"]] == [1, 2, 3]
    assert [s["stop_name"] for s in payload["stops"]] == [
        "Central",
        "People Square",
        "Science Hall",
    ]
    assert payload["stops"][2]["arrival_time"] == "24:30:00"
    assert payload["stops"][2]["arrival_seconds"] == 24 * 3600 + 30 * 60

    by_version = client.get("/api/v1/versions/1/trips/T1/stops")
    assert by_version.status_code == 200
    assert by_version.json() == payload

    alias = client.get("/api/v1/versions/current/trips/T1/stops")
    assert alias.status_code == 200
    assert alias.json()["version_id"] == 1


def test_second_version_keeps_first_queryable(client: TestClient):
    assert upload(client).status_code == 201

    v2_files = {
        "stops.txt": "stop_id,stop_name\nS1,Central\nS2,People Square\nS3,Science Hall\nS9,New Town\n",
        "trips.txt": "route_id,trip_id\nR1,T1\nR1,T2\nR9,T9\n",
        "stop_times.txt": (
            "trip_id,arrival_time,stop_id,stop_sequence\n"
            "T1,06:00:00,S1,1\n"
            "T1,06:20:00,S3,2\n"          # express: skips S2
            "T2,07:00:00,S1,1\n"
            "T2,07:15:00,S3,2\n"
            "T9,25:10:00,S9,1\n"
            "T9,25:30:00,S1,2\n"
        ),
    }
    response = upload(client, zip_bytes(v2_files))
    assert response.status_code == 201, response.text
    assert response.json()["version"]["version_id"] == 2
    assert response.json()["previous_version"] == 1

    v1 = client.get("/api/v1/trips/T1/stops?version=1").json()
    v2 = client.get("/api/v1/trips/T1/stops?version=2").json()
    current = client.get("/api/v1/trips/T1/stops").json()

    assert [s["stop_id"] for s in v1["stops"]] == ["S1", "S2", "S3"]
    assert [s["stop_id"] for s in v2["stops"]] == ["S1", "S3"]
    assert current["version_id"] == 2
    assert current == v2

    # Old version's data is unchanged, and its original ZIP still downloads.
    download = client.get("/api/v1/versions/1/download")
    assert download.status_code == 200
    assert download.content == zip_bytes()

    listing = client.get("/api/v1/versions").json()
    assert listing["current_version"] == 2
    assert [v["version_id"] for v in listing["versions"]] == [2, 1]


def test_download_matches_uploaded_bytes_and_digest(client: TestClient):
    data = zip_bytes()
    published = upload(client, data).json()["version"]
    assert published["size_bytes"] == len(data)
    assert published["sha256"] == hashlib.sha256(data).hexdigest()

    response = client.get("/api/v1/versions/1/download")
    assert response.status_code == 200
    assert response.content == data
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-content-sha256"] == published["sha256"]
    assert 'filename="gtfs-version-1.zip"' in response.headers["content-disposition"]


def test_gtfs_times_past_24_hours_are_accepted(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,23:50:00,S1,1\n"
        "T1,24:05:00,S2,2\n"
        "T1,25:30:00,S3,3\n"
        "T2,27:00:00,S1,1\n"
        "T2,27:15:00,S3,2\n"
    )
    response = upload(client, zip_bytes(files))
    assert response.status_code == 201, response.text
    stops = client.get("/api/v1/trips/T1/stops").json()["stops"]
    assert [s["arrival_time"] for s in stops] == ["23:50:00", "24:05:00", "25:30:00"]


def test_rows_may_be_unordered_in_file(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:10:00,S2,2\n"
        "T1,06:00:00,S1,1\n"
        "T1,24:30:00,S3,3\n"
        "T2,07:15:00,S3,2\n"
        "T2,07:00:00,S1,1\n"
    )
    assert upload(client, zip_bytes(files)).status_code == 201
    stops = client.get("/api/v1/trips/T1/stops").json()["stops"]
    assert [s["stop_sequence"] for s in stops] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def _errors(response) -> list[dict]:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error"] == "validation_failed"
    return body["errors"]


def _assert_error(errors, file, line, code):
    assert any(
        e["file"] == file and e["line"] == line and e["code"] == code for e in errors
    ), errors


def test_duplicate_stop_id_reports_line(client: TestClient):
    files = dict(BASE_FILES)
    files["stops.txt"] = (
        "stop_id,stop_name\nS1,Central\nS2,People Square\nS1,Duplicate\n"
    )
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stops.txt", 4, "duplicate_stop_id")


def test_duplicate_trip_id_reports_line(client: TestClient):
    files = dict(BASE_FILES)
    files["trips.txt"] = "route_id,trip_id\nR1,T1\nR1,T2\nR2,T1\n"
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "trips.txt", 4, "duplicate_trip_id")


def test_unknown_stop_and_trip_report_lines(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,06:10:00,S99,2\n"       # line 3
        "TX,06:20:00,S2,3\n"        # line 4
        "T1,06:30:00,S2,4\n"
    )
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stop_times.txt", 3, "unknown_stop_id")
    _assert_error(errors, "stop_times.txt", 4, "unknown_trip_id")


def test_missing_arrival_time_reports_line(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,,S2,2\n"
        "T1,06:30:00,S3,3\n"
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    )
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stop_times.txt", 3, "empty_arrival_time")


def test_invalid_arrival_time_reports_line(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,6:70:00,S2,2\n"
        "T1,06:30:00,S3,3\n"
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    )
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stop_times.txt", 3, "invalid_arrival_time")


def test_duplicate_sequence_within_trip_reports_line(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,06:10:00,S2,2\n"
        "T1,06:20:00,S3,2\n"       # line 4: duplicate sequence in T1
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    )
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stop_times.txt", 4, "duplicate_sequence")


def test_same_sequence_for_different_trips_is_fine(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T2,06:00:00,S1,1\n"
        "T1,06:10:00,S2,2\n"
        "T2,06:10:00,S2,2\n"
        "T1,06:20:00,S3,3\n"
        "T2,06:20:00,S3,3\n"
    )
    assert upload(client, zip_bytes(files)).status_code == 201


def test_regressing_arrival_time_reports_later_row(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,06:10:00,S2,2\n"
        "T1,06:05:00,S3,3\n"       # line 4: earlier than stop 2
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    )
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stop_times.txt", 4, "arrival_time_regression")


def test_equal_arrival_times_are_allowed(client: TestClient):
    files = dict(BASE_FILES)
    files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,06:00:00,S2,2\n"
        "T1,06:10:00,S3,3\n"
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    )
    assert upload(client, zip_bytes(files)).status_code == 201


def test_missing_required_file_reports_file_name(client: TestClient):
    files = dict(BASE_FILES)
    del files["trips.txt"]
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "trips.txt", 0, "missing_file")


def test_trip_without_stop_times_reports_line(client: TestClient):
    files = dict(BASE_FILES)
    files["trips.txt"] = "route_id,trip_id\nR1,T1\nR1,T2\nR2,T3\n"
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "trips.txt", 4, "trip_without_stops")


def test_optional_route_column_may_be_absent(client: TestClient):
    files = dict(BASE_FILES)
    files["trips.txt"] = "trip_id\nT1\nT2\n"
    response = upload(client, zip_bytes(files))
    assert response.status_code == 201, response.text
    assert client.get("/api/v1/trips/T1/stops").json()["route_id"] is None


def test_missing_required_column_reports_header_line(client: TestClient):
    files = dict(BASE_FILES)
    files["stops.txt"] = "stop_id\nS1\nS2\n"
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stops.txt", 1, "missing_column")


def test_not_a_zip_is_rejected(client: TestClient):
    response = upload(client, b"this is definitely not a zip archive")
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert response.json()["errors"][0]["code"] == "invalid_archive"


def test_empty_upload_is_rejected(client: TestClient):
    response = upload(client, b"")
    assert response.status_code == 422


def test_rejected_upload_does_not_change_versions(client: TestClient):
    assert upload(client).status_code == 201
    before = client.get("/api/v1/versions").json()

    broken = dict(BASE_FILES)
    broken["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,06:10:00,S404,2\n"
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    )
    assert upload(client, zip_bytes(broken)).status_code == 422

    after = client.get("/api/v1/versions").json()
    assert after == before
    assert client.get("/api/v1/versions/1/download").content == zip_bytes()

    store = main_module.get_store()
    assert list(store.staging_dir.iterdir()) == []
    assert len(list(store.artifacts_dir.iterdir())) == 1


def test_header_comment_and_blank_lines_kept_out_of_line_numbers(client: TestClient):
    files = dict(BASE_FILES)
    files["stops.txt"] = (
        "stop_id,stop_name\n"
        "\n"
        "# a comment\n"
        "S1,Central\n"
        "\n"
        "S2,People Square\n"
        "S3,Science Hall\n"
        "S1,Duplicate\n"
    )
    errors = _errors(upload(client, zip_bytes(files)))
    _assert_error(errors, "stops.txt", 8, "duplicate_stop_id")


def test_unknown_version_and_trip_return_404(client: TestClient):
    assert upload(client).status_code == 201
    response = client.get("/api/v1/versions/99/trips/T1/stops")
    assert response.status_code == 404
    assert response.json()["error"] == "version_not_found"

    response = client.get("/api/v1/trips/NOPE/stops")
    assert response.status_code == 404
    assert response.json()["error"] == "trip_not_found"

    response = client.get("/api/v1/versions/abc/trips/T1/stops")
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_version"


def test_no_versions_yet(client: TestClient):
    response = client.get("/api/v1/trips/T1/stops")
    assert response.status_code == 404
    assert response.json()["error"] == "no_published_version"
    listing = client.get("/api/v1/versions")
    assert listing.status_code == 200
    assert listing.json() == {"current_version": None, "versions": []}


# ---------------------------------------------------------------------------
# Failure injection / durability / concurrency
# ---------------------------------------------------------------------------


def test_failed_registration_leaves_no_partial_version(client: TestClient, monkeypatch):
    assert upload(client).status_code == 201
    before = client.get("/api/v1/versions").json()
    store = main_module.get_store()

    def explode(*args, **kwargs):
        raise RuntimeError("simulated disk/database failure")

    monkeypatch.setattr(store, "_insert_rows", explode)
    response = upload(client)
    monkeypatch.undo()

    assert response.status_code == 500
    assert response.json()["error"] == "publication_failed"
    assert client.get("/api/v1/versions").json() == before
    # No residue: staging is empty and only the committed artifact remains.
    assert list(store.staging_dir.iterdir()) == []
    assert len(list(store.artifacts_dir.iterdir())) == 1
    # The old version is still fully readable.
    assert client.get("/api/v1/versions/1/trips/T1/stops").status_code == 200
    assert client.get("/api/v1/versions/1/download").content == zip_bytes()


def test_versions_survive_restart(data_dir: Path):
    first = Store(data_dir)
    first.init()
    main_module.set_store(first)
    with TestClient(app) as c:
        assert upload(c).status_code == 201
        assert upload(c).status_code == 201
    main_module._store = None

    second = Store(data_dir)
    second.init()
    main_module.set_store(second)
    with TestClient(app) as c:
        listing = c.get("/api/v1/versions").json()
        assert listing["current_version"] == 2
        assert [v["version_id"] for v in listing["versions"]] == [2, 1]
        assert c.get("/api/v1/versions/1/trips/T1/stops").status_code == 200
        assert c.get("/api/v1/versions/1/download").content == zip_bytes()
    main_module._store = None


def test_concurrent_uploads_all_become_versions(client: TestClient):
    payloads = []
    for index in range(5):
        files = dict(BASE_FILES)
        files["stops.txt"] = (
            "stop_id,stop_name\nS1,Central\nS2,People Square\nS3,Science Hall\n"
            f"S{100+index},Extra {index}\n"
        )
        payloads.append(zip_bytes(files))

    results = [None] * len(payloads)
    barrier = threading.Barrier(len(payloads))

    def worker(i):
        barrier.wait()
        results[i] = upload(client, payloads[i], name=f"feed-{i}.zip")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(payloads))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(r.status_code == 201 for r in results), [r.text for r in results]
    version_ids = sorted(r.json()["version"]["version_id"] for r in results)
    assert version_ids == [1, 2, 3, 4, 5]

    listing = client.get("/api/v1/versions").json()
    assert listing["current_version"] == 5
    assert len(listing["versions"]) == 5

    # Every version keeps its own data and its own artifact.
    for result in results:
        info = result.json()["version"]
        download = client.get(f"/api/v1/versions/{info['version_id']}/download")
        assert download.status_code == 200
        assert hashlib.sha256(download.content).hexdigest() == info["sha256"]


def test_queries_never_see_a_mixed_version(client: TestClient):
    """Readers must see a coherent snapshot while publishes keep happening."""
    # Seed two large-ish versions so commits take measurable time.
    def big_files(marker: str) -> dict[str, str]:
        stops = ["stop_id,stop_name"] + [f"S{i},{marker}-stop-{i}" for i in range(200)]
        trips = ["route_id,trip_id"] + [f"R,T{i}" for i in range(30)]
        times = ["trip_id,arrival_time,stop_id,stop_sequence"]
        for trip in range(30):
            for seq in range(200):
                hh = 6 + (seq // 60)
                mm = seq % 60
                times.append(f"T{trip},{hh:02d}:{mm:02d}:00,S{seq},{seq+1}")
        return {
            "stops.txt": "\n".join(stops) + "\n",
            "trips.txt": "\n".join(trips) + "\n",
            "stop_times.txt": "\n".join(times) + "\n",
        }

    assert upload(client, zip_bytes(big_files("v1"))).status_code == 201

    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            response = client.get("/api/v1/trips/T0/stops")
            if response.status_code != 200:
                errors.append(response.text)
                return
            body = response.json()
            names = {s["stop_name"].split("-")[0] for s in body["stops"]}
            if len(names) != 1:
                errors.append(f"mixed version snapshot: {names}")
                return

    def publisher():
        for index in range(2, 6):
            response = upload(client, zip_bytes(big_files(f"v{index}")))
            if response.status_code != 201:
                errors.append(response.text)
                return

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    try:
        publisher()
    finally:
        stop.set()
        reader_thread.join()

    assert errors == []
    assert client.get("/api/v1/versions").json()["current_version"] == 5


def test_tampered_artifact_is_refused(client: TestClient):
    assert upload(client).status_code == 201
    store = main_module.get_store()
    artifact = next(store.artifacts_dir.iterdir())
    original = artifact.read_bytes()
    artifact.write_bytes(original + b"tampered")

    response = client.get("/api/v1/versions/1/download")
    assert response.status_code == 409
    assert response.json()["error"] == "artifact_digest_mismatch"
    # Queries are unaffected by a broken artifact.
    assert client.get("/api/v1/versions/1/trips/T1/stops").status_code == 200


def test_missing_artifact_is_refused(client: TestClient):
    assert upload(client).status_code == 201
    store = main_module.get_store()
    next(store.artifacts_dir.iterdir()).unlink()

    response = client.get("/api/v1/versions/1/download")
    assert response.status_code == 410
    assert response.json()["error"] == "artifact_missing"
    assert response.headers["cache-control"] == "no-store"


def test_startup_sweeps_unreferenced_artifacts_but_keeps_versions(data_dir: Path):
    store = Store(data_dir)
    store.init()
    main_module.set_store(store)
    with TestClient(app) as c:
        assert upload(c).status_code == 201
    main_module._store = None

    orphan = data_dir / "artifacts" / "orphan.zip"
    orphan.write_bytes(b"left over from a crashed upload")
    os.utime(orphan, (0, 0))
    stale_staging = data_dir / "staging" / "stale.zip"
    stale_staging.write_bytes(b"partial upload")

    restarted = Store(data_dir)
    restarted.init()
    main_module.set_store(restarted)
    with TestClient(app) as c:
        assert not orphan.exists()
        assert not stale_staging.exists()
        assert c.get("/api/v1/versions/1/download").status_code == 200
    main_module._store = None


def test_download_of_prior_version_after_publish(client: TestClient):
    first_payload = zip_bytes()
    assert upload(client, first_payload).status_code == 201
    assert upload(client).status_code == 201
    response = client.get("/api/v1/versions/1/download")
    assert response.status_code == 200
    assert response.content == first_payload
    assert response.headers["content-length"] == str(len(first_payload))


def test_openapi_and_docs_available(client: TestClient):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/healthz").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Activation (rolling dispatching back to a previously published version)
# ---------------------------------------------------------------------------


def activate(client: TestClient, version, expected_current):
    return client.post(
        f"/api/v1/versions/{version}/activate",
        json={"expected_current_version": expected_current},
    )


def test_activate_previous_version_rolls_dispatch_back(client: TestClient):
    assert upload(client).status_code == 201
    v2_files = dict(BASE_FILES)
    v2_files["stop_times.txt"] = (
        "trip_id,arrival_time,stop_id,stop_sequence\n"
        "T1,06:00:00,S1,1\n"
        "T1,06:20:00,S3,2\n"          # v2: express skips S2
        "T2,07:00:00,S1,1\n"
        "T2,07:15:00,S3,2\n"
    )
    assert upload(client, zip_bytes(v2_files)).status_code == 201
    assert client.get("/api/v1/versions").json()["current_version"] == 2

    response = activate(client, 1, expected_current=2)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["current_version"] == 1
    assert body["previous_version"] == 2
    assert body["version"]["version_id"] == 1

    # Dispatch queries now see v1 again; the listing agrees.
    current = client.get("/api/v1/trips/T1/stops").json()
    assert current["version_id"] == 1
    assert [s["stop_id"] for s in current["stops"]] == ["S1", "S2", "S3"]
    assert client.get("/api/v1/versions").json()["current_version"] == 1
    assert client.get("/api/v1/versions/current").json()["version_id"] == 1

    # The rolled-back-from version is still queryable and downloadable.
    assert client.get("/api/v1/trips/T1/stops?version=2").json()["version_id"] == 2
    assert client.get("/api/v1/versions/2/download").status_code == 200


def test_activate_with_stale_expected_current_is_rejected(client: TestClient):
    assert upload(client).status_code == 201
    assert upload(client).status_code == 201

    response = activate(client, 1, expected_current=1)  # actual current is 2
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "current_version_conflict"
    assert body["current_version"] == 2
    assert body["expected_current_version"] == 1

    # State is untouched.
    assert client.get("/api/v1/versions").json()["current_version"] == 2
    assert client.get("/api/v1/trips/T1/stops").json()["version_id"] == 2


def test_activate_with_null_expected_is_rejected_when_versions_exist(client: TestClient):
    assert upload(client).status_code == 201
    response = activate(client, 1, expected_current=None)
    assert response.status_code == 409
    assert response.json()["error"] == "current_version_conflict"
    assert client.get("/api/v1/versions").json()["current_version"] == 1


def test_activate_missing_artifact_keeps_state(client: TestClient):
    assert upload(client).status_code == 201
    assert upload(client).status_code == 201
    store = main_module.get_store()
    info_v1, _ = store.get_version("1")
    (store.artifacts_dir / info_v1.artifact_name).unlink()

    response = activate(client, 1, expected_current=2)
    assert response.status_code == 410
    assert response.json()["error"] == "artifact_missing"
    assert client.get("/api/v1/versions").json()["current_version"] == 2


def test_activate_tampered_artifact_keeps_state(client: TestClient):
    assert upload(client).status_code == 201
    assert upload(client).status_code == 201
    store = main_module.get_store()
    info_v1, _ = store.get_version("1")
    artifact = store.artifacts_dir / info_v1.artifact_name
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    response = activate(client, 1, expected_current=2)
    assert response.status_code == 409
    assert response.json()["error"] == "artifact_digest_mismatch"
    assert client.get("/api/v1/versions").json()["current_version"] == 2


def test_activate_unknown_or_invalid_target(client: TestClient):
    assert upload(client).status_code == 201

    response = activate(client, 99, expected_current=1)
    assert response.status_code == 404
    assert response.json()["error"] == "version_not_found"

    # 'current' is not a valid activation target; neither is junk.
    for bad in ("current", "abc", "0", "-1"):
        response = activate(client, bad, expected_current=1)
        assert response.status_code == 400, bad
        assert response.json()["error"] == "invalid_version"

    assert client.get("/api/v1/versions").json()["current_version"] == 1


def test_reactivating_current_version_is_a_noop_success(client: TestClient):
    assert upload(client).status_code == 201
    response = activate(client, 1, expected_current=1)
    assert response.status_code == 200
    assert response.json()["current_version"] == 1
    assert response.json()["previous_version"] == 1


def test_concurrent_activations_only_one_wins(client: TestClient):
    assert upload(client).status_code == 201
    assert upload(client).status_code == 201
    assert upload(client).status_code == 201

    results = []
    barrier = threading.Barrier(2)

    def worker(target):
        barrier.wait()
        results.append((target, activate(client, target, expected_current=3)))

    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(2,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    statuses = sorted(response.status_code for _, response in results)
    assert statuses == [200, 409], [(t, r.text) for t, r in results]
    conflict = next(r for _, r in results if r.status_code == 409)
    assert conflict.json()["error"] == "current_version_conflict"
    # The single committed activation decided the current version.
    winner = next(t for t, r in results if r.status_code == 200)
    assert client.get("/api/v1/versions").json()["current_version"] == winner
    assert client.get("/api/v1/trips/T1/stops").json()["version_id"] == winner


def test_activation_serialized_across_store_instances(data_dir: Path):
    """Two Store objects on one database behave like two service instances."""
    first = Store(data_dir)
    first.init()
    main_module.set_store(first)
    with TestClient(app) as c:
        assert upload(c).status_code == 201
        assert upload(c).status_code == 201
    main_module._store = None

    second = Store(data_dir)
    second.init()

    outcomes = []
    barrier = threading.Barrier(2)

    def worker(store):
        barrier.wait()
        try:
            store.activate(1, expected_current=2)
            outcomes.append("activated")
        except store_module.CurrentVersionConflict:
            outcomes.append("conflict")

    threads = [
        threading.Thread(target=worker, args=(first,)),
        threading.Thread(target=worker, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["activated", "conflict"]
    assert first.resolve_version("current") == 1
    assert second.resolve_version("current") == 1


def test_activation_survives_restart(data_dir: Path):
    first = Store(data_dir)
    first.init()
    main_module.set_store(first)
    with TestClient(app) as c:
        assert upload(c).status_code == 201
        assert upload(c).status_code == 201
        assert activate(c, 1, expected_current=2).status_code == 200
    main_module._store = None

    restarted = Store(data_dir)
    restarted.init()
    main_module.set_store(restarted)
    with TestClient(app) as c:
        listing = c.get("/api/v1/versions").json()
        assert listing["current_version"] == 1
        assert c.get("/api/v1/trips/T1/stops").json()["version_id"] == 1
    main_module._store = None


def test_publish_after_manual_activation_still_auto_activates(client: TestClient):
    assert upload(client).status_code == 201
    assert upload(client).status_code == 201
    assert activate(client, 1, expected_current=2).status_code == 200
    assert client.get("/api/v1/versions").json()["current_version"] == 1

    response = upload(client)
    assert response.status_code == 201
    body = response.json()
    assert body["version"]["version_id"] == 3
    assert body["previous_version"] == 1  # the genuinely current version
    assert client.get("/api/v1/versions").json()["current_version"] == 3
    assert client.get("/api/v1/trips/T1/stops").json()["version_id"] == 3


def test_current_pointer_migrated_for_preexisting_db(data_dir: Path):
    """Databases written before the meta table existed adopt MAX(version)."""
    store = Store(data_dir)
    store.init()
    main_module.set_store(store)
    with TestClient(app) as c:
        assert upload(c).status_code == 201
        assert upload(c).status_code == 201
        assert activate(c, 1, expected_current=2).status_code == 200
    main_module._store = None

    # Simulate a pre-activation database: no meta row at all.
    import sqlite3

    conn = sqlite3.connect(data_dir / "gtfs.db")
    conn.execute("DELETE FROM meta WHERE key = 'current_version_id'")
    conn.commit()
    conn.close()

    migrated = Store(data_dir)
    migrated.init()
    main_module.set_store(migrated)
    with TestClient(app) as c:
        # Falls back to the old implicit rule: highest version is current.
        assert c.get("/api/v1/versions").json()["current_version"] == 2
    main_module._store = None
