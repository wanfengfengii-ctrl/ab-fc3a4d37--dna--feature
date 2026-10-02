"""HTTP-level integration tests for POST /api/phase and /health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def make_read(rid, start, end, obs, costs=None, allow=None, penalty=None):
    width = end - start
    r = {
        "id": rid,
        "start": start,
        "end": end,
        "observations": list(obs),
        "mismatch_costs": list(costs) if costs is not None else [1] * width,
        "max_mismatches": width if allow is None else allow,
    }
    if penalty is not None:
        r["contaminant_penalty"] = penalty
    return r


def clean_payload():
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0))
    return {"n_sites": 8, "reads": reads}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_phase_success_envelope():
    r = client.post("/api/phase", json=clean_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    data = body["data"]
    assert data["unique"] is True
    assert data["solution"]["total_mismatch_cost"] == 0
    assert len(data["solution"]["per_read"]) == 10
    # canonical first site
    assert data["solution"]["haplotype"][0] == 0
    # assignments and evidence present per read
    for row in data["solution"]["per_read"]:
        assert set(row) == {"id", "group", "mismatch_count", "mismatch_cost", "mismatch_positions"}
        assert row["group"] in (0, 1)


def test_phase_discontinuous_returns_business_code():
    payload = clean_payload()
    # shrink all group-1 reads so site 7 is uncovered (still 10 reads)
    for r in payload["reads"]:
        if r["id"] == "a4":
            r["end"] = 7
            r["observations"].pop()
            r["mismatch_costs"].pop()
        if r["id"] == "b2":
            r["start"] = 6
            r["end"] = 7
            r["observations"] = [0]
            r["mismatch_costs"] = [1]
            r["max_mismatches"] = 0
        if r["id"] == "b4":
            r["start"] = 4
            r["end"] = 7
            r["observations"] = [1, 0, 0]
            r["mismatch_costs"] = [1, 1, 1]
            r["max_mismatches"] = 0
    r = client.post("/api/phase", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "DISCONTINUOUS_INPUT"


def test_phase_no_solution_returns_business_code():
    reads = []
    pattern = [0, 0, 1, 1, 0, 1, 0, 1]
    other = [1, 1, 0, 0, 1, 0, 1, 0]
    third = [0, 1, 0, 1, 0, 1, 0, 1]
    for j, p in enumerate([pattern, other, third]):
        reads.append(make_read(f"x{j}", 0, 8, p, allow=0))
    for j in range(7):
        p = pattern if j % 2 == 0 else other
        reads.append(make_read(f"p{j}", 0, 8, p, allow=0))
    r = client.post("/api/phase", json={"n_sites": 8, "reads": reads})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "NO_SOLUTION"


def test_phase_invalid_input_422():
    payload = clean_payload()
    payload["n_sites"] = 5
    r = client.post("/api/phase", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_INPUT"


def test_bad_json_400():
    r = client.post("/api/phase", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "BAD_JSON"


def test_ambiguous_payload_reports_two_solutions():
    reads = []
    a_specs = [
        ("a0", 0, 2, [0, 0]),
        ("a1", 1, 4, [0, 1, 1]),
        ("a2", 0, 3, [0, 0, 1]),
        ("a3", 2, 4, [0, 0]),
        ("a4", 0, 2, [1, 1]),
    ]
    b_specs = [
        ("b0", 4, 6, [0, 0]),
        ("b1", 5, 8, [0, 1, 1]),
        ("b2", 4, 7, [0, 0, 1]),
        ("b3", 6, 8, [0, 0]),
        ("b4", 4, 6, [1, 1]),
    ]
    for rid, s, e, obs in a_specs + b_specs:
        reads.append(make_read(rid, s, e, obs, allow=0))
    r = client.post("/api/phase", json={"n_sites": 8, "reads": reads})
    body = r.json()
    assert r.status_code == 200
    assert body["data"]["unique"] is False
    assert len(body["data"]["solutions"]) == 2
    assert body["data"]["note"]


# --------------------------------------------------------------------------
# contaminant mode
# --------------------------------------------------------------------------


def contaminant_payload():
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0, penalty=100))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0, penalty=100))
    # a read that is impossible under allowance 0 on both homologous sides
    reads.append(make_read("c0", 0, 4, [1, 0, 1, 0], costs=[9] * 4, allow=0, penalty=5))
    reads.append(make_read("a5", 0, 4, hap[0:4], allow=0, penalty=100))
    return reads


def test_phase_contaminant_success_envelope():
    r = client.post(
        "/api/phase",
        json={"n_sites": 8, "reads": contaminant_payload(), "max_contaminant_reads": 2},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    sol = data["solution"]
    assert data["max_contaminant_reads"] == 2
    assert sol["contaminant_reads"] == ["c0"]
    assert sol["groups"]["contaminants"] == ["c0"]
    assert sol["contaminant_count"] == 1
    assert sol["total_mismatch_cost"] == 0
    assert sol["total_contaminant_penalty"] == 5
    assert sol["total_objective_cost"] == 5
    # ternary assignments
    assert all(a in (0, 1, 2) for a in sol["assignments"])
    c0 = next(row for row in sol["per_read"] if row["id"] == "c0")
    assert c0["group"] is None
    assert c0["contaminant"] is True
    assert c0["status"] == "contaminant"
    assert c0["contaminant_penalty"] == 5
    assert c0["contaminant_reason"]["allowed_mismatches"] == 0
    assert c0["contaminant_reason"]["mismatches_vs_haplotype"]["count"] == 2
    assert c0["contaminant_reason"]["mismatches_vs_complement"]["count"] == 2
    # non-contaminant per-read rows carry the same fields plus flags
    a0 = next(row for row in sol["per_read"] if row["id"] == "a0")
    assert a0["group"] == 0
    assert a0["contaminant"] is False
    assert a0["mismatch_count"] == 0


def test_phase_contaminant_cap_too_small_409():
    reads = contaminant_payload()
    r = client.post(
        "/api/phase", json={"n_sites": 8, "reads": reads, "max_contaminant_reads": 0}
    )
    assert r.status_code == 422  # 0 is an invalid cap value
    assert r.json()["error"]["code"] == "INVALID_INPUT"

    # two forced contaminants under cap 1 -> specific business error
    reads2 = [dict(r) for r in reads]
    reads2.append(make_read("c1", 4, 8, [0, 1, 0, 1], costs=[9] * 4, allow=0, penalty=5))
    r = client.post(
        "/api/phase", json={"n_sites": 8, "reads": reads2, "max_contaminant_reads": 1}
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "INSUFFICIENT_CONTAMINANT_CAPACITY"


def test_phase_contaminant_missing_penalty_422():
    reads = contaminant_payload()
    reads[0].pop("contaminant_penalty")
    r = client.post(
        "/api/phase", json={"n_sites": 8, "reads": reads, "max_contaminant_reads": 2}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_INPUT"

    # cap absent but penalty present -> also rejected
    reads2 = [
        {k: v for k, v in row.items() if k != "contaminant_penalty"}
        for row in contaminant_payload()
    ]
    reads2[0]["contaminant_penalty"] = 7
    r = client.post("/api/phase", json={"n_sites": 8, "reads": reads2})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_INPUT"


def test_phase_contaminant_group_evidence_409():
    p = [0, 0, 1, 0, 0, 1, 0, 1]
    rpat = [0, 0, 1, 1, 0, 1, 0, 1]
    reads = [make_read(f"p{j}", 0, 8, p, allow=0, penalty=100) for j in range(8)]
    reads += [make_read(f"r{j}", 0, 8, rpat, allow=0, penalty=1) for j in range(2)]
    r = client.post(
        "/api/phase", json={"n_sites": 8, "reads": reads, "max_contaminant_reads": 2}
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "INSUFFICIENT_GROUP_EVIDENCE"


def test_phase_legacy_request_has_no_contaminant_fields():
    r = client.post("/api/phase", json=clean_payload())
    assert r.status_code == 200
    data = r.json()["data"]
    sol = data["solution"]
    assert "contaminant_reads" not in sol
    assert "contaminant_count" not in sol
    assert "total_contaminant_penalty" not in sol
    assert "total_objective_cost" not in data["solution"]
    assert "max_contaminant_reads" not in data
    assert "contaminants" not in sol["groups"]
    for row in sol["per_read"]:
        assert "contaminant" not in row
        assert "contaminant_reason" not in row
