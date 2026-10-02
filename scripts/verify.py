#!/usr/bin/env python3
"""One-shot verification for the phasing service.

Runs inside the image (or a local checkout) and performs, in order:

1. code tests  -> ``pytest`` (if the tests directory is available);
2. waits for the API at ``$API_BASE_URL`` to report healthy;
3. API smoke checks covering:
     * a unique solution with degraded (mismatch-carrying) reads and exact
       mismatch evidence,
     * an ambiguous instance returning the first two distinct solutions,
     * a contaminant instance that isolates an environmental outlier via a
       joint three-way labelling, plus both contaminant business errors,
     * NO_SOLUTION and DISCONTINUOUS_INPUT business errors,
     * INVALID_INPUT shape validation (legacy and contaminant modes).

The exit code is a bit-mask summarizing the stages:

    bit 0 (1)  - code tests failed
    bit 1 (2)  - API smoke checks failed
    bit 2 (4)  - reserved for the image-build stage (set by verify_all.sh)

Exits 0 only when every stage it owns is green.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
HEALTH_TIMEOUT_S = float(os.environ.get("HEALTH_TIMEOUT_S", "60"))

FAIL_TESTS = 1
FAIL_SMOKE = 2


# --------------------------------------------------------------------------
# sample builders
# --------------------------------------------------------------------------


def make_read(rid, start, end, obs, costs=None, allow=None, penalty=None):
    width = end - start
    row = {
        "id": rid,
        "start": start,
        "end": end,
        "observations": list(obs),
        "mismatch_costs": list(costs) if costs is not None else [1] * width,
        "max_mismatches": width if allow is None else allow,
    }
    if penalty is not None:
        row["contaminant_penalty"] = penalty
    return row


def mismatch_sample():
    """Unique phasing with exactly two degraded reads (costs 7 and 3)."""
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0))

    # damage read a0 at site 0 (cost 7) and read b2 at its second observed
    # site = global site 7 (cost 3); allowances raised to exactly 1
    reads[0]["observations"][0] ^= 1
    reads[0]["mismatch_costs"] = [7, 1, 1]
    reads[0]["max_mismatches"] = 1
    reads[7]["observations"][1] ^= 1
    reads[7]["mismatch_costs"] = [5, 3]
    reads[7]["max_mismatches"] = 1
    return {"n_sites": 8, "reads": reads}


def ambiguous_sample():
    """Two unlinked blocks -> two distinct zero-cost canonical solutions."""
    specs = [
        ("a0", 0, 2, [0, 0]),
        ("a1", 1, 4, [0, 1, 1]),
        ("a2", 0, 3, [0, 0, 1]),
        ("a3", 2, 4, [0, 0]),
        ("a4", 0, 2, [1, 1]),
        ("b0", 4, 6, [0, 0]),
        ("b1", 5, 8, [0, 1, 1]),
        ("b2", 4, 7, [0, 0, 1]),
        ("b3", 6, 8, [0, 0]),
        ("b4", 4, 6, [1, 1]),
    ]
    reads = [make_read(rid, s, e, obs, allow=0) for rid, s, e, obs in specs]
    return {"n_sites": 8, "reads": reads}


def no_solution_sample():
    pattern = [0, 0, 1, 1, 0, 1, 0, 1]
    other = [1, 1, 0, 0, 1, 0, 1, 0]
    third = [0, 1, 0, 1, 0, 1, 0, 1]
    reads = [make_read(f"x{j}", 0, 8, p, allow=0) for j, p in enumerate([pattern, other, third])]
    for j in range(7):
        p = pattern if j % 2 == 0 else other
        reads.append(make_read(f"p{j}", 0, 8, p, allow=0))
    return {"n_sites": 8, "reads": reads}


def discontinuous_sample():
    spans = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 3), (1, 4), (2, 5), (7, 8), (0, 4)]
    return {
        "n_sites": 8,
        "reads": [make_read(f"d{j}", s, e, [0] * (e - s), allow=e - s) for j, (s, e) in enumerate(spans)],
    }


def contaminant_sample():
    """Ten clean reads plus one foreign read isolated as a contaminant."""
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0, penalty=10))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0, penalty=10))
    reads.append(make_read("junk", 0, 5, [1, 0, 1, 0, 1], allow=0, penalty=4))
    return {"n_sites": 8, "reads": reads, "max_contaminant_reads": 1}


def contaminant_capacity_short_sample():
    """Two foreign reads but only one contaminant slot -> business error."""
    payload = contaminant_sample()
    payload["reads"].append(
        make_read("junk2", 3, 8, [0, 1, 0, 1, 0], allow=0, penalty=4)
    )
    return payload


def contaminant_evidence_short_sample():
    """Only one side can be supported, regardless of the contaminant cap."""
    p1 = [0, 0, 1, 1, 0, 1, 0, 1]
    p3 = [0, 1, 0, 1, 0, 1, 0, 1]
    reads = [make_read(f"a{i}", 0, 8, p1, allow=0, penalty=5) for i in range(7)]
    reads += [make_read(f"c{i}", 0, 8, p3, allow=0, penalty=5) for i in range(3)]
    return {"n_sites": 8, "reads": reads, "max_contaminant_reads": 4}


# --------------------------------------------------------------------------
# tiny HTTP client
# --------------------------------------------------------------------------


def request(method: str, path: str, payload=None, timeout: float = 10.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_BASE_URL + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def wait_healthy() -> bool:
    deadline = time.time() + HEALTH_TIMEOUT_S
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(API_BASE_URL + "/health", timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if resp.status == 200 and body.get("status") == "healthy":
                    print(f"[verify] service healthy at {API_BASE_URL}")
                    return True
        except Exception as exc:  # connection refused while starting etc.
            last_err = exc
        time.sleep(1)
    print(f"[verify] service did not become healthy: {last_err}")
    return False


# --------------------------------------------------------------------------
# assertions
# --------------------------------------------------------------------------


class CheckFailure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def smoke() -> list[str]:
    failures: list[str] = []

    def run(label, fn):
        try:
            fn()
            print(f"[verify] PASS {label}")
        except CheckFailure as exc:
            failures.append(f"{label}: {exc}")
            print(f"[verify] FAIL {label}: {exc}")
        except Exception as exc:  # unexpected transport error etc.
            failures.append(f"{label}: unexpected {type(exc).__name__}: {exc}")
            print(f"[verify] ERROR {label}: {type(exc).__name__}: {exc}")

    def case_mismatch():
        status, body = request("POST", "/api/phase", mismatch_sample())
        check(status == 200, f"status {status}, body {body}")
        check(body["ok"] is True, "envelope ok flag")
        data = body["data"]
        check(data["unique"] is True, "should be uniquely solvable")
        sol = data["solution"]
        check(sol["haplotype"][0] == 0, "haplotype must be canonical (site 0 = 0)")
        check(sol["complement"] == [1 - b for b in sol["haplotype"]], "complement mismatch")
        check(sol["total_mismatch_cost"] == 10, f"total cost {sol['total_mismatch_cost']} != 10")
        check(sol["max_per_read_mismatches"] == 1, "max per-read mismatches != 1")
        check(len(sol["groups"]["haplotype"]) >= 2, "fewer than 2 reads in haplotype group")
        check(len(sol["groups"]["complement"]) >= 2, "fewer than 2 reads in complement group")
        by_id = {r["id"]: r for r in sol["per_read"]}
        check(by_id["a0"]["mismatch_count"] == 1, "a0 mismatch count")
        check(by_id["a0"]["mismatch_cost"] == 7, "a0 mismatch cost")
        check(by_id["a0"]["mismatch_positions"] == [0], "a0 mismatch positions")
        check(by_id["b2"]["mismatch_count"] == 1, "b2 mismatch count")
        check(by_id["b2"]["mismatch_cost"] == 3, "b2 mismatch cost")
        check(by_id["b2"]["mismatch_positions"] == [7], "b2 mismatch positions")
        check(
            sum(r["mismatch_cost"] for r in sol["per_read"]) == sol["total_mismatch_cost"],
            "per-read costs do not reconcile to total",
        )
        check(
            max(r["mismatch_count"] for r in sol["per_read"]) == sol["max_per_read_mismatches"],
            "per-read counts do not reconcile to max",
        )

    def case_ambiguous():
        status, body = request("POST", "/api/phase", ambiguous_sample())
        check(status == 200, f"status {status}, body {body}")
        data = body["data"]
        check(data["unique"] is False, "ambiguous instance must be flagged")
        sols = data["solutions"]
        check(len(sols) == 2, f"expected two distinct solutions, got {len(sols)}")
        s1, s2 = sols
        check(s1["haplotype"] != s2["haplotype"], "solutions must be distinct")
        check(s1["haplotype"] < s2["haplotype"], "solutions must be lexicographically ordered")
        check(s1["total_mismatch_cost"] == s2["total_mismatch_cost"] == 0, "tie on cost")
        check(
            s1["max_per_read_mismatches"] == s2["max_per_read_mismatches"] == 0,
            "tie on max mismatches",
        )
        for s in sols:
            check(len(s["groups"]["haplotype"]) >= 2, "group 0 too small")
            check(len(s["groups"]["complement"]) >= 2, "group 1 too small")

    def case_no_solution():
        status, body = request("POST", "/api/phase", no_solution_sample())
        check(status == 409, f"status {status}")
        check(body["ok"] is False and body["error"]["code"] == "NO_SOLUTION", f"body {body}")

    def case_discontinuous():
        status, body = request("POST", "/api/phase", discontinuous_sample())
        check(status == 409, f"status {status}")
        check(body["error"]["code"] == "DISCONTINUOUS_INPUT", f"body {body}")

    def case_invalid():
        payload = mismatch_sample()
        payload["n_sites"] = 5
        status, body = request("POST", "/api/phase", payload)
        check(status == 422, f"status {status}")
        check(body["error"]["code"] == "INVALID_INPUT", f"body {body}")

    def case_contaminant():
        status, body = request("POST", "/api/phase", contaminant_sample())
        check(status == 200, f"status {status}, body {body}")
        data = body["data"]
        check(data["contaminant_mode"]["enabled"] is True, "contaminant mode flag")
        check(data["contaminant_mode"]["max_contaminant_reads"] == 1, "echoed cap")
        sol = data["solution"]
        check(sol["groups"]["contaminants"] == ["junk"], "outlier must be contaminant")
        check(sol["contaminant_count"] == 1, "one contaminant")
        check(sol["total_mismatch_cost"] == 0, "assigned reads cost zero")
        check(sol["total_contaminant_penalty"] == 4, "penalty total")
        check(sol["objective_mismatch_cost_plus_penalty"] == 4, "objective total")
        check(sol["max_per_read_mismatches"] == 0, "assigned max mm zero")
        check(len(sol["groups"]["haplotype"]) == 5, "five haplotype reads")
        check(len(sol["groups"]["complement"]) == 5, "five complement reads")
        junk = next(r for r in sol["per_read"] if r["id"] == "junk")
        check(junk["group"] == 2, "junk label is 2")
        check(junk["mismatch_count"] is None, "contaminant exempt from mismatch count")
        check(junk["contaminant_penalty"] == 4, "per-read penalty echoed")
        check(bool(junk["contaminant_reason"]), "contaminant reason present")
        details = sol["contaminants"]
        check(len(details) == 1, "one contaminant detail record")
        check(details[0]["mismatches_vs_haplotype"] == 2, "mm vs haplotype evidence")
        check(details[0]["mismatches_vs_complement"] == 3, "mm vs complement evidence")
        check(details[0]["max_mismatches_allowance"] == 0, "allowance evidence")

    def case_contaminant_capacity_short():
        status, body = request("POST", "/api/phase", contaminant_capacity_short_sample())
        check(status == 409, f"status {status}")
        check(
            body["error"]["code"] == "INSUFFICIENT_CONTAMINANT_CAPACITY",
            f"body {body}",
        )

    def case_contaminant_evidence_short():
        status, body = request("POST", "/api/phase", contaminant_evidence_short_sample())
        check(status == 409, f"status {status}")
        check(body["error"]["code"] == "INSUFFICIENT_GROUP_EVIDENCE", f"body {body}")

    def case_contaminant_invalid():
        # cap without per-read penalties
        payload = contaminant_sample()
        for row in payload["reads"]:
            del row["contaminant_penalty"]
        status, body = request("POST", "/api/phase", payload)
        check(status == 422, f"status {status}")
        check(body["error"]["code"] == "INVALID_INPUT", f"body {body}")
        # cap out of range
        payload = contaminant_sample()
        payload["max_contaminant_reads"] = 5
        status, body = request("POST", "/api/phase", payload)
        check(status == 422, f"status {status}")
        check(body["error"]["code"] == "INVALID_INPUT", f"body {body}")

    run("mismatch sample (unique, exact evidence)", case_mismatch)
    run("ambiguous sample (two tied solutions)", case_ambiguous)
    run("contaminant sample (outlier isolated)", case_contaminant)
    run("contaminant capacity shortage", case_contaminant_capacity_short)
    run("contaminant group-evidence shortage", case_contaminant_evidence_short)
    run("contaminant input validation", case_contaminant_invalid)
    run("no-solution business error", case_no_solution)
    run("discontinuous-coverage business error", case_discontinuous)
    run("invalid input rejected", case_invalid)
    return failures


def run_code_tests() -> bool:
    if not os.path.isdir("tests"):
        print("[verify] tests/ not present; skipping code tests")
        return True
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        capture_output=True,
        text=True,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    print(tail)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    return proc.returncode == 0


def main() -> int:
    print(f"[verify] target API: {API_BASE_URL}")
    code = 0

    tests_ok = run_code_tests()
    if not tests_ok:
        code |= FAIL_TESTS
        print("[verify] code tests FAILED")
    else:
        print("[verify] code tests passed")

    if not wait_healthy():
        code |= FAIL_SMOKE
    else:
        failures = smoke()
        if failures:
            code |= FAIL_SMOKE

    print("-" * 60)
    print(f"[verify] result: {'ALL GREEN' if code == 0 else f'FAILURES (exit code {code})'}")
    print("[verify] exit-code map: 1=code tests, 2=API smoke, 4=image build")
    return code


if __name__ == "__main__":
    sys.exit(main())
