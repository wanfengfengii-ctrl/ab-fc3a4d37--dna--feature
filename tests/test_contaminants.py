"""Solver tests for joint contaminant handling (POST /api/phase extension).

When ``max_contaminant_reads`` (1..4) plus a positive ``contaminant_penalty``
per read is supplied, every read jointly chooses the haplotype group, the
complement group or a *contaminant* state.  Contaminant reads bypass the
mismatch allowance, are capped in number, and each group must still hold at
least two non-contaminant reads.  These tests pin down:

* exact optimality via exhaustive brute force (three-way assignments),
* the contaminant cap and the two-valid-reads-per-group rule,
* the two business errors (capacity / group evidence),
* request validation and byte-for-byte legacy compatibility.
"""

from __future__ import annotations

import itertools
import random

import pytest

from app.solver import PhaseError, parse_input_full, phase


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


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


def clean_reads(penalty=None):
    """Two clean complementary haplotypes, 8 sites, 10 reads, no damage."""
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0, penalty=penalty))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0, penalty=penalty))
    return reads, hap


def strip_penalties(reads):
    return [{k: v for k, v in r.items() if k != "contaminant_penalty"} for r in reads]


def brute_force(n_sites, reads, q):
    """Enumerate every (canonical haplotype, ternary assignment) pair."""
    _, pr, _ = parse_input_full(
        {"n_sites": n_sites, "reads": reads, "max_contaminant_reads": q}
    )
    n = len(pr)
    best = None
    winners = []
    for hap_bits in range(1 << (n_sites - 1)):
        hap = (0,) + tuple(
            (hap_bits >> (n_sites - 2 - s)) & 1 for s in range(n_sites - 1)
        )
        for digits in itertools.product(range(3), repeat=n):
            c0 = digits.count(0)
            c1 = digits.count(1)
            ct = digits.count(2)
            if ct > q or c0 < 2 or c1 < 2:
                continue
            total = 0
            maxmm = 0
            ok = True
            for r, g in zip(pr, digits):
                mm = 0
                spent = 0
                for k, site in enumerate(range(r.start, r.end)):
                    mismatch = (
                        (r.obs[k] != hap[site]) if g == 0 else (r.obs[k] == hap[site])
                    )
                    if mismatch:
                        mm += 1
                        spent += r.costs[k]
                if g == 2:
                    total += r.contaminant_penalty
                    continue
                if mm > r.max_mismatches:
                    ok = False
                    break
                total += spent
                maxmm = max(maxmm, mm)
            if not ok:
                continue
            key = (total, maxmm)
            rec = (total, maxmm, hap, digits)
            if best is None or key < best:
                best = key
                winners = [rec]
            elif key == best:
                winners.append(rec)
    winners.sort(key=lambda r: (r[2], r[3]))
    return winners


def random_instance(n_sites, m, seed):
    rng = random.Random(seed)
    hap = [rng.randrange(2) for _ in range(n_sites)]
    comp = [1 - b for b in hap]
    for _attempt in range(200):
        reads = []
        for j in range(m):
            length = rng.randint(2, min(5, n_sites))
            start = rng.randint(0, n_sites - length)
            src = hap if rng.randrange(2) == 0 else comp
            obs = list(src[start : start + length])
            flips = 0
            for k in range(length):
                if rng.random() < 0.18:
                    obs[k] ^= 1
                    flips += 1
            costs = [rng.randint(1, 9) for _ in range(length)]
            allow = rng.choice([0, flips, flips, min(flips + 1, length), length])
            penalty = rng.choice([1, 2, 5, 20, 100])
            reads.append(make_read(f"r{j}", start, start + length, obs, costs, allow, penalty))
        covered = [False] * n_sites
        for r in reads:
            for s in range(r["start"], r["end"]):
                covered[s] = True
        if all(covered):
            return n_sites, reads
    raise AssertionError("could not generate a covering instance")


# --------------------------------------------------------------------------
# core behaviour
# --------------------------------------------------------------------------


def test_garbage_read_is_flagged_contaminant_and_bypasses_allowance():
    reads, hap = clean_reads(penalty=100)
    # all-zero garbage span is far from both the haplotype and the
    # complement; with allowance 0 it cannot join either homologous group.
    reads.append(make_read("c0", 0, 4, [1, 0, 1, 0], costs=[9] * 4, allow=0, penalty=5))
    reads.append(make_read("a5", 0, 4, hap[0:4], allow=0, penalty=100))

    out = phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 2})
    sol = out["solution"]
    assert out["unique"] is True
    assert sol["haplotype"] == hap
    assert sol["contaminant_reads"] == ["c0"]
    assert sol["contaminant_count"] == 1
    assert sol["total_mismatch_cost"] == 0
    assert sol["total_contaminant_penalty"] == 5
    assert sol["total_objective_cost"] == 5
    assert sol["max_per_read_mismatches"] == 0
    # both groups retain their valid evidence
    assert len(sol["groups"]["haplotype"]) >= 2
    assert len(sol["groups"]["complement"]) >= 2
    # ternary assignment labels the contaminant read with 2
    by_id = {r["id"]: i for i, r in enumerate(reads)}
    assert sol["assignments"][by_id["c0"]] == 2

    row = next(r for r in sol["per_read"] if r["id"] == "c0")
    assert row["contaminant"] is True
    assert row["status"] == "contaminant"
    assert row["group"] is None
    assert row["contaminant_penalty"] == 5
    assert row["contaminant_reason"]["feasible_groups"] == []
    assert row["contaminant_reason"]["mismatches_vs_haplotype"]["count"] == 2
    assert row["contaminant_reason"]["mismatches_vs_complement"]["count"] == 2

    # the same request *without* the feature is not solved-then-pruned: it
    # is a plain legacy NO_SOLUTION
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": 8, "reads": strip_penalties(reads)})
    assert exc.value.code == "NO_SOLUTION"


def test_penalty_cheaper_than_compliant_assignment_picks_contaminant():
    reads, hap = clean_reads(penalty=100)
    # b0 becomes a neutral read: cost 3 on either side, allowance 1,
    # penalty 1 -> contaminant is strictly cheaper in the joint objective.
    reads[5] = make_read("b0", 0, 2, [0, 0], costs=[3, 3], allow=1, penalty=1)

    out = phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 1})
    sol = out["solution"]
    assert sol["haplotype"] == hap
    assert sol["contaminant_reads"] == ["b0"]
    assert sol["total_mismatch_cost"] == 0
    assert sol["total_contaminant_penalty"] == 1
    assert sol["total_objective_cost"] == 1
    row = next(r for r in sol["per_read"] if r["id"] == "b0")
    assert row["contaminant"] is True
    reason = row["contaminant_reason"]
    assert set(reason["feasible_groups"]) == {"haplotype", "complement"}
    # contaminating b0 drops the optimal non-contaminant cap to 0, so its
    # allowance-compliant (one-mismatch) options all exceed that cap
    assert reason["optimal_max_mismatches"] == 0
    assert reason["cheapest_within_cap_option"] is None
    assert reason["cheapest_allowance_compliant_option"]["mismatch_cost"] == 3
    assert "cap" in reason["explanation"]

    # without the feature it is force-assigned at cost 3 (old behaviour)
    legacy = phase({"n_sites": 8, "reads": strip_penalties(reads)})
    assert legacy["solution"]["total_mismatch_cost"] == 3


def test_cap_is_enforced():
    reads, hap = clean_reads(penalty=100)
    reads.append(make_read("x0", 0, 3, [1, 0, 1], costs=[9] * 3, allow=0, penalty=1))
    reads.append(make_read("x1", 5, 8, [0, 1, 0], costs=[9] * 3, allow=0, penalty=1))

    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 1})
    assert exc.value.code == "INSUFFICIENT_CONTAMINANT_CAPACITY"

    out = phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 2})
    assert set(out["solution"]["contaminant_reads"]) == {"x0", "x1"}


def test_group_evidence_rule():
    # ten identical all-zero reads with allowance 0: one homologous group
    # can never collect two non-contaminant reads, even though the
    # contaminant cap could absorb four reads.
    reads = [make_read(f"r{j}", 0, 8, [0] * 8, allow=0, penalty=1) for j in range(10)]
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 4})
    assert exc.value.code == "INSUFFICIENT_GROUP_EVIDENCE"


def test_contaminants_do_not_count_toward_group_sizes():
    # Eight reads of pattern P and two of a near-P pattern R (P with one
    # site flipped), all at zero allowance.  At canonical haplotype P the
    # two R reads are forced contaminants (they match neither side), but
    # the eight remaining P reads support only the haplotype side, leaving
    # the complement group with zero non-contaminant evidence.  At
    # haplotype R all eight P reads become forced contaminants.  No cap up
    # to 4 can conjure the missing group.
    p = [0, 0, 1, 1, 0, 1, 0, 1]
    rpat = [0, 0, 1, 0, 0, 1, 0, 1]
    reads = [make_read(f"p{j}", 0, 8, p, allow=0, penalty=100) for j in range(8)]
    reads += [make_read(f"r{j}", 0, 8, rpat, allow=0, penalty=1) for j in range(2)]
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 2})
    assert exc.value.code == "INSUFFICIENT_GROUP_EVIDENCE"
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 4})
    assert exc.value.code == "INSUFFICIENT_GROUP_EVIDENCE"


def test_contaminant_ambiguity_keeps_two_lexicographic_solutions():
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
    reads = [make_read(rid, s, e, obs, allow=0, penalty=10) for rid, s, e, obs in a_specs + b_specs]
    out = phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 2})
    assert out["unique"] is False
    s1, s2 = out["solutions"]
    assert s1["haplotype"] < s2["haplotype"]
    assert s1["total_objective_cost"] == s2["total_objective_cost"] == 0
    assert s1["contaminant_count"] == s2["contaminant_count"] == 0


def test_three_way_tie_digit_order():
    """When contaminant and a group tie, digit 0 precedes digit 2.

    b0 can join the haplotype group within the optimal cap at mismatch cost
    3 (the complement side needs 2 mismatches, infeasible under allowance
    1); a separate damaged read already fixes the secondary objective at 1
    mismatch.  With b0's contaminant penalty equal to 3, the group-assigned
    and contaminant explanations tie on both objectives, so canonical
    three-way order (0 < 2) makes them the first two distinct solutions.
    """
    reads, hap = clean_reads(penalty=100)
    # damaged group-0 read spanning sites 1..3 (site-0 evidence untouched so
    # the canonical haplotype stays the true one): allowance 1, cost 1
    reads[3] = make_read("a3", 1, 4, [1, 1, 1], costs=[1, 1, 1], allow=1, penalty=100)
    # b0: within cap only on the haplotype side (1 mismatch, cost 3);
    # contaminant penalty 3 ties with that cost
    reads[5] = make_read("b0", 0, 3, [1, 1, 1], costs=[3, 1, 1], allow=1, penalty=3)

    out = phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 1})
    assert out["unique"] is False
    s1, s2 = out["solutions"]
    assert s1["haplotype"] == s2["haplotype"] == hap
    assert s1["assignments"][5] == 0
    assert s2["assignments"][5] == 2
    assert s1["contaminant_reads"] == []
    assert s2["contaminant_reads"] == ["b0"]
    for s in (s1, s2):
        assert s["total_objective_cost"] == 4
        assert s["max_per_read_mismatches"] == 1


# --------------------------------------------------------------------------
# validation / legacy compatibility
# --------------------------------------------------------------------------


def test_cap_validation():
    reads, _ = clean_reads(penalty=1)

    for bad in (0, 5, -1, True, "2", 1.0):
        with pytest.raises(PhaseError) as exc:
            phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": bad})
        assert exc.value.code == "INVALID_INPUT", bad

    # 1..4 are the accepted range
    for good in (1, 2, 3, 4):
        out = phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": good})
        assert out["max_contaminant_reads"] == good

    # penalty fields without any cap are rejected (ambiguous feature state)
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": 8, "reads": reads})
    assert exc.value.code == "INVALID_INPUT"


def test_penalty_validation():
    reads, _ = clean_reads(penalty=1)

    # cap enabled but a penalty missing / non-positive / wrong type
    for mutate in (
        lambda rs: rs[0].pop("contaminant_penalty"),
        lambda rs: rs[0].__setitem__("contaminant_penalty", 0),
        lambda rs: rs[0].__setitem__("contaminant_penalty", -3),
        lambda rs: rs[0].__setitem__("contaminant_penalty", "5"),
        lambda rs: rs[0].__setitem__("contaminant_penalty", True),
    ):
        import copy

        rs = copy.deepcopy(reads)
        mutate(rs)
        with pytest.raises(PhaseError) as exc:
            phase({"n_sites": 8, "reads": rs, "max_contaminant_reads": 2})
        assert exc.value.code == "INVALID_INPUT"

    # penalty present without a cap is rejected (feature state unambiguous)
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": 8, "reads": reads})
    assert exc.value.code == "INVALID_INPUT"


def test_legacy_response_shape_unchanged():
    reads, _ = clean_reads()
    out = phase({"n_sites": 8, "reads": reads})
    sol = out["solution"]
    assert set(sol) == {
        "haplotype",
        "complement",
        "groups",
        "assignments",
        "per_read",
        "total_mismatch_cost",
        "max_per_read_mismatches",
        "mismatch_positions",
    }
    assert set(sol["groups"]) == {"haplotype", "complement"}
    assert set(sol["per_read"][0]) == {
        "id",
        "group",
        "mismatch_count",
        "mismatch_cost",
        "mismatch_positions",
    }
    assert "max_contaminant_reads" not in out
    assert all(a in (0, 1) for a in sol["assignments"])


# --------------------------------------------------------------------------
# exhaustive cross-check
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_matches_brute_force_random(seed):
    q = (seed % 4) + 1
    n_sites, reads = random_instance(8, 10, seed)
    expected = brute_force(n_sites, reads, q)
    payload = {"n_sites": n_sites, "reads": reads, "max_contaminant_reads": q}

    if not expected:
        with pytest.raises(PhaseError) as exc:
            phase(payload)
        assert exc.value.code in (
            "INSUFFICIENT_CONTAMINANT_CAPACITY",
            "INSUFFICIENT_GROUP_EVIDENCE",
        )
        return

    out = phase(payload)
    assert out["unique"] is (len(expected) == 1)
    assert len(out["solutions"]) == (1 if len(expected) == 1 else 2)
    for k, sol in enumerate(out["solutions"]):
        exp = expected[k]
        assert sol["haplotype"] == list(exp[2])
        assert tuple(sol["assignments"]) == exp[3]
        assert sol["total_objective_cost"] == exp[0]
        assert sol["max_per_read_mismatches"] == exp[1]
        assert sol["total_mismatch_cost"] + sol["total_contaminant_penalty"] == exp[0]
        assert sol["contaminant_count"] == exp[3].count(2)
        assert sol["contaminant_count"] <= q


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force_tiny_chunks(monkeypatch, seed):
    """Force the vectorized DP down the multi-chunk indexing path."""
    import app.solver as solver_mod

    original = solver_mod.contaminant_dp_costs

    def forced_chunk(*args, **kwargs):
        kwargs["chunk"] = 13
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_mod, "contaminant_dp_costs", forced_chunk)

    q = (seed % 4) + 1
    n_sites, reads = random_instance(8, 10, 1000 + seed)
    expected = brute_force(n_sites, reads, q)
    payload = {"n_sites": n_sites, "reads": reads, "max_contaminant_reads": q}
    if not expected:
        with pytest.raises(PhaseError):
            phase(payload)
        return
    out = phase(payload)
    for k, sol in enumerate(out["solutions"]):
        exp = expected[k]
        assert sol["haplotype"] == list(exp[2])
        assert tuple(sol["assignments"]) == exp[3]
        assert sol["total_objective_cost"] == exp[0]
        assert sol["max_per_read_mismatches"] == exp[1]
