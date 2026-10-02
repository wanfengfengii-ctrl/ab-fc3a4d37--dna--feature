"""Unit tests for the phasing solver, including exhaustive cross-checks."""

from __future__ import annotations

import random

import numpy as np
import pytest

from app.solver import (
    ContaminantConfig,
    PhaseError,
    enumerate_solutions,
    parse_input,
    phase,
)


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def make_read(rid, start, end, obs, costs=None, allow=None):
    width = end - start
    return {
        "id": rid,
        "start": start,
        "end": end,
        "observations": list(obs),
        "mismatch_costs": list(costs) if costs is not None else [1] * width,
        "max_mismatches": width if allow is None else allow,
    }


def clean_instance():
    """Two clean complementary haplotypes, 8 sites, 10 reads, no damage."""
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    # group-0 reads
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    # group-1 reads
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read(f"a{i}", s, e, hap[s:e], allow=0))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read(f"b{i}", s, e, comp[s:e], allow=0))
    return 8, reads, hap


def random_instance(n_sites, m, seed):
    """Generate a random covered read set with damaged observations."""
    rng = random.Random(seed)
    hap = [rng.randrange(2) for _ in range(n_sites)]
    comp = [1 - b for b in hap]

    reads = []
    for _attempt in range(200):
        reads = []
        for j in range(m):
            length = rng.randint(2, min(5, n_sites))
            start = rng.randint(0, n_sites - length)
            src = hap if rng.randrange(2) == 0 else comp
            obs = list(src[start : start + length])
            flips = 0
            for k in range(length):
                if rng.random() < 0.15:
                    obs[k] ^= 1
                    flips += 1
            costs = [rng.randint(1, 9) for _ in range(length)]
            # allowance is sometimes tight, sometimes loose; feasibility is
            # left for the solver to decide
            allow = rng.choice([0, flips, flips, min(flips + 1, length), length])
            reads.append(make_read(f"r{j}", start, start + length, obs, costs, allow))
        covered = [False] * n_sites
        for r in reads:
            for s in range(r["start"], r["end"]):
                covered[s] = True
        if all(covered):
            return n_sites, reads
    raise AssertionError("could not generate a covering instance")


def brute_force(n_sites, reads):
    """Enumerate every (canonical haplotype, assignment) pair.

    Returns the sorted list of optimum distinct solutions as
    ``(total, maxmm, hap_tuple, assignment_tuple)``.
    """
    n_sites, pr, _cfg = parse_input({"n_sites": n_sites, "reads": reads})
    n = len(pr)
    best = None
    winners = []
    for hap_bits in range(1 << (n_sites - 1)):
        hap = (0,) + tuple((hap_bits >> (n_sites - 2 - s)) & 1 for s in range(n_sites - 1))
        for ab in range(1 << n):
            assign = tuple((ab >> (n - 1 - i)) & 1 for i in range(n))
            if assign.count(0) < 2 or assign.count(1) < 2:
                continue
            total = 0
            maxmm = 0
            ok = True
            for r, g in zip(pr, assign):
                mm = 0
                for k, site in enumerate(range(r.start, r.end)):
                    mismatch = (r.obs[k] != hap[site]) if g == 0 else (r.obs[k] == hap[site])
                    if mismatch:
                        mm += 1
                        total += r.costs[k]
                if mm > r.max_mismatches:
                    ok = False
                    break
                maxmm = max(maxmm, mm)
            if not ok:
                continue
            key = (total, maxmm)
            rec = (total, maxmm, hap, assign)
            if best is None or key < best:
                best = key
                winners = [rec]
            elif key == best:
                winners.append(rec)
    # Canonical solution order: haplotype first, then assignment.
    winners.sort(key=lambda r: (r[2], r[3]))
    return winners


# --------------------------------------------------------------------------
# basic behaviour
# --------------------------------------------------------------------------


def test_clean_instance_unique_zero_cost():
    n_sites, reads, hap = clean_instance()
    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is True
    sol = out["solution"]
    assert sol["total_mismatch_cost"] == 0
    assert sol["max_per_read_mismatches"] == 0
    assert sol["haplotype"][0] == 0  # canonical
    assert sol["haplotype"] == hap or sol["haplotype"] == [1 - b for b in hap]
    # canonical normalization: the returned haplotype itself must be the
    # oriented one; its first bit is fixed to 0
    assert sol["haplotype"][0] == 0
    assert len(sol["groups"]["haplotype"]) >= 2
    assert len(sol["groups"]["complement"]) >= 2
    assert len(out["solutions"]) == 1


def test_mismatch_evidence_is_consistent():
    n_sites, reads, _ = clean_instance()
    # damage two observations on reads a0 and b2, each with an allowance of 1
    reads[0]["observations"][0] ^= 1
    reads[0]["mismatch_costs"][0] = 7
    reads[0]["max_mismatches"] = 1
    reads[7]["observations"][1] ^= 1
    reads[7]["mismatch_costs"] = [5, 3]
    reads[7]["max_mismatches"] = 1

    out = phase({"n_sites": n_sites, "reads": reads})
    sol = out["solution"]
    assert sol["total_mismatch_cost"] == 10
    assert sol["max_per_read_mismatches"] == 1

    by_id = {row["id"]: row for row in sol["per_read"]}
    assert by_id["a0"]["mismatch_count"] == 1
    assert by_id["a0"]["mismatch_cost"] == 7
    assert by_id["a0"]["mismatch_positions"] == [0]
    assert by_id["b2"]["mismatch_count"] == 1
    assert by_id["b2"]["mismatch_cost"] == 3
    assert by_id["b2"]["mismatch_positions"] == [7]

    # evidence totals reconcile
    assert sum(r["mismatch_cost"] for r in sol["per_read"]) == sol["total_mismatch_cost"]
    assert max(r["mismatch_count"] for r in sol["per_read"]) == sol["max_per_read_mismatches"]
    # assignments reconcile with the group member lists
    for row in sol["per_read"]:
        side = "haplotype" if row["group"] == 0 else "complement"
        assert row["id"] in sol["groups"][side]
    # haplotype and complement are bitwise complements
    assert sol["complement"] == [1 - b for b in sol["haplotype"]]


def test_higher_cost_does_not_buy_lower_max():
    """Secondary objective is optimized only among minimum-cost solutions."""
    n_sites, reads, _ = clean_instance()
    # cheap single damage vs an alternative assignment that would be dearer
    reads[0]["observations"][0] ^= 1
    reads[0]["mismatch_costs"] = [1, 100, 100]
    reads[0]["max_mismatches"] = 2
    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["solution"]["total_mismatch_cost"] == 1


# --------------------------------------------------------------------------
# ambiguity
# --------------------------------------------------------------------------


def test_ambiguous_unlinked_components():
    """Two read blocks with no spanning read => relative phase is ambiguous.

    Block A covers sites 0..3 (haplotype 0011) and block B sites 4..7
    (haplotype 0011).  Each block contains reads from both chromosomes, so
    both groups are populated either way; flipping block B produces a second
    distinct zero-cost canonical haplotype, while flipping block A would put
    a 1 at site 0 and is already quotiented by canonicalization.
    """
    n_sites = 8
    reads = []
    # Block A, sites 0..3, hap 0011 / comp 1100
    a_specs = [
        ("a0", 0, 2, [0, 0]),       # group 0
        ("a1", 1, 4, [0, 1, 1]),    # group 0
        ("a2", 0, 3, [0, 0, 1]),    # group 0
        ("a3", 2, 4, [0, 0]),       # group 1 (comp tail)
        ("a4", 0, 2, [1, 1]),       # group 1 (comp head)
    ]
    # Block B, sites 4..7, hap 0011 / comp 1100
    b_specs = [
        ("b0", 4, 6, [0, 0]),       # group 0
        ("b1", 5, 8, [0, 1, 1]),    # group 0
        ("b2", 4, 7, [0, 0, 1]),    # group 0
        ("b3", 6, 8, [0, 0]),       # group 1
        ("b4", 4, 6, [1, 1]),       # group 1
    ]
    for rid, s, e, obs in a_specs + b_specs:
        reads.append(make_read(rid, s, e, obs, allow=0))

    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is False
    assert len(out["solutions"]) == 2
    s1, s2 = out["solutions"]
    assert s1["total_mismatch_cost"] == s2["total_mismatch_cost"] == 0
    assert s1["max_per_read_mismatches"] == s2["max_per_read_mismatches"] == 0
    assert s1["haplotype"] != s2["haplotype"]
    # first two canonical (lexicographic) solutions
    assert s1["haplotype"] < s2["haplotype"]
    assert len(s1["groups"]["haplotype"]) >= 2
    assert len(s1["groups"]["complement"]) >= 2
    assert len(s2["groups"]["haplotype"]) >= 2
    assert len(s2["groups"]["complement"]) >= 2


# --------------------------------------------------------------------------
# ambiguity from two optimal assignments under one haplotype
# --------------------------------------------------------------------------


def test_ambiguous_two_assignments_under_one_haplotype():
    """A perfectly neutral read can join either group at equal cost.

    The neutral read spans sites 0..1 with observations [0,0]; against the
    true haplotype [0,1] it mismatches once (cost 3), and against the
    complement [1,0] it also mismatches once (cost 3).  Every other read is
    damage-free, so both placements are globally optimal with the same
    haplotype -- two distinct solutions that are NOT swap-equivalent (each
    differs by one read only).
    """
    n_sites, reads, hap = clean_instance()
    # replace read b0 (a clean complement read over [0,2)) with a neutral one
    reads[5] = make_read("b0", 0, 2, [0, 0], costs=[3, 3], allow=1)
    for r in reads:
        if r["id"] != "b0":
            r["max_mismatches"] = 0

    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is False
    s1, s2 = out["solutions"]
    assert s1["haplotype"] == s2["haplotype"] == hap
    assert s1["assignments"] != s2["assignments"]
    # they differ exactly in the neutral read's group
    diffs = [i for i, (a, b) in enumerate(zip(s1["assignments"], s2["assignments"])) if a != b]
    assert diffs == [5]
    assert s1["total_mismatch_cost"] == s2["total_mismatch_cost"] == 3
    assert s1["max_per_read_mismatches"] == s2["max_per_read_mismatches"] == 1
    # first solution assigns the earliest-differing read to group 0
    assert s1["assignments"][5] == 0
    for s in (s1, s2):
        assert len(s["groups"]["haplotype"]) >= 2
        assert len(s["groups"]["complement"]) >= 2


# --------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------


def test_discontinuous_coverage_rejected():
    n_sites = 8
    reads = []
    # ten reads tiling sites 0..5 and 7, leaving site 6 uncovered
    spans = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 3), (1, 4), (2, 5), (7, 8), (0, 4)]
    for j, (s, e) in enumerate(spans):
        obs = [0] * (e - s)
        reads.append(make_read(f"d{j}", s, e, obs, allow=e - s))
    with pytest.raises(PhaseError) as exc:
        parse_input({"n_sites": n_sites, "reads": reads})
    assert exc.value.code == "DISCONTINUOUS_INPUT"
    assert "6" in exc.value.message


def test_no_solution_when_allowances_too_tight():
    # Three reads over the same span with mutually incompatible observations
    # and zero mismatch allowance: no single binary haplotype can match all.
    n_sites = 8
    reads = []
    pattern = [0, 0, 1, 1, 0, 1, 0, 1]
    other = [1, 1, 0, 0, 1, 0, 1, 0]
    third = [0, 1, 0, 1, 0, 1, 0, 1]
    for j, p in enumerate([pattern, other, third]):
        reads.append(make_read(f"x{j}", 0, 8, p, allow=0))
    # pad with duplicate-origin reads (still incompatible everywhere)
    for j in range(7):
        p = pattern if j % 2 == 0 else other
        reads.append(make_read(f"p{j}", 0, 8, p, allow=0))
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": reads})
    assert exc.value.code == "NO_SOLUTION"


def test_invalid_counts_and_shapes():
    n_sites, reads, _ = clean_instance()
    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": 7, "reads": reads})
    assert ei.value.code == "INVALID_INPUT"

    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": n_sites, "reads": reads[:9]})
    assert ei.value.code == "INVALID_INPUT"

    bad = [dict(r) for r in reads]
    bad[0]["observations"] = bad[0]["observations"][:-1]
    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": n_sites, "reads": bad})
    assert ei.value.code == "INVALID_INPUT"

    bad = [dict(r) for r in reads]
    bad[0]["mismatch_costs"] = [1, 0, 1][: len(bad[0]["observations"])]
    with pytest.raises(PhaseError) as ei:
        phase({"n_sites": n_sites, "reads": bad})
    assert ei.value.code == "INVALID_INPUT"


# --------------------------------------------------------------------------
# exhaustive cross-check
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_random(seed):
    n_sites, reads = random_instance(8, 10, seed)
    expected = brute_force(n_sites, reads)
    n_sites_p, parsed, _cfgp = parse_input({"n_sites": n_sites, "reads": reads})
    got = enumerate_solutions(n_sites_p, parsed)

    if not expected:
        assert got == []
        return

    assert len(got) <= 2
    for k, sol in enumerate(got):
        exp = expected[k]
        assert sol.haplotype == exp[2]
        assert sol.assignments == exp[3]
        assert sol.total_cost == exp[0]
        assert sol.max_mismatches == exp[1]

    # ambiguity flag must match whether more than one distinct optimum exists
    out = phase({"n_sites": n_sites, "reads": reads})
    assert out["unique"] is (len(expected) == 1)
    if len(expected) >= 2:
        assert len(out["solutions"]) == 2
    else:
        assert len(out["solutions"]) == 1


# --------------------------------------------------------------------------
# contaminant mode
# --------------------------------------------------------------------------


def make_read_c(rid, start, end, obs, costs=None, allow=None, penalty=None):
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


def contaminant_clean_instance():
    """10 clean reads (5+5) plus one foreign read, 8 sites."""
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    spans0 = [(0, 3), (2, 5), (4, 7), (1, 4), (5, 8)]
    spans1 = [(0, 2), (3, 6), (6, 8), (2, 4), (4, 8)]
    reads = []
    for i, (s, e) in enumerate(spans0):
        reads.append(make_read_c(f"a{i}", s, e, hap[s:e], allow=0, penalty=10))
    for i, (s, e) in enumerate(spans1):
        reads.append(make_read_c(f"b{i}", s, e, comp[s:e], allow=0, penalty=10))
    reads.append(make_read_c("junk", 0, 5, [1, 0, 1, 0, 1], allow=0, penalty=4))
    return 8, reads, hap


def test_contaminant_outlier_labelled_jointly_not_pruned():
    n_sites, reads, hap = contaminant_clean_instance()
    out = phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": 1})
    assert out["unique"] is True
    sol = out["solution"]
    assert sol["haplotype"] == hap
    assert sol["groups"]["haplotype"] == [r["id"] for r in reads if r["id"].startswith("a")]
    assert sol["groups"]["complement"] == [r["id"] for r in reads if r["id"].startswith("b")]
    assert sol["groups"]["contaminants"] == ["junk"]
    assert sol["contaminant_count"] == 1
    assert sol["total_mismatch_cost"] == 0
    assert sol["total_contaminant_penalty"] == 4
    assert sol["objective_mismatch_cost_plus_penalty"] == 4
    assert sol["max_per_read_mismatches"] == 0
    # three-way labels
    assert sol["assignments"] == [0] * 5 + [1] * 5 + [2]
    # contaminant is exempt from the zero allowance: no mismatch evidence
    junk = next(r for r in sol["per_read"] if r["id"] == "junk")
    assert junk["group"] == 2
    assert junk["mismatch_count"] is None
    assert junk["mismatch_cost"] is None
    assert junk["mismatch_positions"] == []
    assert junk["contaminant"] is True
    assert junk["contaminant_penalty"] == 4
    assert "allowance" in junk["contaminant_reason"]
    detail = sol["contaminants"][0]
    assert detail["id"] == "junk"
    assert detail["mismatches_vs_haplotype"] == 2
    assert detail["mismatches_vs_complement"] == 3
    assert detail["max_mismatches_allowance"] == 0
    assert detail["penalty"] == 4
    # non-contaminant rows carry the flag too
    for row in sol["per_read"]:
        if row["id"] != "junk":
            assert row["contaminant"] is False
            assert row["contaminant_penalty"] == 0


def test_contaminant_groups_each_need_two_non_contaminant_reads():
    """The group-0 lower bound forces an otherwise-contaminant read to assign.

    One clean haplotype read and two clean complement reads pin the canonical
    haplotype (their penalties are prohibitive, so none can be contaminated).
    Seven mutually incompatible full-span neutral reads each prefer the
    contaminant state (penalty 1 vs cost 40 on either side), but cap 4 means
    three must be assigned -- and group 0 still needs a second member, so a
    neutral is forced onto the haplotype group at cost 40 rather than paying
    penalty 1.  Nothing is solved first and pruned afterwards.
    """
    hap = [0, 1, 1, 0, 1, 0, 0, 1]
    comp = [1 - b for b in hap]
    reads = [
        make_read_c("h0", 0, 8, hap, allow=0, penalty=1000),
        make_read_c("c1", 0, 4, comp[0:4], allow=0, penalty=1000),
        make_read_c("c2", 4, 8, comp[4:8], allow=0, penalty=1000),
    ]
    patterns = [
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]
    for j in range(7):
        reads.append(
            make_read_c(f"n{j}", 0, 8, patterns[j % 3], costs=[10] * 8, allow=8, penalty=1)
        )
    out = phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": 4})
    sol = out["solution"]
    assert sol["haplotype"] == hap
    assert len(sol["groups"]["haplotype"]) >= 2
    assert len(sol["groups"]["complement"]) >= 2
    assert sol["contaminant_count"] == 4
    assert set(sol["groups"]["complement"]) == {"c1", "c2"}
    assert "h0" in sol["groups"]["haplotype"]
    assert any(r.startswith("n") for r in sol["groups"]["haplotype"])
    # three assigned neutrals cost 40 each; four contaminants pay 1 each
    assert sol["objective_mismatch_cost_plus_penalty"] == 124


def test_contaminant_count_never_exceeds_limit():
    n_sites, reads, _ = contaminant_clean_instance()
    for cap in (1, 2, 3, 4):
        sol = phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": cap})["solution"]
        assert sol["contaminant_count"] <= cap


def test_contaminant_capacity_shortage_is_business_error():
    n_sites, reads, _ = contaminant_clean_instance()
    reads.append(make_read_c("junk2", 3, 8, [0, 1, 0, 1, 0], allow=0, penalty=4))
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": 1})
    assert exc.value.code == "INSUFFICIENT_CONTAMINANT_CAPACITY"
    assert "2" in exc.value.message
    # raising the limit solves it
    out = phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": 2})
    assert set(out["solution"]["groups"]["contaminants"]) == {"junk", "junk2"}
    assert out["solution"]["contaminant_count"] == 2


def test_group_evidence_shortage_is_business_error():
    p1 = [0, 0, 1, 1, 0, 1, 0, 1]
    p3 = [0, 1, 0, 1, 0, 1, 0, 1]
    reads = [make_read_c(f"a{i}", 0, 8, p1, allow=0, penalty=5) for i in range(7)]
    reads += [make_read_c(f"c{i}", 0, 8, p3, allow=0, penalty=5) for i in range(3)]
    for cap in (1, 2, 3, 4):
        with pytest.raises(PhaseError) as exc:
            phase({"n_sites": 8, "reads": reads, "max_contaminant_reads": cap})
        assert exc.value.code == "INSUFFICIENT_GROUP_EVIDENCE"


def test_contaminant_mode_input_gating():
    n_sites, reads, _ = contaminant_clean_instance()
    # cap without per-read penalties
    bare = [{k: v for k, v in r.items() if k != "contaminant_penalty"} for r in reads]
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": bare, "max_contaminant_reads": 1})
    assert exc.value.code == "INVALID_INPUT"
    # penalties without cap
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": reads})
    assert exc.value.code == "INVALID_INPUT"
    # bad cap
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": 0})
    assert exc.value.code == "INVALID_INPUT"
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": 5})
    assert exc.value.code == "INVALID_INPUT"
    # one read missing its penalty once the feature is on
    one_missing = [dict(r) for r in reads]
    del one_missing[0]["contaminant_penalty"]
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": one_missing, "max_contaminant_reads": 1})
    assert exc.value.code == "INVALID_INPUT"
    # non-positive penalty
    bad_pen = [dict(r) for r in reads]
    bad_pen[0]["contaminant_penalty"] = 0
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": bad_pen, "max_contaminant_reads": 1})
    assert exc.value.code == "INVALID_INPUT"


def test_legacy_path_unchanged_when_contaminant_fields_absent():
    """No contaminant fields => identical payload decisions (no extra keys)."""
    n_sites, reads, _ = contaminant_clean_instance()
    bare = [{k: v for k, v in r.items() if k != "contaminant_penalty"} for r in reads]
    # outlier forces legacy NO_SOLUTION (cannot be removed)
    with pytest.raises(PhaseError) as exc:
        phase({"n_sites": n_sites, "reads": bare})
    assert exc.value.code == "NO_SOLUTION"
    # and a plain legacy success exposes none of the contaminant fields
    ok_reads = bare[:-1]
    out = phase({"n_sites": n_sites, "reads": ok_reads})
    sol = out["solution"]
    assert "contaminants" not in sol
    assert "contaminant_count" not in sol
    assert "contaminant_mode" not in out
    for row in sol["per_read"]:
        assert "contaminant" not in row
        assert row["group"] in (0, 1)


def test_contaminant_cheaper_than_forcing():
    """A within-allowance read may still be contaminant if penalty is cheaper."""
    n_sites, reads, hap = contaminant_clean_instance()
    # junk fits the complement at 0 mismatches: making it a genuine
    # complement read is free, so it is never taken as contaminant even if
    # cheap; instead flip the case -- penalty smaller than the minimum forced
    # assignment cost when allowances make the read's cheapest side cost 3.
    reads[-1] = make_read_c(
        "junk", 0, 2, [1, 0], costs=[3, 3], allow=1, penalty=2
    )
    # junk spans [0,2): hap starts [0,1] -> junk [1,0] is the complement,
    # matching it perfectly -> cost 0 in group 1, so it joins group 1.
    out = phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": 1})
    assert "junk" in out["solution"]["groups"]["complement"]
    assert out["solution"]["contaminant_count"] == 0
    # now make its observations damage one site each side so cheapest cost is
    # positive and penalty 1 undercuts it
    reads[-1] = make_read_c(
        "junk", 0, 3, [0, 0, 0], costs=[5, 5, 5], allow=1, penalty=1
    )
    out = phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": 1})
    # hap[0:3] = [0,1,1]: as haplotype mismatches sites 1,2 (allowance 1 ->
    # infeasible); as complement [1,0,0] mismatches site 0, cost 5. Penalty
    # 1 is cheaper -> contaminant
    assert out["solution"]["groups"]["contaminants"] == ["junk"]
    assert out["solution"]["objective_mismatch_cost_plus_penalty"] == 1


# --------------------------------------------------------------------------
# contaminant exhaustive cross-check (numpy-vectorized three-way brute force)
# --------------------------------------------------------------------------


def _contaminant_brute(n_sites, parsed_reads, cap):
    """Vectorized enumeration of every (haplotype, 3^n labelling) outcome.

    Returns ``(best_key, records)`` with ``records`` sorted by
    ``(haplotype, labels)``; ``best_key`` is None when infeasible.
    """
    n = len(parsed_reads)
    c_count = 1 << (n_sites - 1)
    haps = np.zeros((c_count, n_sites), dtype=np.int8)
    idx = np.arange(c_count)
    haps[:, 0] = 0
    for s in range(1, n_sites):
        haps[:, s] = (idx >> (n_sites - 1 - s)) & 1

    best_key = None
    records: list[tuple[int, int, tuple, tuple, int]] = []
    for hb in range(c_count):
        hap = tuple(int(x) for x in haps[hb])
        # all 3**n three-way labels, most significant read first
        labels_arr = np.zeros((3 ** n, n), dtype=np.int8)
        codes = np.arange(3 ** n)
        x = codes
        for i in range(n):
            labels_arr[:, i] = x % 3
            x //= 3
        labels_arr = labels_arr[:, ::-1]
        g0c = (labels_arr == 0).sum(axis=1)
        g1c = (labels_arr == 1).sum(axis=1)
        qc = (labels_arr == 2).sum(axis=1)
        feasible_lab = (g0c >= 2) & (g1c >= 2) & (qc <= cap)

        mismatch_cost = np.zeros(3 ** n, dtype=np.int64)
        penalty_cost = np.zeros(3 ** n, dtype=np.int64)
        maxmm = np.zeros(3 ** n, dtype=np.int64)
        allowed = np.ones(3 ** n, dtype=bool)
        for i, r in enumerate(parsed_reads):
            h = haps[hb, r.start : r.end].astype(np.int64)
            obs = np.asarray(r.obs, dtype=np.int64)
            costs = np.asarray(r.costs, dtype=np.int64)
            mis0 = (obs != h)
            mis1 = ~mis0
            mm0v = int(mis0.sum())
            mm1v = int(mis1.sum())
            c0v = int(costs[mis0].sum())
            c1v = int(costs[mis1].sum())
            lab = labels_arr[:, i]
            edge_cost = np.where(lab == 0, c0v, np.where(lab == 1, c1v, 0))
            edge_mm = np.where(lab == 0, mm0v, np.where(lab == 1, mm1v, 0))
            edge_pen = np.where(lab == 2, r.contaminant_penalty, 0)
            mismatch_cost += edge_cost
            penalty_cost += edge_pen
            maxmm = np.maximum(maxmm, edge_mm)
            edge_allowed = (lab != 0) | (mm0v <= r.max_mismatches)
            edge_allowed &= (lab != 1) | (mm1v <= r.max_mismatches)
            allowed &= edge_allowed
        ok = feasible_lab & allowed
        if not ok.any():
            continue
        scores = mismatch_cost + penalty_cost
        sub = np.where(ok)[0]
        minscore = int(scores[sub].min())
        tied = sub[scores[sub] == minscore]
        mink = int(maxmm[tied].min())
        key = (minscore, mink)
        if best_key is None or key < best_key:
            best_key = key
            records = []
        if key == best_key:
            for code in tied:
                if int(maxmm[code]) == mink:
                    labels = tuple(int(v) for v in labels_arr[code])
                    records.append((key[0], key[1], hap, labels, code))
    records.sort(key=lambda t: (t[2], t[3]))
    return best_key, records


def contaminant_random_instance(seed):
    rng = random.Random(seed)
    hap = [rng.randrange(2) for _ in range(8)]
    comp = [1 - b for b in hap]
    while True:
        reads = []
        for j in range(10):
            length = rng.randint(2, 5)
            start = rng.randint(0, 8 - length)
            src = hap if rng.randrange(2) == 0 else comp
            obs = list(src[start : start + length])
            for k in range(length):
                if rng.random() < 0.18:
                    obs[k] ^= 1
            costs = [rng.randint(1, 5) for _ in range(length)]
            allow = rng.choice([0, 0, 1, 2, length])
            pen = rng.choice([1, 3, 50])
            reads.append(make_read_c(f"r{j}", start, start + length, obs, costs, allow, pen))
        covered = [False] * 8
        for r in reads:
            for s in range(r["start"], r["end"]):
                covered[s] = True
        if all(covered):
            cap = rng.choice([1, 2, 3])
            return 8, reads, cap


@pytest.mark.parametrize("seed", range(24))
def test_contaminant_matches_brute_force_random(seed):
    n_sites, reads, cap = contaminant_random_instance(seed)
    _, parsed, cfg = parse_input(
        {"n_sites": n_sites, "reads": reads, "max_contaminant_reads": cap}
    )
    best, expected = _contaminant_brute(n_sites, parsed, cap)
    got = enumerate_solutions(n_sites, parsed, ContaminantConfig(cap))
    if best is None:
        assert got == []
        return
    expected = expected[:2]
    assert len(got) == len(expected)
    for sol, e in zip(got, expected):
        assert sol.haplotype == e[2]
        assert sol.assignments == e[3]
        assert sol.total_cost + sol.total_penalty == e[0]
        assert sol.max_mismatches == e[1]
    # unique flag through the public API
    out = phase({"n_sites": n_sites, "reads": reads, "max_contaminant_reads": cap})
    assert out["unique"] is (len(expected) == 1)
