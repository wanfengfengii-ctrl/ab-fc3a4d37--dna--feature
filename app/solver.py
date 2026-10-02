"""Complementary-haplotype phasing core.

Jointly recovers a pair of complementary binary haplotypes from degraded
molecular reads.  For every canonical haplotype candidate (first site fixed
to ``0``, which removes the group-swap symmetry) each read is assigned to
exactly one of the two groups (haplotype / complement) -- or, when
contaminant handling is enabled, to a third *contaminant* state -- so that:

1. assigned reads' mismatch counts do not exceed their allowances and the
   objective (assigned mismatch cost + contaminant penalties) is minimized,
2. each group holds at least two *non-contaminant* reads,
3. the largest per-assigned-read mismatch count is minimized, then
4. the assignment (and, on full ties, the haplotype) is lexicographically
   smallest -- giving a stable decision between uniqueness and ambiguity.

Contaminant reads are exempt from the mismatch allowance and never forced
onto either homologue; instead they pay a per-read positive penalty, and at
most ``max_contaminant_reads`` (1..4) may be labelled contaminant.  The
choice is joint: every read simultaneously chooses group 0, group 1 or
contaminant, so outliers are never solved for first and pruned afterwards.

Legacy mode (neither ``max_contaminant_reads`` nor any
``contaminant_penalty`` is supplied) keeps the original request, response,
decision and error behaviour.

Algorithm (n_sites <= 18, reads <= 36):

* all 2**(n_sites-1) canonical candidates are tabulated with vectorized
  numpy (per-read mismatch counts and costs against each side);
* legacy mode: an O(reads) greedy analysis gives the exact minimum cost per
  candidate (group bounds >= 2 never require flipping more than two reads
  to their dearer side, equal-cost neutral reads fill deficits for free);
* contaminant mode: a vectorized DP whose state is the capped count of
  reads in each group plus the number of contaminants gives the exact
  minimum objective per candidate;
* among candidates attaining the global minimum, a vectorized DP (same
  state) is run with a rising cap K on the per-assigned-read mismatch
  count -- contaminant edges carry no mismatch constraint.  The first K at
  which the minimum objective is reachable with 2..n-2 assigned reads per
  group is optimal; the first two distinct candidates feasible there are
  the reported tie set;
* a final exact (Python) DP for those at most two candidates yields the
  lexicographically smallest three-way labelling and its mismatch /
  penalty evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----- input validation -----------------------------------------------------


class PhaseError(ValueError):
    """Business-level validation / infeasibility error.

    ``code`` is a stable machine-readable reason returned by the API.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Read:
    id: str
    start: int  # inclusive, 0-based
    end: int  # exclusive
    obs: tuple[int, ...]
    costs: tuple[int, ...]
    max_mismatches: int
    contaminant_penalty: int | None = None  # None => read cannot be contaminant


@dataclass(frozen=True)
class ContaminantConfig:
    max_reads: int  # 1..4


@dataclass(frozen=True)
class Solution:
    haplotype: tuple[int, ...]
    assignments: tuple[int, ...]  # 0/1 (legacy) or 0/1/2 contaminant
    total_cost: int
    max_mismatches: int
    mismatch_counts: tuple[int, ...]
    mismatch_costs: tuple[int, ...]
    mismatch_positions: tuple[tuple[int, ...], ...]  # global site indices
    contaminant_penalties: tuple[int, ...] = ()  # per read, 0 unless contaminant
    total_penalty: int = 0


def _as_int_list(value, what: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PhaseError("INVALID_INPUT", f"{what} must be a non-empty list")
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise PhaseError("INVALID_INPUT", f"{what}[{i}] must be an integer")
        out.append(item)
    return out


def parse_input(payload: object) -> tuple[int, list[Read], ContaminantConfig | None]:
    if not isinstance(payload, dict):
        raise PhaseError("INVALID_INPUT", "request body must be a JSON object")

    n_sites = payload.get("n_sites")
    if isinstance(n_sites, bool) or not isinstance(n_sites, int):
        raise PhaseError("INVALID_INPUT", "n_sites must be an integer")
    if not 8 <= n_sites <= 18:
        raise PhaseError("INVALID_INPUT", "n_sites must be between 8 and 18")

    # Contaminant handling is opt-in.  It is enabled exactly when at least one
    # of the two contaminant fields is present; once enabled the count limit
    # must be an integer in 1..4 and every read must carry a positive
    # contaminant_penalty (its option to be labelled contaminant).
    has_cap = "max_contaminant_reads" in payload and payload["max_contaminant_reads"] is not None
    cfg: ContaminantConfig | None = None
    if has_cap:
        cap = payload["max_contaminant_reads"]
        if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 4:
            raise PhaseError(
                "INVALID_INPUT", "max_contaminant_reads must be an integer between 1 and 4"
            )
        cfg = ContaminantConfig(max_reads=cap)

    raw_reads = payload.get("reads")
    if not isinstance(raw_reads, list):
        raise PhaseError("INVALID_INPUT", "reads must be a list")
    if not 10 <= len(raw_reads) <= 36:
        raise PhaseError("INVALID_INPUT", "reads must contain between 10 and 36 items")

    reads: list[Read] = []
    seen_ids: set[str] = set()
    penalty_seen = False
    for idx, item in enumerate(raw_reads):
        if not isinstance(item, dict):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}] must be an object")
        rid = item.get("id", f"r{idx}")
        if not isinstance(rid, str) or not rid:
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].id must be a non-empty string")
        if rid in seen_ids:
            raise PhaseError("INVALID_INPUT", f"duplicate read id: {rid}")
        seen_ids.add(rid)

        start = item.get("start")
        end = item.get("end")
        if isinstance(start, bool) or not isinstance(start, int):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].end must be an integer")
        if not 0 <= start < end <= n_sites:
            raise PhaseError(
                "DISCONTINUOUS_INPUT",
                f"reads[{idx}] span [{start},{end}) is empty or outside [0,{n_sites})",
            )

        obs = _as_int_list(item.get("observations"), f"reads[{idx}].observations")
        costs = _as_int_list(item.get("mismatch_costs"), f"reads[{idx}].mismatch_costs")
        width = end - start
        if len(obs) != width or len(costs) != width:
            raise PhaseError(
                "INVALID_INPUT",
                f"reads[{idx}] observations/mismatch_costs length must equal end-start",
            )
        if any(b not in (0, 1) for b in obs):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}] observations must be binary 0/1")
        if any(c <= 0 for c in costs):
            raise PhaseError("INVALID_INPUT", f"reads[{idx}] mismatch costs must be positive integers")

        allow = item.get("max_mismatches")
        if isinstance(allow, bool) or not isinstance(allow, int) or allow < 0:
            raise PhaseError(
                "INVALID_INPUT", f"reads[{idx}].max_mismatches must be a non-negative integer"
            )
        if allow > width:
            raise PhaseError("INVALID_INPUT", f"reads[{idx}].max_mismatches exceeds its span")

        # Per-read contaminant penalty; the option to label this read a
        # contaminant exists only when it carries a positive penalty.
        penalty: int | None = None
        if "contaminant_penalty" in item and item["contaminant_penalty"] is not None:
            penalty_seen = True
            pv = item["contaminant_penalty"]
            if isinstance(pv, bool) or not isinstance(pv, int) or pv <= 0:
                raise PhaseError(
                    "INVALID_INPUT",
                    f"reads[{idx}].contaminant_penalty must be a positive integer",
                )
            penalty = pv

        reads.append(
            Read(
                id=rid,
                start=start,
                end=end,
                obs=tuple(obs),
                costs=tuple(costs),
                max_mismatches=allow,
                contaminant_penalty=penalty,
            )
        )

    # Feature gating: it activates iff at least one of the two fields is
    # present; then both must be supplied completely.
    if cfg is not None and not penalty_seen:
        raise PhaseError(
            "INVALID_INPUT",
            "every read must carry a positive contaminant_penalty when "
            "max_contaminant_reads is set",
        )
    if penalty_seen and cfg is None:
        raise PhaseError(
            "INVALID_INPUT",
            "max_contaminant_reads (an integer from 1 to 4) is required when "
            "contaminant_penalty is supplied",
        )
    if cfg is not None:
        missing = [r.id for r in reads if r.contaminant_penalty is None]
        if missing:
            raise PhaseError(
                "INVALID_INPUT",
                "every read must carry a positive contaminant_penalty; "
                f"missing for: {', '.join(missing[:5])}",
            )

    # The optimizer tabulates costs in signed 64-bit integers; reject inputs
    # whose theoretical maximum total score could interfere with the DP's
    # infinity sentinel (inf ~ 2**61, so path sums must stay far below it).
    grand_total = sum(sum(r.costs) for r in reads)
    if cfg is not None:
        grand_total += sum(r.contaminant_penalty for r in reads)
    if grand_total > (1 << 58):
        raise PhaseError(
            "INVALID_INPUT",
            "sum of mismatch costs and contaminant penalties is too large to "
            "score exactly; values must be small positive integers "
            "(aggregate score must fit in 59 bits)",
        )

    # Every read is a contiguous interval by construction.  Across reads the
    # union must tile the full locus: an uncovered site cannot be phased and
    # is a business-level "discontinuous input" rejection.
    covered = [False] * n_sites
    for r in reads:
        for s in range(r.start, r.end):
            covered[s] = True
    if not all(covered):
        gap = next(i for i, c in enumerate(covered) if not c)
        raise PhaseError(
            "DISCONTINUOUS_INPUT",
            f"site {gap} is not covered by any read; reads do not form a continuous tiling",
        )

    return n_sites, reads, cfg


# ----- candidate tables -----------------------------------------------------


def candidate_tables(n_sites: int, reads: list[Read]):
    """Per-candidate mismatch / cost / feasibility tables.

    Arrays shaped ``(C, m)`` with ``C = 2**(n_sites-1)``; candidates are in
    lexicographic order of the free sites (candidate 0 = all-zero
    haplotype; site 0 is fixed to 0):

    * ``mm0``   uint8 mismatches vs the canonical haplotype (group 0)
    * ``mm1``   uint8 mismatches vs the complement (group 1)
    * ``cost0`` int64 mismatch cost vs the canonical haplotype
    * ``cost1`` int64 mismatch cost vs the complement
    * ``feas0`` bool  group-0 assignment respects the mismatch allowance
    * ``feas1`` bool  group-1 assignment respects the mismatch allowance
    """
    m = len(reads)
    n_bits = n_sites - 1
    c_count = 1 << n_bits

    mm0 = np.zeros((c_count, m), dtype=np.uint8)
    cost0 = np.zeros((c_count, m), dtype=np.int64)

    idx = np.arange(c_count, dtype=np.int64)
    site_cols: list[np.ndarray] = [np.zeros(c_count, dtype=np.uint8)]
    for s in range(1, n_sites):
        site_cols.append(((idx >> (n_bits - s)) & 1).astype(np.uint8))

    for j, r in enumerate(reads):
        span = r.end - r.start
        obs = np.asarray(r.obs, dtype=np.uint8)
        costs = np.asarray(r.costs, dtype=np.int64)
        mis = np.empty((span, c_count), dtype=np.uint8)
        for k, site in enumerate(range(r.start, r.end)):
            np.bitwise_xor(obs[k], site_cols[site], out=mis[k])
        mm0[:, j] = mis.sum(axis=0)
        cost0[:, j] = costs @ mis

    allow = np.asarray([r.max_mismatches for r in reads], dtype=np.uint8)
    spans = np.asarray([r.end - r.start for r in reads], dtype=np.uint8)
    span_cost = np.asarray([sum(r.costs) for r in reads], dtype=np.int64)
    mm1 = spans[None, :].astype(np.uint8) - mm0
    cost1 = span_cost[None, :] - cost0
    feas0 = mm0 <= allow[None, :]
    feas1 = mm1 <= allow[None, :]
    return mm0, mm1, cost0, cost1, feas0, feas1


# ----- minimum-cost greedy analysis ----------------------------------------


def min_cost_for_rows(
    m0: np.ndarray,
    m1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    offset: int,
) -> dict[int, int]:
    """Exact minimum achievable cost for each row of a candidate chunk.

    With every read sent to its cheapest feasible side, each group already
    holds some count of strict-preference and forced reads.  Satisfying the
    "at least two reads per group" bounds can require flipping at most two
    strict reads per side to their dearer side (equal-cost *neutral* reads
    fill deficits free of charge), so only the two cheapest flip penalties
    per side ever matter.
    """
    out: dict[int, int] = {}
    n = m0.shape[1]
    for row in range(m0.shape[0]):
        forced_count0 = 0
        forced_count1 = 0
        pen1to0: list[int] = []  # strict group-1 pref: penalty to flip to 0
        pen0to1: list[int] = []  # strict group-0 pref: penalty to flip to 1
        neutral = 0
        base = 0
        dead = False
        for i in range(n):
            a = bool(f0[row, i])
            b = bool(f1[row, i])
            if not a and not b:
                dead = True
                break
            cc0 = int(c0[row, i])
            cc1 = int(c1[row, i])
            if a and not b:
                forced_count0 += 1
                base += cc0
            elif b and not a:
                forced_count1 += 1
                base += cc1
            elif cc0 < cc1:
                forced_count0 += 1
                pen0to1.append(cc1 - cc0)
                base += cc0
            elif cc1 < cc0:
                forced_count1 += 1
                pen1to0.append(cc0 - cc1)
                base += cc1
            else:
                neutral += 1
                base += cc0
        if dead:
            continue

        a0 = forced_count0
        a1 = forced_count1
        need_x = max(0, 2 - a0)  # extra group-0 reads potentially required
        need_y = max(0, 2 - a1)  # extra group-1 reads potentially required
        if need_x > len(pen1to0) + neutral or need_y > len(pen0to1) + neutral:
            continue

        pen1to0.sort()
        pen0to1.sort()
        sx = [0]
        for p in pen1to0[:need_x]:
            sx.append(sx[-1] + p)
        sy = [0]
        for p in pen0to1[:need_y]:
            sy.append(sy[-1] + p)

        best_extra: int | None = None
        max_flip0 = min(need_x, len(pen1to0))  # strict group-1 reads -> 0
        max_flip1 = min(need_y, len(pen0to1))  # strict group-0 reads -> 1
        for x in range(max_flip0 + 1):
            for y in range(max_flip1 + 1):
                # group-0 count = a0 + x - y + t neutrals placed in group 0
                lo_t = max(0, 2 - a0 - x + y)
                hi_t = min(neutral, n - 2 - a0 - x + y)
                if lo_t <= hi_t:
                    extra = sx[x] + sy[y]
                    if best_extra is None or extra < best_extra:
                        best_extra = extra
        if best_extra is not None:
            out[offset + row] = base + best_extra
    return out


# ----- secondary objective: vectorized capped DP ---------------------------


def feasible_under_cap(
    mm0: np.ndarray,
    mm1: np.ndarray,
    cost0: np.ndarray,
    cost1: np.ndarray,
    feas0: np.ndarray,
    feas1: np.ndarray,
    candidate_ids: np.ndarray,
    target_cost: int,
    cap: int,
    chunk: int = 4096,
) -> list[int]:
    """Candidates reaching ``target_cost`` with every read mismatching <= cap.

    Per candidate the DP keeps, for each possible number of reads assigned to
    group 0, the minimum total mismatch cost achievable; an edge is only
    usable when its side is feasible (allowance) and its mismatch count does
    not exceed ``cap``.  A candidate succeeds iff some group-0 count in
    ``[2, n-2]`` attains ``target_cost``.
    """
    n = mm0.shape[1]
    inf = np.int64(np.iinfo(np.int64).max // 4)
    winners: list[int] = []

    for start in range(0, len(candidate_ids), chunk):
        ids = candidate_ids[start : start + chunk]
        t = len(ids)
        c0v = cost0[ids]
        c1v = cost1[ids]
        a0 = feas0[ids] & (mm0[ids] <= cap)
        a1 = feas1[ids] & (mm1[ids] <= cap)

        work = np.full((t, n + 1), inf, dtype=np.int64)
        work[:, 0] = 0
        for i in range(n):
            nxt = np.full((t, n + 1), inf, dtype=np.int64)
            can0 = a0[:, i]
            can1 = a1[:, i]
            if np.any(can0):
                rows0 = np.where(can0)[0]
                added = work[:, :-1] + c0v[:, i : i + 1]
                nxt[rows0, 1:] = np.minimum(nxt[rows0, 1:], added[rows0])
            if np.any(can1):
                rows1 = np.where(can1)[0]
                add1 = work + c1v[:, i : i + 1]
                nxt[rows1, :] = np.minimum(nxt[rows1, :], add1[rows1])
            work = nxt
        ok = (work[:, 2 : n - 1] == target_cost).any(axis=1)
        winners.extend(int(candidate_ids[start + k]) for k in np.where(ok)[0])
    return winners


# ----- exact assignment DP for reported candidates -------------------------


def solve_assignments(
    m0: list[int],
    m1: list[int],
    c0: list[int],
    c1: list[int],
    f0: list[bool],
    f1: list[bool],
    target_cost: int,
    cap: int,
    limit: int = 2,
) -> list[tuple[int, int]]:
    """Up to ``limit`` lexicographically smallest assignments at optimum.

    Every returned pair ``(actual max mismatches, assignment bits)`` reaches
    ``target_cost`` with every read mismatching at most ``cap`` and both
    groups populated.  When ``cap`` is the optimal secondary-objective value,
    each returned assignment has max mismatches exactly ``cap`` (an
    assignment with a smaller maximum would have been feasible at the
    previous cap).  Reads occupy bits from the most significant end, so
    integer order is the lexicographic order of group labels.
    """
    n = len(m0)
    # state: (group0 count, max-mm group0, max-mm group1) -> (cost, bits)
    dp: dict[tuple[int, int, int], tuple[int, int]] = {(0, 0, 0): (0, 0)}

    for i in range(n):
        bit = 1 << (n - 1 - i)
        can0 = f0[i] and m0[i] <= cap
        can1 = f1[i] and m1[i] <= cap
        nxt: dict[tuple[int, int, int], tuple[int, int]] = {}
        for (cnt, mx0, mx1), (tot, assign) in dp.items():
            if can0:
                t = tot + c0[i]
                if t <= target_cost:
                    key = (cnt + 1, max(mx0, m0[i]), mx1)
                    val = (t, assign)
                    old = nxt.get(key)
                    if old is None or val < old:
                        nxt[key] = val
            if can1:
                t = tot + c1[i]
                if t <= target_cost:
                    key = (cnt, mx0, max(mx1, m1[i]))
                    val = (t, assign | bit)
                    old = nxt.get(key)
                    if old is None or val < old:
                        nxt[key] = val
        dp = nxt

    finals: list[tuple[int, int]] = []  # (bits, max mm)
    for (cnt, mx0, mx1), (tot, assign) in dp.items():
        if tot == target_cost and 2 <= cnt <= n - 2:
            finals.append((assign, max(mx0, mx1)))
    finals.sort()
    return [(mx, bits) for bits, mx in finals[:limit]]


# ----- contaminant mode: joint three-way labelling -------------------------


def _contam_dp_chunk(
    c0v: np.ndarray,
    c1v: np.ndarray,
    penalties: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    cq_limit: int,
) -> np.ndarray:
    """Vectorized three-way labelling DP for a chunk of candidates.

    Every read jointly chooses group 0, group 1 or contaminant.  State is
    ``(reads in group 0 capped at 2, reads in group 1 capped at 2, number of
    contaminants)``: the cap is exact because the only group constraints are
    "at least two non-contaminant reads each", while the contaminant count
    has a hard upper bound.  Group edges carry the mismatch cost and are
    usable only when the side is feasible *and* within the current mismatch
    cap (already encoded in ``a0``/``a1``); the contaminant edge carries the
    fixed penalty and is never mismatch-constrained.

    Returns the final DP table shaped ``(t, 3, 3, cq_limit+1)``.
    """
    t = c0v.shape[0]
    inf = np.int64(np.iinfo(np.int64).max // 4)
    work = np.full((t, 3, 3, cq_limit + 1), inf, dtype=np.int64)
    work[:, 0, 0, 0] = 0

    for i in range(c0v.shape[1]):
        nxt = np.full((t, 3, 3, cq_limit + 1), inf, dtype=np.int64)
        can0 = a0[:, i]
        can1 = a1[:, i]
        if np.any(can0):
            r = np.where(can0)[0]
            added = work[r, :-1, :, :] + c0v[r, i, None, None, None]
            nxt[r, 1:, :, :] = np.minimum(nxt[r, 1:, :, :], added)
            # saturated group-0 reads self-loop on the capped count
            loop = work[r, 2:3, :, :] + c0v[r, i, None, None, None]
            nxt[r, 2:3, :, :] = np.minimum(nxt[r, 2:3, :, :], loop)
        if np.any(can1):
            r = np.where(can1)[0]
            added = work[r, :, :-1, :] + c1v[r, i, None, None, None]
            nxt[r, :, 1:, :] = np.minimum(nxt[r, :, 1:, :], added)
            # saturated group-1 reads self-loop on the capped count
            loop = work[r, :, 2:3, :] + c1v[r, i, None, None, None]
            nxt[r, :, 2:3, :] = np.minimum(nxt[r, :, 2:3, :], loop)
        # contaminant mode guarantees a positive penalty for every read
        added = work[:, :, :, :-1] + np.int64(penalties[i])
        nxt[:, :, :, 1:] = np.minimum(nxt[:, :, :, 1:], added)
        work = nxt
    return work


def contaminant_scores(
    mm0: np.ndarray,
    mm1: np.ndarray,
    cost0: np.ndarray,
    cost1: np.ndarray,
    feas0: np.ndarray,
    feas1: np.ndarray,
    penalties: np.ndarray,
    cq_limit: int,
    cap: int,
    candidate_ids: np.ndarray | None = None,
    chunk: int = 4096,
) -> np.ndarray:
    """Minimum joint objective per candidate (inf where infeasible).

    When ``candidate_ids`` is given, scores are returned only for those rows
    in the same order; otherwise for every candidate row.  ``cap`` bounds the
    mismatch count of *assigned* reads only; contaminant edges ignore it.
    """
    if candidate_ids is None:
        candidate_ids = np.arange(mm0.shape[0], dtype=np.int64)
    scores = np.empty(len(candidate_ids), dtype=np.int64)

    for start in range(0, len(candidate_ids), chunk):
        ids = candidate_ids[start : start + chunk]
        a0 = feas0[ids] & (mm0[ids] <= cap)
        a1 = feas1[ids] & (mm1[ids] <= cap)
        work = _contam_dp_chunk(cost0[ids], cost1[ids], penalties, a0, a1, cq_limit)
        # valid end states: both groups saturated (>=2 reads), any q <= limit
        scores[start : start + len(ids)] = work[:, 2, 2, :].min(axis=1)
    return scores


def contaminant_feasible_under_cap(
    mm0: np.ndarray,
    mm1: np.ndarray,
    cost0: np.ndarray,
    cost1: np.ndarray,
    feas0: np.ndarray,
    feas1: np.ndarray,
    penalties: np.ndarray,
    cq_limit: int,
    candidate_ids: np.ndarray,
    target_score: int,
    cap: int,
) -> list[int]:
    """Candidates reaching ``target_score`` under mismatch ``cap``."""
    scores = contaminant_scores(
        mm0,
        mm1,
        cost0,
        cost1,
        feas0,
        feas1,
        penalties,
        cq_limit,
        cap,
        candidate_ids=candidate_ids,
    )
    ok = scores == target_score
    return [int(candidate_ids[k]) for k in np.where(ok)[0]]


def contaminant_failure_reason(
    feas0: np.ndarray,
    feas1: np.ndarray,
    cq_limit: int,
) -> PhaseError:
    """Distinguish "cap too small" from "too little valid group evidence".

    Ignoring the contaminant count cap, a candidate admits *some* valid
    labelling iff two distinct reads can cover group 0 and two distinct
    reads group 1 (Hall: ``|S0|>=2``, ``|S1|>=2``, ``|S0 u S1|>=4`` with
    ``Sg`` the reads feasible against group g within allowance).  Then the
    minimum contaminant count is ``n - |S0 u S1|``.
    """
    c0 = feas0.sum(axis=1)
    c1 = feas1.sum(axis=1)
    union = (feas0 | feas1).sum(axis=1)
    n = feas0.shape[1]
    viable = (c0 >= 2) & (c1 >= 2) & (union >= 4)
    if not bool(viable.any()):
        return PhaseError(
            "INSUFFICIENT_GROUP_EVIDENCE",
            "no canonical haplotype leaves at least two non-contaminant reads "
            "within allowance for each of the two groups; the extract does not "
            "contain enough valid evidence for both homologues",
        )
    needed = int((n - union[viable]).min())
    return PhaseError(
        "INSUFFICIENT_CONTAMINANT_CAPACITY",
        f"at least {needed} read{'s' if needed != 1 else ''} must be labelled "
        f"contaminant but max_contaminant_reads={cq_limit}; raise the limit or "
        "inspect the extract",
    )


def solve_contaminant_assignments(
    m0: list[int],
    m1: list[int],
    c0: list[int],
    c1: list[int],
    f0: list[bool],
    f1: list[bool],
    penalties: list[int],
    cq_limit: int,
    target_score: int,
    cap: int,
    limit: int = 2,
) -> list[tuple[int, int]]:
    """Up to ``limit`` lexicographically smallest three-way labellings.

    Returns ``(actual max assigned mismatch, base-3 label digits)`` pairs.
    Labels use digits 0/1/2 per read (2 = contaminant), most significant read
    first, so integer order is lexicographic order with contaminants sorting
    last.  State tracks capped group counts, the exact contaminant count and
    the per-group running mismatch maximum.

    Distinct labellings can share a state (e.g. two reads swap sides without
    changing the counts), so each state keeps the two smallest
    ``(score, digits)`` paths: a 2-best DP, which suffices to recover the two
    smallest complete explanations (future edges never depend on which reads
    filled a group, so a costlier prefix of the same state can never catch up
    to the cheapest one).
    """
    n = len(m0)
    dp: dict[tuple[int, int, int, int, int], list[tuple[int, int]]] = {
        (0, 0, 0, 0, 0): [(0, 0)]
    }

    for i in range(n):
        weight = 3 ** (n - 1 - i)
        can0 = f0[i] and m0[i] <= cap
        can1 = f1[i] and m1[i] <= cap
        nxt: dict[tuple[int, int, int, int, int], list[tuple[int, int]]] = {}

        def extend(key, candidates):
            bucket = nxt.setdefault(key, [])
            bucket.extend(candidates)

        for (a0, a1, q, mx0, mx1), paths in dp.items():
            if can0:
                key0 = (min(2, a0 + 1), a1, q, max(mx0, m0[i]), mx1)
                extend(
                    key0,
                    [
                        (tot + c0[i], digits)
                        for tot, digits in paths
                        if tot + c0[i] <= target_score
                    ],
                )
            if can1:
                key1 = (a0, min(2, a1 + 1), q, mx0, max(mx1, m1[i]))
                extend(
                    key1,
                    [
                        (tot + c1[i], digits + weight)
                        for tot, digits in paths
                        if tot + c1[i] <= target_score
                    ],
                )
            if q < cq_limit:  # contaminant edge: penalty, no mismatch bound
                keyq = (a0, a1, q + 1, mx0, mx1)
                extend(
                    keyq,
                    [
                        (tot + penalties[i], digits + 2 * weight)
                        for tot, digits in paths
                        if tot + penalties[i] <= target_score
                    ],
                )
        # k-best DP: the two smallest (score, digits) per state are enough to
        # recover the two globally smallest complete explanations.
        dp = {key: sorted(set(paths))[:2] for key, paths in nxt.items()}

    finals: list[tuple[int, int]] = []  # (digits, max mm)
    for (a0, a1, _q, mx0, mx1), paths in dp.items():
        if a0 == 2 and a1 == 2:
            for tot, digits in paths:
                if tot == target_score:
                    finals.append((digits, max(mx0, mx1)))
    finals.sort()
    return [(mx, digits) for digits, mx in finals[:limit]]


def _labels_to_tuple(digits: int, n: int) -> tuple[int, ...]:
    return tuple((digits // 3 ** (n - 1 - i)) % 3 for i in range(n))


def _hap_from_bits(hap_bits: int, n_sites: int) -> tuple[int, ...]:
    return (0,) + tuple((hap_bits >> (n_sites - 2 - s)) & 1 for s in range(n_sites - 1))


def _bits_to_tuple(bits: int, n: int) -> tuple[int, ...]:
    return tuple((bits >> (n - 1 - i)) & 1 for i in range(n))


def enumerate_solutions(
    n_sites: int, reads: list[Read], cfg: ContaminantConfig | None = None
) -> list[Solution]:
    """Legacy-style accessor: return just the solution list.

    Contaminant callers use :func:`_enumerate_contaminant` directly so they
    can diagnose infeasibility from the feasibility tables.
    """
    if cfg is None:
        return _enumerate_legacy(n_sites, reads)
    solutions, *_ = _enumerate_contaminant(n_sites, reads, cfg)
    return solutions

def _enumerate_legacy(n_sites: int, reads: list[Read]) -> list[Solution]:
    n = len(reads)
    mm0, mm1, cost0, cost1, feas0, feas1 = candidate_tables(n_sites, reads)
    c_count = 1 << (n_sites - 1)

    # Pass 1: exact minimum cost per candidate, in lexicographic chunk order.
    min_costs: dict[int, int] = {}
    global_best: int | None = None
    for base in range(0, c_count, 4096):
        stop = min(c_count, base + 4096)
        part = min_cost_for_rows(
            mm0[base:stop],
            mm1[base:stop],
            cost0[base:stop],
            cost1[base:stop],
            feas0[base:stop],
            feas1[base:stop],
            base,
        )
        if part:
            min_costs.update(part)
            chunk_best = min(part.values())
            if global_best is None or chunk_best < global_best:
                global_best = chunk_best
    if global_best is None:
        return []

    tied = np.asarray(
        sorted(hb for hb, mc in min_costs.items() if mc == global_best), dtype=np.int64
    )

    # Pass 2: rising cap on the per-read mismatch count.  The first cap at
    # which any tied candidate reaches the minimum cost with both groups
    # populated is the optimal secondary objective; candidates arrive in
    # lexicographic order, so the first two are the canonical tie set.
    max_span = max(r.end - r.start for r in reads)
    chosen: list[int] = []
    chosen_cap = 0
    for cap in range(0, max_span + 1):
        found = feasible_under_cap(
            mm0, mm1, cost0, cost1, feas0, feas1, tied, global_best, cap=cap
        )
        if found:
            chosen = sorted(found)[:2]
            chosen_cap = cap
            break
    if not chosen:  # pragma: no cover - min_cost already guarantees feasibility
        return []

    # Gather the first two distinct *solutions*.  Two assignments under the
    # same haplotype are distinct solutions (the swap-equivalent copy is the
    # complement-labelled pair, which is already quotiented out); candidates
    # and assignments are both visited in lexicographic order.
    picked: list[tuple[int, int, int]] = []  # (hap bits, max mm, assignment bits)
    for hb in chosen:
        if len(picked) >= 2:
            break
        exact_list = solve_assignments(
            mm0[hb].tolist(),
            mm1[hb].tolist(),
            cost0[hb].tolist(),
            cost1[hb].tolist(),
            feas0[hb].tolist(),
            feas1[hb].tolist(),
            global_best,
            chosen_cap,
            limit=2 - len(picked),
        )
        for maxmm, assign_bits in exact_list:
            picked.append((hb, maxmm, assign_bits))

    solutions: list[Solution] = []
    for hb, maxmm, assign_bits in picked:
        hap = _hap_from_bits(hb, n_sites)
        assignments = _bits_to_tuple(assign_bits, n)
        mm_counts, mm_costs, mm_pos = _assignment_evidence(reads, hap, assignments)
        solutions.append(
            Solution(
                haplotype=hap,
                assignments=assignments,
                total_cost=global_best,
                max_mismatches=maxmm,
                mismatch_counts=mm_counts,
                mismatch_costs=mm_costs,
                mismatch_positions=mm_pos,
            )
        )
    return solutions


def _enumerate_contaminant(
    n_sites: int, reads: list[Read], cfg: ContaminantConfig
) -> tuple[list[Solution], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(solutions, mm0, mm1, feas0, feas1)``.

    On infeasibility the solution list is empty and the caller diagnoses the
    business reason from the feasibility tables.
    """
    n = len(reads)
    mm0, mm1, cost0, cost1, feas0, feas1 = candidate_tables(n_sites, reads)
    penalties = np.asarray([r.contaminant_penalty for r in reads], dtype=np.int64)
    cq_limit = cfg.max_reads
    max_span = max(r.end - r.start for r in reads)

    # Pass 1: the joint objective (non-contaminant mismatch cost + penalties)
    # with the mismatch cap fully relaxed.  Assigned reads must still respect
    # their allowances (feas0/feas1); contaminant edges never do.
    scores = contaminant_scores(
        mm0, mm1, cost0, cost1, feas0, feas1, penalties, cq_limit, cap=max_span
    )
    inf = np.iinfo(np.int64).max // 4
    finite = scores < inf
    if not bool(finite.any()):
        return [], mm0, mm1, feas0, feas1
    global_best = int(scores[finite].min())
    tied = np.asarray(sorted(int(h) for h in np.where(scores == global_best)[0]), dtype=np.int64)

    # Pass 2: rising cap on the per-assigned-read mismatch count.  Contaminant
    # edges stay unconditional, so an outlier never blocks the cap optimum.
    chosen: list[int] = []
    chosen_cap = 0
    for cap in range(0, max_span + 1):
        found = contaminant_feasible_under_cap(
            mm0, mm1, cost0, cost1, feas0, feas1, penalties, cq_limit, tied, global_best, cap
        )
        if found:
            chosen = sorted(found)[:2]
            chosen_cap = cap
            break
    if not chosen:  # pragma: no cover - pass 1 already guarantees a winner
        return [], mm0, mm1, feas0, feas1

    picked: list[tuple[int, int, int]] = []  # (hap bits, max mm, label digits)
    for hb in chosen:
        if len(picked) >= 2:
            break
        exact_list = solve_contaminant_assignments(
            mm0[hb].tolist(),
            mm1[hb].tolist(),
            cost0[hb].tolist(),
            cost1[hb].tolist(),
            feas0[hb].tolist(),
            feas1[hb].tolist(),
            [r.contaminant_penalty for r in reads],
            cq_limit,
            global_best,
            chosen_cap,
            limit=2 - len(picked),
        )
        for maxmm, digits in exact_list:
            picked.append((hb, maxmm, digits))

    solutions: list[Solution] = []
    for hb, maxmm, digits in picked:
        hap = _hap_from_bits(hb, n_sites)
        labels = _labels_to_tuple(digits, n)
        mm_counts, mm_costs, mm_pos = _assignment_evidence(reads, hap, labels)
        pen_paid = tuple(
            (reads[i].contaminant_penalty if g == 2 else 0) for i, g in enumerate(labels)
        )
        solutions.append(
            Solution(
                haplotype=hap,
                assignments=labels,
                total_cost=sum(c for g, c in zip(labels, mm_costs) if g != 2),
                max_mismatches=maxmm,
                mismatch_counts=mm_counts,
                mismatch_costs=mm_costs,
                mismatch_positions=mm_pos,
                contaminant_penalties=pen_paid,
                total_penalty=sum(pen_paid),
            )
        )
    return solutions, mm0, mm1, feas0, feas1


def _assignment_evidence(
    reads: list[Read], hap: tuple[int, ...], assignments: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Per-read mismatch count/cost/positions.

    Contaminant-labelled reads (2) contribute zeros and no positions: they
    are not scored against either homologue.
    """
    mm_counts: list[int] = []
    mm_costs: list[int] = []
    mm_pos: list[tuple[int, ...]] = []
    for r, g in zip(reads, assignments):
        count = 0
        spent = 0
        positions: list[int] = []
        if g != 2:
            for k, site in enumerate(range(r.start, r.end)):
                mismatch = (r.obs[k] != hap[site]) if g == 0 else (r.obs[k] == hap[site])
                if mismatch:
                    count += 1
                    spent += r.costs[k]
                    positions.append(site)
        mm_counts.append(count)
        mm_costs.append(spent)
        mm_pos.append(tuple(positions))
    return tuple(mm_counts), tuple(mm_costs), tuple(mm_pos)


def phase(payload: object) -> dict:
    """Validate, solve and build the API response payload."""
    n_sites, reads, cfg = parse_input(payload)

    if cfg is None:
        solutions = enumerate_solutions(n_sites, reads)
        if not solutions:
            raise PhaseError(
                "NO_SOLUTION",
                "no complementary haplotype pair admits an assignment with at "
                "least two reads in each group within the mismatch allowances",
            )
        return _build_response(n_sites, reads, solutions, cfg=None)

    solutions, _mm0, _mm1, feas0, feas1 = _enumerate_contaminant(n_sites, reads, cfg)
    if not solutions:
        raise contaminant_failure_reason(feas0, feas1, cfg.max_reads)
    return _build_response(n_sites, reads, solutions, cfg=cfg)


def _contaminant_reason(read: Read, mm_vs_hap: int, mm_vs_comp: int) -> str:
    """Human-readable evidence for why a read was labelled contaminant."""
    allow = read.max_mismatches
    over_hap = mm_vs_hap > allow
    over_comp = mm_vs_comp > allow
    if over_hap and over_comp:
        return (
            f"{mm_vs_hap} mismatch(es) vs the haplotype and {mm_vs_comp} vs the "
            f"complement both exceed the allowance of {allow}; the read fits "
            "neither homologue"
        )
    if over_hap:
        return (
            f"{mm_vs_hap} mismatch(es) vs the haplotype exceed the allowance of "
            f"{allow} (only {mm_vs_comp} vs the complement); joint optimization "
            f"pays penalty {read.contaminant_penalty} instead of distorting a group"
        )
    if over_comp:
        return (
            f"{mm_vs_comp} mismatch(es) vs the complement exceed the allowance of "
            f"{allow} (only {mm_vs_hap} vs the haplotype); joint optimization "
            f"pays penalty {read.contaminant_penalty} instead of distorting a group"
        )
    return (
        f"within the allowance of {allow} against both homologues "
        f"({mm_vs_hap}/{mm_vs_comp} mismatches), but the joint optimum pays the "
        f"penalty of {read.contaminant_penalty} rather than force it into a group"
    )


def _build_response(
    n_sites: int,
    reads: list[Read],
    solutions: list[Solution],
    cfg: ContaminantConfig | None,
) -> dict:
    contaminant_mode = cfg is not None

    def serialize(sol: Solution) -> dict:
        groups: list[list[str]] = [[], []]
        contaminant_ids: list[str] = []
        contaminant_details: list[dict] = []
        per_read = []
        for i, (r, g, mc, mco, pos) in enumerate(
            zip(
                reads,
                sol.assignments,
                sol.mismatch_counts,
                sol.mismatch_costs,
                sol.mismatch_positions,
            )
        ):
            if g == 2:
                contaminant_ids.append(r.id)
                row = {
                    "id": r.id,
                    "group": 2,
                    "mismatch_count": None,
                    "mismatch_cost": None,
                    "mismatch_positions": [],
                    "contaminant": True,
                    "contaminant_penalty": sol.contaminant_penalties[i],
                }
                per_read.append(row)
            else:
                groups[g].append(r.id)
                row = {
                    "id": r.id,
                    "group": g,
                    "mismatch_count": mc,
                    "mismatch_cost": mco,
                    "mismatch_positions": list(pos),
                }
                if contaminant_mode:
                    row["contaminant"] = False
                    row["contaminant_penalty"] = 0
                per_read.append(row)

        out = {
            "haplotype": list(sol.haplotype),
            "complement": [1 - b for b in sol.haplotype],
            "groups": {"haplotype": groups[0], "complement": groups[1]},
            "assignments": list(sol.assignments),
            "per_read": per_read,
            "total_mismatch_cost": sol.total_cost,
            "max_per_read_mismatches": sol.max_mismatches,
            "mismatch_positions": sorted({p for tup in sol.mismatch_positions for p in tup}),
        }
        if contaminant_mode:
            # Contaminant evidence: recompute each outlier's mismatch profile
            # against this solution's canonical haplotype and its complement.
            for i, g in enumerate(sol.assignments):
                if g != 2:
                    continue
                r = reads[i]
                mm_vs_hap = _mm_for_read(sol.haplotype, r, side=0)
                mm_vs_comp = (r.end - r.start) - mm_vs_hap
                reason = _contaminant_reason(r, mm_vs_hap, mm_vs_comp)
                per_read[i]["contaminant_reason"] = reason
                contaminant_details.append(
                    {
                        "id": r.id,
                        "penalty": sol.contaminant_penalties[i],
                        "mismatches_vs_haplotype": mm_vs_hap,
                        "mismatches_vs_complement": mm_vs_comp,
                        "max_mismatches_allowance": r.max_mismatches,
                        "reason": reason,
                    }
                )
            out["groups"]["contaminants"] = contaminant_ids
            out["contaminants"] = contaminant_details
            out["contaminant_count"] = len(contaminant_ids)
            out["total_contaminant_penalty"] = sol.total_penalty
            out["objective_mismatch_cost_plus_penalty"] = sol.total_cost + sol.total_penalty
        return out

    serialized = [serialize(sol) for sol in solutions]

    response: dict = {
        "n_sites": n_sites,
        "unique": len(solutions) == 1,
        "solutions": serialized,
    }
    response["solution"] = response["solutions"][0]
    if contaminant_mode:
        response["contaminant_mode"] = {
            "enabled": True,
            "max_contaminant_reads": cfg.max_reads,
        }
    if len(solutions) > 1:
        first = (
            "total mismatch cost plus contaminant penalty"
            if contaminant_mode
            else "total_mismatch_cost"
        )
        response["note"] = (
            "two distinct solutions tie on ("
            + first
            + ", max_per_read_mismatches); they may differ in haplotype, in the "
            "per-read assignment"
            + (" or in the contaminant labelling" if contaminant_mode else "")
            + ". The first two canonical solutions are returned"
        )
    return response


def _mm_for_read(hap: tuple[int, ...], r: Read, side: int) -> int:
    """Mismatch count of one read against the haplotype (side 0) or complement."""
    count = 0
    for k, site in enumerate(range(r.start, r.end)):
        mismatch = (r.obs[k] != hap[site]) if side == 0 else (r.obs[k] == hap[site])
        if mismatch:
            count += 1
    return count


