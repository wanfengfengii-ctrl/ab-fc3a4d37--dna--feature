"""Complementary-haplotype phasing core.

Jointly recovers a pair of complementary binary haplotypes from degraded
molecular reads.  For every canonical haplotype candidate (first site fixed
to ``0``, which removes the group-swap symmetry) each read is assigned to
exactly one of the two groups (haplotype / complement) so that:

1. the read's number of mismatching positions does not exceed its allowance,
2. each group holds at least two reads,
3. total mismatch cost is minimized, then
4. the largest per-read mismatch count is minimized, then
5. the assignment (and, on full ties, the haplotype) is lexicographically
   smallest -- giving a stable decision between uniqueness and ambiguity.

Contaminant mode (``max_contaminant_reads`` 1..4 plus a positive
``contaminant_penalty`` per read) widens every read's joint choice to three
states: haplotype group, complement group or *contaminant*.  Contaminant
reads ignore the per-read mismatch allowance, at most ``q`` of them are
accepted, and each of the two groups must still hold at least two
non-contaminant reads.  The optimization is genuinely joint -- the
contaminant set is chosen inside the same DP, never by solving the legacy
problem first and removing outliers afterwards:

1. minimize (non-contaminant mismatch cost) + (contaminant penalties);
2. then minimize the largest per-read mismatch count among non-contaminants;
3. then lexicographically order (canonical haplotype, ternary assignment
   with labels 0=haplotype, 1=complement, 2=contaminant) and report the
   first one or two distinct full explanations.

Legacy algorithm (contaminant mode disabled, n_sites <= 18, reads <= 36):

* all 2**(n_sites-1) canonical candidates are tabulated with vectorized
  numpy (per-read mismatch counts and costs against each side);
* an O(reads) greedy analysis gives the exact minimum cost per
  candidate (group bounds >= 2 never require flipping more than two reads to
  their dearer side, and equal-cost neutral reads fill deficits for free);
* among candidates attaining the global minimum cost, a vectorized dynamic
  program whose only state is the group-0 count is run with a rising cap K on
  the per-read mismatch count.  The first K at which the minimum cost is
  reachable with 2..n-2 reads in group 0 is optimal; the first two distinct
  candidates feasible there are the reported tie set;
* a final exact (Python) DP for those at most two candidates yields the
  lexicographically smallest assignment and its mismatch evidence.
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
    contaminant_penalty: int | None = None


@dataclass(frozen=True)
class Solution:
    haplotype: tuple[int, ...]
    assignments: tuple[int, ...]  # 0/1 (legacy) or 0/1/2 per read
    total_cost: int  # non-contaminant mismatch cost
    max_mismatches: int
    mismatch_counts: tuple[int, ...]
    mismatch_costs: tuple[int, ...]
    mismatch_positions: tuple[tuple[int, ...], ...]  # global site indices
    # contaminant-mode only (empty / zero when the feature is disabled):
    contaminants: tuple[int, ...] = ()  # input-order indices of contaminants
    penalty_total: int = 0
    per_read_penalty: tuple[int, ...] = ()


def _as_int_list(value, what: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PhaseError("INVALID_INPUT", f"{what} must be a non-empty list")
    out: list[int] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise PhaseError("INVALID_INPUT", f"{what}[{i}] must be an integer")
        out.append(item)
    return out


def parse_input(payload: object) -> tuple[int, list[Read]]:
    """Legacy parse entry point: ``(n_sites, reads)``.

    Kept as a 2-tuple for backwards compatibility; contaminant mode is
    rejected here (a cap without this helper's awareness would be silently
    ignored).  Use :func:`parse_input_full` to accept contaminant options.
    """
    n_sites, reads, max_contaminants = parse_input_full(payload)
    if max_contaminants is not None:  # pragma: no cover - defensive guard
        raise PhaseError(
            "INVALID_INPUT",
            "parse_input() does not support max_contaminant_reads; use parse_input_full()",
        )
    return n_sites, reads


def parse_input_full(payload: object) -> tuple[int, list[Read], int | None]:
    if not isinstance(payload, dict):
        raise PhaseError("INVALID_INPUT", "request body must be a JSON object")

    n_sites = payload.get("n_sites")
    if isinstance(n_sites, bool) or not isinstance(n_sites, int):
        raise PhaseError("INVALID_INPUT", "n_sites must be an integer")
    if not 8 <= n_sites <= 18:
        raise PhaseError("INVALID_INPUT", "n_sites must be between 8 and 18")

    # Contaminant handling is opt-in.  When the cap is absent the request is
    # parsed exactly as a legacy request (any contaminant_penalty field is
    # rejected so the feature state is unambiguous); when present it must be
    # an integer in 1..4 and every read must carry a positive penalty.
    raw_cap = payload.get("max_contaminant_reads")
    if raw_cap is None:
        max_contaminants = None
    else:
        if isinstance(raw_cap, bool) or not isinstance(raw_cap, int):
            raise PhaseError(
                "INVALID_INPUT", "max_contaminant_reads must be an integer between 1 and 4"
            )
        if not 1 <= raw_cap <= 4:
            raise PhaseError(
                "INVALID_INPUT", "max_contaminant_reads must be between 1 and 4"
            )
        max_contaminants = raw_cap

    raw_reads = payload.get("reads")
    if not isinstance(raw_reads, list):
        raise PhaseError("INVALID_INPUT", "reads must be a list")
    if not 10 <= len(raw_reads) <= 36:
        raise PhaseError("INVALID_INPUT", "reads must contain between 10 and 36 items")

    reads: list[Read] = []
    seen_ids: set[str] = set()
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

        penalty: int | None = None
        if max_contaminants is not None:
            raw_penalty = item.get("contaminant_penalty")
            if isinstance(raw_penalty, bool) or not isinstance(raw_penalty, int) or raw_penalty <= 0:
                raise PhaseError(
                    "INVALID_INPUT",
                    f"reads[{idx}].contaminant_penalty must be a positive integer when "
                    "max_contaminant_reads is set",
                )
            penalty = raw_penalty
        elif "contaminant_penalty" in item:
            raise PhaseError(
                "INVALID_INPUT",
                f"reads[{idx}].contaminant_penalty requires max_contaminant_reads (1..4) "
                "to be enabled; remove the penalty or set the cap",
            )

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

    # The optimizer tabulates costs in signed 64-bit integers; reject inputs
    # whose theoretical maximum total cost could interfere with the DP's
    # infinity sentinel (inf ~ 2**61, so path sums must stay far below it).
    grand_total = sum(sum(r.costs) for r in reads)
    if max_contaminants is not None:
        grand_total += sum(r.contaminant_penalty for r in reads)  # type: ignore[misc]
    if grand_total > (1 << 58):
        raise PhaseError(
            "INVALID_INPUT",
            "sum of mismatch costs and contaminant penalties is too large to score "
            "exactly; values must be small positive integers (aggregate must fit in 59 bits)",
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

    return n_sites, reads, max_contaminants


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


# ----- minimum-cost greedy analysis (legacy path) ---------------------------


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


# ----- secondary objective: vectorized capped DP (legacy path) --------------


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


# ----- exact assignment DP for reported candidates (legacy path) ------------


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


# ----- contaminant-mode joint DP --------------------------------------------


def _contaminant_state_layout(m: int, q: int):
    """Index scheme for flattened DP state ``(group0 count, contaminant count)``.

    State ``s = c * (q + 1) + t`` with ``0 <= c <= m`` and ``0 <= t <= q``.
    Pre-computed source/target index maps make each transition a single
    vectorized scatter; ``final`` marks states satisfying both group bounds.
    """
    width = q + 1
    states = np.arange((m + 1) * width, dtype=np.int64)
    cc = states // width
    tt = states % width

    src0 = states[cc < m]  # group 0: c -> c+1
    tgt0 = src0 + width
    src1 = states  # group 1: c unchanged
    src_c = states[tt < q]  # contaminant: t -> t+1
    tgt_c = src_c + 1

    final = (cc >= 2) & (cc + tt <= m - 2)
    return src0, tgt0, src1, src_c, tgt_c, final


def _contaminant_viable(
    feas0: np.ndarray, feas1: np.ndarray, candidate_ids: np.ndarray, q: int
) -> np.ndarray:
    """Cheap necessary-condition screen for the joint DP.

    A candidate cannot be feasible when it forces more than ``q`` reads
    (those exceeding their allowance against both sides) into contaminant
    status, or when the feasible reads cannot place at least two in each
    group.  These are only necessary conditions, so the exact DP still has
    the final word; they merely keep hopeless candidates out of it.
    """
    f0 = feas0[candidate_ids]
    f1 = feas1[candidate_ids]
    mandatory = ((~f0) & (~f1)).sum(axis=1)
    only0 = (f0 & ~f1).sum(axis=1)
    only1 = (f1 & ~f0).sum(axis=1)
    either = (f0 & f1).sum(axis=1)
    keep = (
        (mandatory <= q)
        & (only0 + either >= 2)
        & (only1 + either >= 2)
        & (only0 + only1 + either >= 4)
    )
    return candidate_ids[keep]


def contaminant_dp_costs(
    mm0: np.ndarray,
    mm1: np.ndarray,
    cost0: np.ndarray,
    cost1: np.ndarray,
    feas0: np.ndarray,
    feas1: np.ndarray,
    penalties: np.ndarray,
    q: int,
    candidate_ids: np.ndarray,
    cap: int | None = None,
    chunk: int = 1024,
) -> dict[int, int]:
    """Exact minimum of (non-contaminant mismatch cost + penalties) per candidate.

    Every read jointly chooses group 0, group 1 or contaminant.  Group edges
    require allowance feasibility and, when ``cap`` is given, a mismatch
    count at or below it; the contaminant edge always exists (it bypasses the
    allowance) but at most ``q`` such edges may be taken.  Final states must
    leave at least two reads in each group.  Returns ``{candidate_id: cost}``
    for feasible rows only.
    """
    m = mm0.shape[1]
    inf = np.int64(np.iinfo(np.int64).max // 4)
    src0, tgt0, src1, src_c, tgt_c, final = _contaminant_state_layout(m, q)
    n_states = (m + 1) * (q + 1)
    out: dict[int, int] = {}

    # Necessary-condition pre-screen (allowance-based, hence valid for every
    # tighter cap as well).
    viable = _contaminant_viable(feas0, feas1, candidate_ids, q)

    for start in range(0, len(viable), chunk):
        ids = viable[start : start + chunk]
        t = len(ids)
        can0 = feas0[ids] if cap is None else feas0[ids] & (mm0[ids] <= cap)
        can1 = feas1[ids] if cap is None else feas1[ids] & (mm1[ids] <= cap)
        c0v = cost0[ids]
        c1v = cost1[ids]

        work = np.full((t, n_states), inf, dtype=np.int64)
        work[:, 0] = 0
        for i in range(m):
            nxt = np.full((t, n_states), inf, dtype=np.int64)

            # contaminant edge: unrestricted, costs the read's penalty
            nxt[:, tgt_c] = work[:, src_c] + np.int64(penalties[i])
            # group edges as branchless min-scatters; impossible edges add inf
            add0 = work[:, src0] + np.where(can0[:, i, None], c0v[:, i : i + 1], inf)
            nxt[:, tgt0] = np.minimum(nxt[:, tgt0], add0)
            add1 = work[:, src1] + np.where(can1[:, i, None], c1v[:, i : i + 1], inf)
            nxt[:, src1] = np.minimum(nxt[:, src1], add1)
            work = nxt

        vals = work[:, final].min(axis=1)
        finite = np.where(vals < inf)[0]
        for k in finite:
            out[int(ids[int(k)])] = int(vals[int(k)])
    return out


def solve_assignments_contaminant(
    m0: list[int],
    m1: list[int],
    c0: list[int],
    c1: list[int],
    f0: list[bool],
    f1: list[bool],
    penalties: list[int],
    target_cost: int,
    cap: int,
    q: int,
    limit: int = 2,
) -> list[tuple[int, int]]:
    """Up to ``limit`` lexicographically smallest three-way assignments.

    Same joint semantics as :func:`contaminant_dp_costs` at the optimal cap;
    the assignment is encoded in base 3, most significant read first, with
    digits ``0`` (haplotype), ``1`` (complement) and ``2`` (contaminant), so
    integer order is exactly the required three-way lexicographic order.
    Returns ``(max non-contaminant mismatches, ternary code)`` pairs.
    """
    n = len(m0)
    pow3 = [3 ** (n - 1 - i) for i in range(n)]
    # state: (group0 count, contaminant count, max-mm g0, max-mm g1)
    dp: dict[tuple[int, int, int, int], tuple[int, int]] = {(0, 0, 0, 0): (0, 0)}

    for i in range(n):
        digit = pow3[i]
        edge0 = f0[i] and m0[i] <= cap
        edge1 = f1[i] and m1[i] <= cap
        nxt: dict[tuple[int, int, int, int], tuple[int, int]] = {}
        for (cnt, tcnt, mx0, mx1), (tot, code) in dp.items():
            if edge0:
                ntot = tot + c0[i]
                if ntot <= target_cost:
                    key = (cnt + 1, tcnt, max(mx0, m0[i]), mx1)
                    val = (ntot, code)
                    old = nxt.get(key)
                    if old is None or val < old:
                        nxt[key] = val
            if edge1:
                ntot = tot + c1[i]
                if ntot <= target_cost:
                    key = (cnt, tcnt, mx0, max(mx1, m1[i]))
                    val = (ntot, code + digit)
                    old = nxt.get(key)
                    if old is None or val < old:
                        nxt[key] = val
            if tcnt < q:  # contaminant: no allowance cap, digit 2
                ntot = tot + penalties[i]
                if ntot <= target_cost:
                    key = (cnt, tcnt + 1, mx0, mx1)
                    val = (ntot, code + 2 * digit)
                    old = nxt.get(key)
                    if old is None or val < old:
                        nxt[key] = val
        dp = nxt

    finals: list[tuple[int, int]] = []  # (ternary code, max mm)
    for (cnt, tcnt, mx0, mx1), (tot, code) in dp.items():
        if tot == target_cost and cnt >= 2 and cnt + tcnt <= n - 2:
            finals.append((code, max(mx0, mx1)))
    finals.sort()
    return [(mx, code) for code, mx in finals[:limit]]


class _ContaminantInfeasible(Exception):
    """Internal: contaminant-mode joint problem has no feasible solution."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def contaminant_failure_reason(
    feas0: np.ndarray, feas1: np.ndarray, q: int
) -> tuple[str, str]:
    """Classify why no joint (haplotype, complement, contaminant) solution exists.

    A read is *mandatory contaminant* for a candidate when it exceeds its
    allowance against both sides.  If every candidate needs more than ``q``
    mandatory contaminants the cap is the binding constraint; otherwise some
    candidate could hold all mandatory contaminants but the remaining reads
    cannot supply at least two feasible reads to each group.
    """
    forced_both = (~feas0) & (~feas1)
    required = forced_both.sum(axis=1)
    cap_ok = np.where(required <= q)[0]
    if cap_ok.size == 0:
        min_required = int(required.min())
        return (
            "INSUFFICIENT_CONTAMINANT_CAPACITY",
            f"every canonical haplotype forces at least {min_required} read(s) to "
            f"exceed their mismatch allowance against both homologous groups, but "
            f"max_contaminant_reads={q} can exempt at most {q}; raise the cap or "
            "loosen the affected reads' max_mismatches",
        )

    only0 = (feas0 & ~feas1).sum(axis=1)
    only1 = (feas1 & ~feas0).sum(axis=1)
    either = (feas0 & feas1).sum(axis=1)
    # Positive deficit: the either-side pool cannot fill both groups' needs.
    deficits = np.maximum(0, 2 - only0) + np.maximum(0, 2 - only1) - either
    # Among cap-feasible candidates prefer the smallest group-evidence
    # deficit (D <= 0 means the either-side pool can fill both bounds), then
    # the largest pool of usable reads.
    order = sorted(
        (int(k) for k in cap_ok),
        key=lambda h: (
            int(deficits[h]),
            -(int(only0[h]) + int(only1[h]) + int(either[h])),
        ),
    )
    h = order[0]
    return (
        "INSUFFICIENT_GROUP_EVIDENCE",
        f"no canonical haplotype leaves at least two feasible non-contaminant reads "
        f"per group (mandatory contaminants fit within max_contaminant_reads={q}). "
        f"Best candidate: {int(only0[h])} read(s) support only the haplotype side, "
        f"{int(only1[h])} only the complement side, {int(either[h])} either side; "
        "both homologous groups require >=2 non-contaminant reads",
    )


# ----- top level ------------------------------------------------------------


def _hap_from_bits(hap_bits: int, n_sites: int) -> tuple[int, ...]:
    return (0,) + tuple((hap_bits >> (n_sites - 2 - s)) & 1 for s in range(n_sites - 1))


def _bits_to_tuple(bits: int, n: int, base: int = 2) -> tuple[int, ...]:
    return tuple((bits // (base ** (n - 1 - i))) % base for i in range(n))


def enumerate_solutions(
    n_sites: int, reads: list[Read], max_contaminant_reads: int | None = None
) -> list[Solution]:
    n = len(reads)
    tables = candidate_tables(n_sites, reads)
    mm0, mm1, cost0, cost1, feas0, feas1 = tables
    c_count = 1 << (n_sites - 1)

    if max_contaminant_reads is not None:
        return _enumerate_contaminant(
            n_sites, reads, max_contaminant_reads, tables
        )

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

    return [
        _build_solution(reads, n_sites, hb, _bits_to_tuple(bits, n, base=2), maxmm)
        for hb, maxmm, bits in picked
    ]


def _enumerate_contaminant(
    n_sites: int,
    reads: list[Read],
    q: int,
    tables,
) -> list[Solution]:
    mm0, mm1, cost0, cost1, feas0, feas1 = tables
    n = len(reads)
    penalties = np.asarray([r.contaminant_penalty for r in reads], dtype=np.int64)
    all_candidates = np.arange(1 << (n_sites - 1), dtype=np.int64)

    # Pass 1: joint minimum objective (non-contaminant cost + penalties).
    min_costs = contaminant_dp_costs(
        mm0, mm1, cost0, cost1, feas0, feas1, penalties, q, all_candidates
    )
    if not min_costs:
        code, message = contaminant_failure_reason(feas0, feas1, q)
        raise _ContaminantInfeasible(code, message)
    global_best = min(min_costs.values())
    tied = np.asarray(sorted(hb for hb, v in min_costs.items() if v == global_best), dtype=np.int64)

    # Pass 2: rising cap on non-contaminant per-read mismatch counts.
    max_span = max(r.end - r.start for r in reads)
    chosen: list[int] = []
    chosen_cap = 0
    for cap in range(0, max_span + 1):
        found = contaminant_dp_costs(
            mm0, mm1, cost0, cost1, feas0, feas1, penalties, q, tied, cap=cap
        )
        winners = sorted(hb for hb, v in found.items() if v == global_best)
        if winners:
            chosen = winners[:2]
            chosen_cap = cap
            break
    if not chosen:  # pragma: no cover - pass 1 already guarantees feasibility
        return []

    # Pass 3: exact three-way assignments (base-3 lexicographic order) for
    # the first one or two candidates.
    picked: list[tuple[int, int, int]] = []  # (hap bits, max mm, ternary code)
    for hb in chosen:
        if len(picked) >= 2:
            break
        exact_list = solve_assignments_contaminant(
            mm0[hb].tolist(),
            mm1[hb].tolist(),
            cost0[hb].tolist(),
            cost1[hb].tolist(),
            feas0[hb].tolist(),
            feas1[hb].tolist(),
            penalties.tolist(),
            global_best,
            chosen_cap,
            q,
            limit=2 - len(picked),
        )
        for maxmm, code in exact_list:
            picked.append((hb, maxmm, code))

    solutions: list[Solution] = []
    for hb, maxmm, code in picked:
        labels = _bits_to_tuple(code, n, base=3)
        solutions.append(
            _build_solution(reads, n_sites, hb, labels, maxmm, penalties=penalties)
        )
    return solutions


def _build_solution(
    reads: list[Read],
    n_sites: int,
    hap_bits: int,
    labels: tuple[int, ...],
    maxmm: int,
    penalties: np.ndarray | None = None,
) -> Solution:
    hap = _hap_from_bits(hap_bits, n_sites)

    mm_counts: list[int] = []
    mm_costs: list[int] = []
    mm_pos: list[tuple[int, ...]] = []
    contaminants: list[int] = []
    per_read_penalty: list[int] = []
    noncontam_cost = 0

    for i, (r, g) in enumerate(zip(reads, labels)):
        if g == 2:
            contaminants.append(i)
            paid = int(penalties[i]) if penalties is not None else 0
            per_read_penalty.append(paid)
            mm_counts.append(0)
            mm_costs.append(0)
            mm_pos.append(())
            continue
        per_read_penalty.append(0)
        count = 0
        spent = 0
        positions: list[int] = []
        for k, site in enumerate(range(r.start, r.end)):
            mismatch = (r.obs[k] != hap[site]) if g == 0 else (r.obs[k] == hap[site])
            if mismatch:
                count += 1
                spent += r.costs[k]
                positions.append(site)
        mm_counts.append(count)
        mm_costs.append(spent)
        mm_pos.append(tuple(positions))
        noncontam_cost += spent

    return Solution(
        haplotype=hap,
        assignments=labels,
        total_cost=noncontam_cost,
        max_mismatches=maxmm,
        mismatch_counts=tuple(mm_counts),
        mismatch_costs=tuple(mm_costs),
        mismatch_positions=tuple(mm_pos),
        contaminants=tuple(contaminants),
        penalty_total=sum(per_read_penalty),
        per_read_penalty=tuple(per_read_penalty),
    )


def phase(payload: object) -> dict:
    """Validate, solve and build the API response payload."""
    n_sites, reads, max_contaminants = parse_input_full(payload)
    try:
        solutions = enumerate_solutions(n_sites, reads, max_contaminants)
    except _ContaminantInfeasible as exc:
        raise PhaseError(exc.code, exc.message) from None
    if not solutions:
        raise PhaseError(
            "NO_SOLUTION",
            "no complementary haplotype pair admits an assignment with at "
            "least two reads in each group within the mismatch allowances",
        )

    contam_mode = max_contaminants is not None

    def side_evidence(r: Read, hap: tuple[int, ...]) -> tuple[dict, dict]:
        against0: list[int] = []
        against1: list[int] = []
        cost0 = 0
        cost1 = 0
        for k, site in enumerate(range(r.start, r.end)):
            if r.obs[k] != hap[site]:
                against0.append(site)
                cost0 += r.costs[k]
            else:
                against1.append(site)
                cost1 += r.costs[k]
        return (
            {"count": len(against0), "cost": cost0, "positions": against0},
            {"count": len(against1), "cost": cost1, "positions": against1},
        )

    def serialize(sol: Solution) -> dict:
        groups: list[list[str]] = [[], [], []]
        per_read = []
        for i, (r, g) in enumerate(zip(reads, sol.assignments)):
            groups[g].append(r.id)
            if not contam_mode:
                per_read.append(
                    {
                        "id": r.id,
                        "group": g,
                        "mismatch_count": sol.mismatch_counts[i],
                        "mismatch_cost": sol.mismatch_costs[i],
                        "mismatch_positions": list(sol.mismatch_positions[i]),
                    }
                )
                continue

            if g == 2:
                ev0, ev1 = side_evidence(r, sol.haplotype)
                cap = sol.max_mismatches
                feasible_sides = []
                compliant_options: list[tuple[str, int, int]] = []
                in_cap_options: list[tuple[str, int, int]] = []
                for side, ev in (("haplotype", ev0), ("complement", ev1)):
                    if ev["count"] <= r.max_mismatches:
                        feasible_sides.append(side)
                        compliant_options.append((side, ev["count"], ev["cost"]))
                        if ev["count"] <= cap:
                            in_cap_options.append((side, ev["count"], ev["cost"]))
                if not compliant_options:
                    explanation = (
                        f"exceeds max_mismatches={r.max_mismatches} against both the "
                        f"haplotype ({ev0['count']} mismatches, cost {ev0['cost']}) and "
                        f"the complement ({ev1['count']} mismatches, cost {ev1['cost']}); "
                        f"assigned contaminant (no allowance cap), paying penalty "
                        f"{r.contaminant_penalty}"
                    )
                    cheapest = None
                    cheapest_in_cap = None
                elif not in_cap_options:
                    side, side_mm, cheap_cost = min(
                        compliant_options, key=lambda p: p[2]
                    )
                    explanation = (
                        f"allowance-compliant assignments exist (cheapest: {side}, "
                        f"{side_mm} mismatches, cost {cheap_cost}), but all exceed the "
                        f"solution's optimal per-read mismatch cap of {cap}; marking the "
                        f"read contaminant keeps the maximum non-contaminant mismatch "
                        f"count at {cap} at penalty {r.contaminant_penalty}"
                    )
                    cheapest = {"group": side, "mismatch_count": side_mm, "mismatch_cost": cheap_cost}
                    cheapest_in_cap = None
                else:
                    side, side_mm, cheap_cost = min(in_cap_options, key=lambda p: p[2])
                    penalty = r.contaminant_penalty
                    if penalty < cheap_cost:
                        tradeoff = (
                            f"its penalty {penalty} is lower than the cheapest assignment "
                            f"within the optimal cap ({side}, mismatch cost {cheap_cost})"
                        )
                    else:
                        tradeoff = (
                            f"its penalty {penalty} equals the cheapest assignment within "
                            f"the optimal cap ({side}, mismatch cost {cheap_cost}); the "
                            "group-assigned and contaminant explanations tie on both "
                            "objectives, and this is the later one in canonical "
                            "(three-way lexicographic) order"
                        )
                    explanation = (
                        "joint optimum marks this read as contaminant: "
                        f"{tradeoff}; it is exempt from the mismatch allowance cap"
                    )
                    cheapest_allowance = min(compliant_options, key=lambda p: p[2])
                    cheapest = {
                        "group": cheapest_allowance[0],
                        "mismatch_count": cheapest_allowance[1],
                        "mismatch_cost": cheapest_allowance[2],
                    }
                    cheapest_in_cap = {
                        "group": side,
                        "mismatch_count": side_mm,
                        "mismatch_cost": cheap_cost,
                    }
                per_read.append(
                    {
                        "id": r.id,
                        "group": None,
                        "status": "contaminant",
                        "contaminant": True,
                        "contaminant_penalty": sol.per_read_penalty[i],
                        "contaminant_reason": {
                            "allowed_mismatches": r.max_mismatches,
                            "optimal_max_mismatches": cap,
                            "feasible_groups": feasible_sides,
                            "mismatches_vs_haplotype": ev0,
                            "mismatches_vs_complement": ev1,
                            "cheapest_allowance_compliant_option": cheapest,
                            "cheapest_within_cap_option": cheapest_in_cap,
                            "explanation": explanation,
                        },
                        "mismatch_count": None,
                        "mismatch_cost": None,
                        "mismatch_positions": None,
                    }
                )
            else:
                per_read.append(
                    {
                        "id": r.id,
                        "group": g,
                        "status": "haplotype" if g == 0 else "complement",
                        "contaminant": False,
                        "contaminant_penalty": 0,
                        "contaminant_reason": None,
                        "mismatch_count": sol.mismatch_counts[i],
                        "mismatch_cost": sol.mismatch_costs[i],
                        "mismatch_positions": list(sol.mismatch_positions[i]),
                    }
                )

        serialized = {
            "haplotype": list(sol.haplotype),
            "complement": [1 - b for b in sol.haplotype],
            "groups": {"haplotype": groups[0], "complement": groups[1]},
            "assignments": list(sol.assignments),
            "per_read": per_read,
            "total_mismatch_cost": sol.total_cost,
            "max_per_read_mismatches": sol.max_mismatches,
            "mismatch_positions": sorted(
                {p for tup in sol.mismatch_positions for p in tup}
            ),
        }
        if contam_mode:
            serialized["groups"] = {
                "haplotype": groups[0],
                "complement": groups[1],
                "contaminants": groups[2],
            }
            serialized["contaminant_reads"] = groups[2]
            serialized["contaminant_count"] = len(groups[2])
            serialized["total_contaminant_penalty"] = sol.penalty_total
            serialized["total_objective_cost"] = sol.total_cost + sol.penalty_total
        return serialized

    response: dict = {
        "n_sites": n_sites,
        "unique": len(solutions) == 1,
        "solutions": [serialize(s) for s in solutions],
    }
    if contam_mode:
        response["max_contaminant_reads"] = max_contaminants
    response["solution"] = response["solutions"][0]
    if len(solutions) > 1:
        if contam_mode:
            response["note"] = (
                "two distinct solutions tie on (non-contaminant mismatch cost + "
                "contaminant penalties, max non-contaminant per-read mismatches); "
                "they may differ in haplotype or in the three-way per-read assignment "
                "(0=haplotype, 1=complement, 2=contaminant). The first two canonical "
                "solutions are returned"
            )
        else:
            response["note"] = (
                "two distinct solutions tie on (total_mismatch_cost, "
                "max_per_read_mismatches); they may differ in haplotype or in the "
                "per-read assignment. The first two canonical solutions are returned"
            )
    return response
