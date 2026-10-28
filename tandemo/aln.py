"""Reference <-> read coordinates of one alignment (after the somatic_sv tr_somatic_instability Aln).

q_at(r) gives the number of read bases before reference position r, including insertions placed
just before r. The repeat length of a spanning read is then a difference of two q_at values taken
in the unique flanks, so it does not depend on where minimap2 put the indels inside the repeat.
"""

from __future__ import annotations

import bisect

REF_OPS = {0, 2, 3, 7, 8}       # M D N = X consume the reference
QRY_OPS = {0, 1, 4, 7, 8}       # M I S = X consume the query
MATCH_OPS = {0, 7, 8}


class Aln:
    def __init__(self, read):
        self.read = read
        ops, rs, qs, ln = [], [], [], []
        r, q = read.reference_start, 0
        for op, n in read.cigartuples:
            ops.append(op); rs.append(r); qs.append(q); ln.append(n)
            if op in REF_OPS:
                r += n
            if op in QRY_OPS:
                q += n
        self.ops, self.rs, self.qs, self.ln = ops, rs, qs, ln
        self.ref_idx = [i for i, o in enumerate(ops) if o in REF_OPS]
        self.ref_starts = [rs[i] for i in self.ref_idx]
        self.start, self.end = read.reference_start, read.reference_end
        self.qlen = q
        self.clip_left = ln[0] if ops[0] == 4 else 0
        self.clip_right = ln[-1] if ops[-1] == 4 else 0

    def q_at(self, r: int) -> int:
        """Read bases consumed before reference position r (insertions just before r included)."""
        if r <= self.start:
            return self.qs[self.ref_idx[0]] if self.ref_idx else 0
        if r >= self.end:
            return self.qlen - self.clip_right
        j = bisect.bisect_right(self.ref_starts, r) - 1
        i = self.ref_idx[j]
        return self.qs[i] + (r - self.rs[i] if self.ops[i] in MATCH_OPS else 0)

    def query_pos(self, r: int) -> int | None:
        """Read position aligned to reference position r, or None when r is deleted or not covered."""
        if r < self.start or r >= self.end:
            return None
        j = bisect.bisect_right(self.ref_starts, r) - 1
        i = self.ref_idx[j]
        if self.ops[i] not in MATCH_OPS:
            return None
        return self.qs[i] + r - self.rs[i]

    def flank_identity(self, a: int, b: int, ref: str, ref_off: int, seq: str) -> float:
        """Alignment identity over reference [a, b): mismatches, inserted and deleted bases count as
        errors. -1 when the read does not cover [a, b) or is clipped inside it."""
        if a < self.start or b > self.end:
            return -1.0
        errors = 0
        j = max(bisect.bisect_right(self.ref_starts, a) - 1, 0)
        i = self.ref_idx[j] if self.ref_idx else 0
        while i < len(self.ops):
            op, r0, q0, n = self.ops[i], self.rs[i], self.qs[i], self.ln[i]
            if r0 >= b and op in REF_OPS:
                break
            if op == 1:
                if a < r0 < b:
                    errors += n
            elif op in (2, 3):
                errors += max(0, min(r0 + n, b) - max(r0, a))
            elif op in MATCH_OPS:
                lo, hi = max(r0, a), min(r0 + n, b)
                if hi > lo and op != 7:
                    rq = ref[lo - ref_off:hi - ref_off]
                    qq = seq[q0 + lo - r0:q0 + hi - r0]
                    errors += sum(x != y for x, y in zip(rq, qq))
            elif op == 4 and a < r0 < b:
                return -1.0
            i += 1
        return 1.0 - errors / (b - a)
