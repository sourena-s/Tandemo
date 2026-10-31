"""Stage 2: the distribution of true repeat lengths per locus and haplotype.

Inputs per locus: each read's Stage 1 curve log l_i(L) on an integer grid, its phase weight w1_i
(NaN without het evidence), and the selection curve s(L).

Per haplotype h, f_h(L) is a free distribution on the grid: the nonparametric maximum-likelihood estimate
(NPMLE, Kiefer-Wolfowitz) of the mixing distribution. Reads keep their haplotype by phase evidence, not length:

    a_ih = (w1_i, 1 - w1_i)    when the read has het evidence
         = (rho, 1 - rho)       when it has none; rho is fitted

    p_i  = sum_h a_ih  sum_L l_i(L) f_h(L) s(L) / Z_h,        Z_h = sum_L f_h(L) s(L)

sum_i log p_i is concave in f, so EM reaches the global maximum. The NPMLE is discrete (a few atoms). It is not
smoothed: smoothing blurs the germline atom into |d| >= Delta and inflates the spread. s(L) corrects for spanning
selection: a molecule of length L is only seen when a read covers it plus both flanks (s(L) proportional to
the sum over reads overlapping the locus of max(0, R - L - 2 * flank)), so f_h is the distribution of
molecules, not of kept reads. M0 (one length per haplotype) is fitted first: it starts EM and gives the
reference log-likelihood.

Per haplotype, on the deviation scale d = L - L0 and Delta = max(3, period). L0 is the germline allele (the anchor
nearest the mode of f_h; the mode itself without anchors), so a spread f whose mode wanders off the germline is
still measured from it:
U = P(|d| >= Delta) (the unstable fraction), p_exp = P(d >= Delta), p_con = P(d <= -Delta),
mad = E|d| (bp; the Wasserstein-1 distance from a stable allele: the spread), mean_shift = E[d] (bp: the direction),
n_atoms (grid points with mass >= 1%), the reads it received (sum of posteriors) and unphased_var_frac (the
share of its unstable part carried by reads without het evidence; when high, trust the pooled numbers).
--bootstrap B adds 95% intervals (U_lo/U_hi, mad_lo/mad_hi) from B read resamples. f_h itself is kept (f_L,
f_p: grid points with mass >= 1e-4) so samples can be compared later without refitting.
Locus level (pooled_*): the haplotypes' deviation distributions, each on its own d scale, mixed by their read
shares; the germline lengths drop out, so the same locus compares across samples whatever its alleles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FitParams:
    min_reads: int = 3
    iters: int = 5000
    tol: float = 1e-9                  # EM stops when the log-likelihood gains less than tol * |ll|
    bootstrap: int = 0                 # read resamples for 95% intervals (0: none)
    seed: int = 0
    selection: bool = True
    flank: int = 128


def selection_curve(grid: np.ndarray, read_lens: str | None, flank: int) -> np.ndarray:
    if not read_lens or read_lens == "." or read_lens != read_lens:
        return np.ones(len(grid))
    R = np.array([int(x) for x in str(read_lens).split(",") if x], dtype=float)
    s = np.clip(R[None, :] - grid[:, None] - 2 * flank, 0, None).sum(1)
    if s.max() <= 0:
        return np.ones(len(grid))
    return np.clip(s / s.max(), 1e-3, 1.0)


class LocusFit:
    """One locus. logl [n, G], grid [G], w1 [n] (NaN = no evidence), H = 1 or 2."""

    def __init__(self, logl, grid, w1, H, period, s=None, anchors=(), p: FitParams | None = None):
        self.p = p or FitParams()
        self.grid = np.asarray(grid)
        self.G = len(self.grid)
        self.m = logl.max(1)
        self.E = np.exp(logl - self.m[:, None])                 # [n, G], rows max 1
        self.n = len(self.m)
        self.H = H
        w1 = np.asarray(w1, dtype=float)
        self.unph = np.isnan(w1) if H == 2 else np.zeros(self.n, bool)
        self.w = np.stack([np.nan_to_num(w1, nan=0.5), 1 - np.nan_to_num(w1, nan=0.5)], 1)
        self.period = max(int(period), 1)
        self.s = np.ones(self.G) if s is None else np.asarray(s, dtype=float)
        self.anchors = [a for a in anchors if a is not None and a >= 0]
        self.delta = max(3, self.period)

    # -------------------------------------------------- shared pieces
    def prior(self, rho: float) -> np.ndarray:
        if self.H == 1:
            return np.ones((self.n, 1))
        a = self.w.copy()
        a[self.unph] = [rho, 1 - rho]
        return a

    def read_lik(self, f: np.ndarray) -> np.ndarray:
        """P_ih = sum_L E_iL f_h(L) s(L) / Z_h  for f [H, G] -> [n, H] (scaled by exp(-m_i))."""
        q = f * self.s[None, :]
        q = q / q.sum(1, keepdims=True)
        return self.E @ q.T

    def loglik(self, f: np.ndarray, rho: float) -> float:
        P = (self.prior(rho) * self.read_lik(f)).sum(1)
        return float(np.sum(np.log(np.maximum(P, 1e-300)) + self.m))

    def idx(self, L: float) -> int:
        return int(np.argmin(np.abs(self.grid - L)))

    # -------------------------------------------------- M0: one length per haplotype
    def _m0_from(self, k, rho):
        k = list(k)
        for _ in range(8):
            prev = (tuple(k), rho)
            for h in (0, 1):
                a = self.prior(rho)
                other = a[:, 1 - h] * self.E[:, k[1 - h]]
                tot = np.log(np.maximum(a[:, h:h + 1] * self.E + other[:, None], 1e-300)).sum(0)
                k[h] = int(np.argmax(tot))
            if self.unph.any():
                for _ in range(20):
                    a = self.prior(rho)
                    num = a[:, 0] * self.E[:, k[0]]
                    den = num + a[:, 1] * self.E[:, k[1]]
                    rho = float(np.clip(np.mean((num / np.maximum(den, 1e-300))[self.unph]), 1e-3, 1 - 1e-3))
            if (tuple(k), rho) == prev:
                break
        f = np.zeros((2, self.G)); f[0, k[0]] = 1; f[1, k[1]] = 1
        return k, rho, self.loglik(f, rho)

    def fit_m0(self):
        """One length per haplotype. Coordinate ascent from several starts (phase-weighted modes,
        the TRGT alleles and the two largest modes, each in both orders); the best is kept, so
        haplotype labels follow the phase evidence rather than the order of the search."""
        logE = np.log(np.maximum(self.E, 1e-300))
        if self.H == 1:
            k = int(np.argmax(logE.sum(0)))
            return [k], 0.5, float(logE[:, k].sum() + self.m.sum())
        starts = []
        ph = ~self.unph
        if ph.any():
            starts.append([int(np.argmax((self.w[ph, h:h + 1] * logE[ph]).sum(0))) for h in (0, 1)])
        anc = sorted(self.anchors)[:2]
        if len(anc) == 2:
            starts.append([self.idx(anc[0]), self.idx(anc[1])])
        pooled = logE.sum(0)
        k0 = int(np.argmax(pooled))
        resp = self.E / np.maximum(self.E.sum(1, keepdims=True), 1e-300)
        dens = resp.sum(0)
        far = np.abs(np.arange(self.G) - k0) > max(2, self.period)
        k1 = int(np.argmax(np.where(far, dens, -1))) if far.any() else k0
        starts += [[k0, k1], [k0, k0]]
        best = None
        for st in starts:
            for order in (st, st[::-1]):
                r = self._m0_from(order, 0.5)
                if best is None or r[2] > best[2]:
                    best = r
        return best

    # -------------------------------------------------- free f (NPMLE by EM)
    def fit_free(self, k, rho0, f0=None):
        """EM from f0, or from the M0 lengths (half a local bump at each, half uniform: EM cannot grow mass where
        it starts at zero)."""
        H = self.H
        if f0 is None:
            f = np.zeros((H, self.G))
            for h in range(H):
                bump = np.exp(-0.5 * ((self.grid - self.grid[k[h]]) / max(2.0 * self.period, 2.0)) ** 2)
                f[h] = 0.5 * bump / bump.sum() + 0.5 / self.G
        else:
            f = 0.98 * f0 + 0.02 / self.G
        rho, prev, conv, it = rho0, -np.inf, False, 0
        a = self.prior(rho)
        for it in range(1, self.p.iters + 1):
            q = f * self.s[None, :]
            q /= q.sum(1, keepdims=True)
            P = self.E @ q.T                                     # [n, H]
            tot = (a * P).sum(1)
            ll = float(np.sum(np.log(np.maximum(tot, 1e-300)) + self.m))
            if ll - prev < self.p.tol * max(1.0, abs(ll)):
                conv = True
                break
            prev = ll
            post = a * P / np.maximum(tot, 1e-300)[:, None]     # [n, H]
            if H == 2 and self.unph.any():
                rho = float(np.clip(post[self.unph, 0].mean(), 1e-3, 1 - 1e-3))
                a = self.prior(rho)
            for h in range(H):                                   # expected molecules per L, selection undone
                c = (self.E * (post[:, h] / np.maximum(P[:, h], 1e-300))[:, None]).sum(0) * q[h] / self.s
                if c.sum() > 0:
                    f[h] = c / c.sum()
        return dict(f=f, rho=rho, ll=self.loglik(f, rho), converged=conv, iters=it)

    # -------------------------------------------------- run + summaries
    def run(self):
        k0, rho0, ll0 = self.fit_m0()
        fit = self.fit_free(k0, rho0)
        fit["ll0"] = ll0
        rows, f, post = self.summarise(fit)
        if self.p.bootstrap:
            self.bootstrap(rows, fit)
        return rows, f, post

    def stats(self, f):
        """Per haplotype: (L0, U, p_exp, p_con, mad, mean_shift, n_atoms) on the deviation scale d = L - L0."""
        out = []
        for h in range(self.H):
            mode = int(self.grid[int(np.argmax(f[h]))])
            L0 = int(min(self.anchors, key=lambda a: abs(a - mode))) if self.anchors else mode
            d = self.grid - L0
            far = np.abs(d) >= self.delta
            out.append((L0, float(f[h][far].sum()), float(f[h][d >= self.delta].sum()),
                        float(f[h][d <= -self.delta].sum()), float((np.abs(d) * f[h]).sum()), float((d * f[h]).sum()),
                        int((f[h] >= 0.01).sum())))
        return out

    def bootstrap(self, rows, fit):
        """95% intervals of U and mad from read resamples, warm-started at the fit; haplotypes matched by L0."""
        rng = np.random.default_rng(self.p.seed)
        full = (self.E, self.m, self.w, self.unph, self.n)
        L0 = [r["L0"] for r in rows]
        U, M = [], []
        try:
            for _ in range(self.p.bootstrap):
                ii = rng.integers(0, full[4], full[4])
                self.E, self.m, self.w, self.unph, self.n = full[0][ii], full[1][ii], full[2][ii], full[3][ii], full[4]
                st = self.stats(self.fit_free(None, fit["rho"], fit["f"])["f"])
                if self.H == 2 and abs(st[0][0] - L0[1]) + abs(st[1][0] - L0[0]) < abs(st[0][0] - L0[0]) + abs(st[1][0] - L0[1]):
                    st = st[::-1]
                U.append([x[1] for x in st]); M.append([x[4] for x in st])
        finally:
            self.E, self.m, self.w, self.unph, self.n = full
        U, M = np.array(U), np.array(M)
        for h, r in enumerate(rows):
            r.update(U_lo=float(np.quantile(U[:, h], 0.025)), U_hi=float(np.quantile(U[:, h], 0.975)),
                     mad_lo=float(np.quantile(M[:, h], 0.025)), mad_hi=float(np.quantile(M[:, h], 0.975)))

    def summarise(self, fit):
        H, grid = self.H, self.grid
        f, rho = fit["f"], fit["rho"]
        a = self.prior(rho)
        q = f * self.s[None, :]
        q = q / q.sum(1, keepdims=True)
        P = self.E @ q.T
        tot = (a * P).sum(1)
        post = a * P / np.maximum(tot, 1e-300)[:, None]
        rows = []
        for h, (L0, U, pe, pc, mad, shift, atoms) in enumerate(self.stats(f)):
            far = np.abs(grid - L0) >= self.delta
            v = a[:, h] * (self.E @ (q[h] * far)) / np.maximum(tot, 1e-300)   # reads from the unstable part
            keep = f[h] >= 1e-4
            rows.append(dict(
                hap=h + 1, L0=L0, U=U, p_exp=pe, p_con=pc, mad=mad, mean_shift=shift,
                n_atoms=atoms, n_eff=float(post[:, h].sum()), n_phased_eff=float(post[~self.unph, h].sum()),
                n_var_eff=float(v.sum()), n_support=int((v > 0.5).sum()),
                unphased_var_frac=float(v[self.unph].sum() / v.sum()) if v.sum() > 1e-9 else np.nan,
                f_L=grid[keep].astype(int).tolist(), f_p=f[h][keep].round(5).tolist()))
        share = np.array([rho, 1 - rho]) if (H == 2 and self.unph.any()) else \
            np.array([r["n_eff"] for r in rows]) / max(sum(r["n_eff"] for r in rows), 1e-12)
        pooled = {f"pooled_{c}": float(sum(share[h] * rows[h][c] for h in range(H)))
                  for c in ("U", "p_exp", "p_con", "mad", "mean_shift")}
        L0s = [r["L0"] for r in rows]
        for h, r in enumerate(rows):
            if H == 1:
                r["hap_rank"] = "haploid"
            elif L0s[0] == L0s[1]:
                r["hap_rank"] = "tie"
            else:
                r["hap_rank"] = "short" if L0s[h] == min(L0s) else "long"
            r.update(pooled, share=float(share[h]), rho=float(rho), n_reads=self.n,
                     n_unphased=int(self.unph.sum()), ll_m0=fit["ll0"], ll=fit["ll"], d_ll=fit["ll"] - fit["ll0"],
                     converged=fit["converged"], iters=fit["iters"], delta=self.delta)
        return rows, f, post


def fit_locus(args):
    """Picklable worker: (locus_id, logl, grid, w1, H, period, read_lens, anchors, FitParams)."""
    lid, logl, grid, w1, H, period, read_lens, anchors, p = args
    s = selection_curve(grid, read_lens, p.flank) if p.selection else None
    try:
        rows, _, _ = LocusFit(logl, grid, w1, H, period, s, anchors, p).run()
        for r in rows:
            r["locus"] = lid
            r["status"] = "ok"
        return rows
    except Exception as e:                                   # one bad locus must not stop a genome run
        return [dict(locus=lid, hap=1, status=f"error: {type(e).__name__}: {e}")]
