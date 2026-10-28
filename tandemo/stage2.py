"""Stage 2: the distribution of true repeat lengths per locus and haplotype.

Inputs per locus: each read's Stage 1 curve log l_i(L) on an integer grid, its phase weight w1_i
(NaN without het evidence), and the selection curve s(L).

Per haplotype h the true-length distribution is

    parametric (default)   f_h(L) = pi_h delta(L - L0_h) + (1 - pi_h) g_h(L),
                           g_h = skew-normal(xi_h, omega_h, alpha_h) discretised on the grid
    nonparametric          the same point mass, g_h a free distribution on the grid fitted by EM with
                           smoothing (EMS)

In both, g_h has no mass within g_min_shift (default max(2, period) bp) of L0. Otherwise a narrow g
can sit on L0 and absorb ordinary +-1 bp sequencing noise, which makes pi unidentifiable.

Reads keep their haplotype by phase evidence, not by length:

    a_ih = (w1_i, 1 - w1_i)    when the read has het evidence
         = (rho, 1 - rho)       when it has none; rho is fitted

    p_i  = sum_h a_ih  sum_L l_i(L) f_h(L) s(L) / Z_h,        Z_h = sum_L f_h(L) s(L)

The log-likelihood sum_i log p_i is maximised over L0, pi, the shape of g and rho. s(L) corrects
for spanning selection: a molecule of length L is only seen when a read is long enough to cover it
plus both flanks (s(L) proportional to the sum over reads overlapping the locus of
max(0, R - L - 2 * flank)). f_h is therefore the distribution of molecules, not of kept reads.

M0 (pi = 1 for every haplotype) is fitted first; it gives L0 and the reference log-likelihood.
M1 starts from it. No p-values: the output reports the fitted distributions, the log-likelihood
gain, and the quantities below.

Per haplotype: L0, pi, var_weight = 1 - pi, g parameters and moments, p_exp = P(L >= L0 + Delta),
p_con = P(L <= L0 - Delta) with Delta = max(3, period), mean_abs_dev = E|L - L0|, the reads it
received (sum of posteriors), and unphased_var_frac: the share of its variable component carried
by reads without het evidence (their haplotype is decided by length; when this is high, trust the
locus-level pooled numbers over the per-haplotype split).
Per locus (repeated on both rows): the pooled numbers, weighted by the haplotype shares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm


@dataclass
class FitParams:
    shape: str = "parametric"         # or "nonparametric"
    min_reads: int = 3
    omega_min: float | None = None     # g width floor (bp); default max(1, period / 2)
    np_smooth: float | None = None     # nonparametric smoothing kernel sd (bp); default max(1, period / 2)
    np_iters: int = 500
    selection: bool = True
    flank: int = 128
    l0_refine: int = 2                 # refine L0 within +- this many periods after M1
    g_min_shift: int | None = None     # g has no mass within this many bp of L0; default max(2, period)


def selection_curve(grid: np.ndarray, read_lens: str | None, flank: int) -> np.ndarray:
    if not read_lens or read_lens == "." or read_lens != read_lens:
        return np.ones(len(grid))
    R = np.array([int(x) for x in str(read_lens).split(",") if x], dtype=float)
    s = np.clip(R[None, :] - grid[:, None] - 2 * flank, 0, None).sum(1)
    if s.max() <= 0:
        return np.ones(len(grid))
    return np.clip(s / s.max(), 1e-3, 1.0)


def sn_pmf(grid: np.ndarray, xi: float, omega: float, alpha: float, support=None) -> np.ndarray:
    """Skew-normal density at the grid points, restricted to `support`, normalised to sum 1."""
    z = (grid - xi) / omega
    lp = norm.logpdf(z) + norm.logcdf(alpha * z)
    if support is not None:
        lp = np.where(support, lp, -np.inf)
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def _gauss_smooth(v: np.ndarray, sd: float) -> np.ndarray:
    r = int(np.ceil(4 * sd))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sd) ** 2)
    k /= k.sum()
    return np.convolve(v, k, mode="same")


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
        self.omega_min = self.p.omega_min or max(1.0, self.period / 2)
        self.smooth = self.p.np_smooth or max(1.0, self.period / 2)
        self.min_shift = self.p.g_min_shift if self.p.g_min_shift is not None else max(2, self.period)

    def g_support(self, k: int) -> np.ndarray:
        """Grid points g may use: |L - L0| >= min_shift (0 = everywhere except L0's own point mass)."""
        sup = np.abs(self.grid - self.grid[k]) >= max(self.min_shift, 1)
        return sup if sup.any() else np.ones(self.G, bool)

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

    def outlier_lengths(self, k, rho, max_n: int = 4):
        """Most likely true lengths of the reads M0 explains worst (their curve peaks away from both
        L0s by at least the g exclusion), at most max_n distinct values."""
        a = self.prior(rho)
        pk = np.array([self.E[:, kk] for kk in k]).T          # [n, H] under the point masses
        fit0 = (a * pk).sum(1)
        peak = self.grid[np.argmax(self.E, 1)]
        far = np.all(np.abs(peak[:, None] - self.grid[k][None, :]) >= max(self.min_shift, 1), 1)
        order = np.argsort(fit0)
        out = []
        for i in order:
            if far[i] and fit0[i] < 0.01 and peak[i] not in out:
                out.append(int(peak[i]))
            if len(out) >= max_n:
                break
        return out

    # -------------------------------------------------- parametric M1
    def _unpack(self, th, k):
        H = self.H
        f = np.zeros((H, self.G))
        pis, shapes = [], []
        for h in range(H):
            lp, d, lw, al = th[4 * h:4 * h + 4]
            pi = expit(lp)
            xi, om = self.grid[k[h]] + d, self.omega_min + np.exp(lw)
            g = sn_pmf(self.grid, xi, om, al, self.g_support(k[h]))
            f[h] = (1 - pi) * g
            f[h, k[h]] += pi
            pis.append(pi); shapes.append((xi, om, al, g))
        rho = expit(th[4 * H]) if (H == 2 and self.unph.any()) else 0.5
        return f, pis, shapes, rho

    def fit_parametric(self, k, rho0):
        H, p = self.H, self.period
        span = float(self.grid[-1] - self.grid[0])
        # (location, width, skew, location is absolute?)
        inits = [(+2 * p, max(2.0 * p, 3.0), 3.0, False), (-2 * p, max(2.0 * p, 3.0), -3.0, False),
                 (0.0, max(4.0 * p, 6.0), 0.0, False), (0.0, max(span / 2, 6.0), 0.0, False)]
        # starts at the reads M0 explains worst: one isolated expanded read is otherwise out of reach
        for L in self.outlier_lengths(k, rho0):
            inits.append((float(L), max(1.0 * p, 2.0), 0.0, True))
        bounds = []
        for _ in range(H):
            bounds += [(-9, 9), (-span, span), (-6, np.log(max(span, 2.0))), (-10, 10)]
        has_rho = H == 2 and self.unph.any()
        if has_rho:
            bounds.append((-7, 7))
        best = None
        for d, om, al, absolute in inits:
            th = []
            for h in range(H):
                off = d - self.grid[k[h]] if absolute else d
                th += [np.log(0.95 / 0.05), off, np.log(max(om - self.omega_min, 0.05)), al]
            if has_rho:
                th.append(np.log(rho0 / (1 - rho0)))
            obj = lambda t, k=k: -self.loglik(self._unpack(t, k)[0], self._unpack(t, k)[3])
            r = minimize(obj, np.array(th, float), method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best[0].fun:
                best = (r, list(k))
        r, k = best
        th = r.x
        # refine L0 by a local scan, holding the shape fixed, then polish
        for _ in range(1):
            for h in range(H):
                lo = max(0, k[h] - self.p.l0_refine * p); hi = min(self.G - 1, k[h] + self.p.l0_refine * p)
                cands = range(lo, hi + 1)
                vals = []
                for c in cands:
                    kk = list(k); kk[h] = c
                    f, _, _, rho = self._unpack(th, kk)
                    vals.append(self.loglik(f, rho))
                k[h] = list(cands)[int(np.argmax(vals))]
            obj = lambda t: -self.loglik(self._unpack(t, k)[0], self._unpack(t, k)[3])
            r2 = minimize(obj, th, method="L-BFGS-B", bounds=bounds)
            if r2.fun <= r.fun:
                r, th = r2, r2.x
        f, pis, shapes, rho = self._unpack(th, k)
        at_bound = any(abs(th[4 * h]) >= 8.99 or abs(th[4 * h + 3]) >= 9.99 for h in range(H))
        return dict(k=k, f=f, pis=pis, shapes=shapes, rho=rho, ll=-float(r.fun), converged=bool(r.success),
                    at_boundary=at_bound)

    # -------------------------------------------------- nonparametric M1 (EM with smoothing)
    def fit_nonparametric(self, k, rho0):
        H = self.H
        f = np.zeros((H, self.G))
        g = np.zeros((H, self.G))
        for h in range(H):
            sup = self.g_support(k[h])
            local = _gauss_smooth(np.eye(1, self.G, k[h]).ravel(), max(3.0 * self.period, 3.0)) * sup
            # half local, half uniform over the support: EM cannot grow mass where it starts at zero
            g[h] = 0.5 * local / max(local.sum(), 1e-300) + 0.5 * sup / sup.sum()
            g[h] = g[h] / g[h].sum() if g[h].sum() > 0 else self.g_support(k[h]) / self.g_support(k[h]).sum()
        pis = np.full(H, 0.9)
        rho = rho0
        prev = -np.inf
        conv = False
        for it in range(self.p.np_iters):
            for h in range(H):
                f[h] = (1 - pis[h]) * g[h]; f[h, k[h]] += pis[h]
            a = self.prior(rho)
            q = f * self.s[None, :]
            q /= q.sum(1, keepdims=True)
            P = self.E @ q.T                                     # [n, H]
            tot = (a * P).sum(1)
            ll = float(np.sum(np.log(np.maximum(tot, 1e-300)) + self.m))
            if ll - prev < 1e-7 * max(1.0, abs(ll)):
                conv = True
                break
            prev = ll
            post_h = a * P / np.maximum(tot, 1e-300)[:, None]  # [n, H]
            if H == 2 and self.unph.any():
                rho = float(np.clip(post_h[self.unph, 0].mean(), 1e-3, 1 - 1e-3))
            for h in range(H):
                # expected counts over L for haplotype h, then undo selection
                R = self.E * q[h][None, :] / np.maximum(P[:, h:h + 1], 1e-300) * post_h[:, h:h + 1]
                c = R.sum(0) / self.s
                c /= c.sum()
                # the point mass takes L0 and the excluded neighbours (noise around L0)
                pis[h] = float(np.clip(c[~self.g_support(k[h])].sum(), 1e-6, 1 - 1e-9))
                sup = self.g_support(k[h])
                gh = _gauss_smooth(c * sup, self.smooth) * sup
                g[h] = gh / gh.sum() if gh.sum() > 0 else g[h]
        for h in range(H):
            f[h] = (1 - pis[h]) * g[h]; f[h, k[h]] += pis[h]
        shapes = [(np.nan, np.nan, np.nan, g[h]) for h in range(H)]
        return dict(k=list(k), f=f, pis=list(pis), shapes=shapes, rho=rho, ll=self.loglik(f, rho),
                    converged=conv, at_boundary=False)

    # -------------------------------------------------- run + summaries
    def run(self):
        k0, rho0, ll0 = self.fit_m0()
        if self.p.shape == "nonparametric":
            fit = self.fit_nonparametric(k0, rho0)
        else:
            fit = self.fit_parametric(k0, rho0)
        if fit["ll"] < ll0:          # M1 contains M0; fall back to it if the optimiser did worse
            f = np.zeros((self.H, self.G))
            for h in range(self.H):
                f[h, k0[h]] = 1
            fit = dict(k=list(k0), f=f, pis=[1.0] * self.H,
                       shapes=[(np.nan, np.nan, np.nan, np.zeros(self.G))] * self.H, rho=rho0, ll=ll0,
                       converged=True, at_boundary=True)
        fit["ll0"] = ll0
        return self.summarise(fit)

    def summarise(self, fit):
        H, grid = self.H, self.grid
        f, k, rho = fit["f"], fit["k"], fit["rho"]
        a = self.prior(rho)
        q = f * self.s[None, :]
        Z = q.sum(1, keepdims=True)
        P = self.E @ (q / Z).T
        tot = (a * P).sum(1)
        post = a * P / np.maximum(tot, 1e-300)[:, None]
        delta = max(3, self.period)
        rows = []
        for h in range(H):
            pi = fit["pis"][h]
            xi, om, al, g = fit["shapes"][h]
            L0 = int(grid[k[h]])
            gv = (g * self.s) / max((f[h] * self.s).sum(), 1e-300) * (1 - pi)
            var_lik = self.E @ gv                                # variable part of read lik
            v = a[:, h] * var_lik / np.maximum(tot, 1e-300)
            gm = float((grid * g).sum()) if g.sum() > 0 else np.nan
            gsd = float(np.sqrt(max((g * (grid - gm) ** 2).sum(), 0))) if g.sum() > 0 else np.nan
            rows.append(dict(
                hap=h + 1, L0=L0, pi=pi, var_weight=1 - pi, g_xi=xi, g_omega=om, g_alpha=al, g_mean=gm, g_sd=gsd,
                p_exp=float(f[h][grid >= L0 + delta].sum()), p_con=float(f[h][grid <= L0 - delta].sum()),
                mean_abs_dev=float((np.abs(grid - L0) * f[h]).sum()),
                n_eff=float(post[:, h].sum()), n_phased_eff=float(post[~self.unph, h].sum()),
                n_var_eff=float(v.sum()), n_support=int((v > 0.5).sum()),
                unphased_var_frac=float(v[self.unph].sum() / v.sum()) if v.sum() > 1e-9 else np.nan))
        share = np.array([rho, 1 - rho]) if (H == 2 and self.unph.any()) else \
            np.array([r["n_eff"] for r in rows]) / max(sum(r["n_eff"] for r in rows), 1e-12)
        pooled = dict(pooled_var_weight=float(sum(share[h] * rows[h]["var_weight"] for h in range(H))),
                      pooled_p_exp=float(sum(share[h] * rows[h]["p_exp"] for h in range(H))),
                      pooled_p_con=float(sum(share[h] * rows[h]["p_con"] for h in range(H))))
        L0s = [r["L0"] for r in rows]
        for h, r in enumerate(rows):
            if H == 1:
                r["hap_rank"] = "haploid"
            elif L0s[0] == L0s[1]:
                r["hap_rank"] = "tie"
            else:
                r["hap_rank"] = "short" if L0s[h] == min(L0s) else "long"
            r.update(pooled, share=float(share[h]), rho=float(rho), n_reads=self.n,
                     n_unphased=int(self.unph.sum()), ll_m0=fit["ll0"], ll_m1=fit["ll"],
                     d_ll=fit["ll"] - fit["ll0"], converged=fit["converged"], at_boundary=fit["at_boundary"],
                     shape=self.p.shape, delta=delta)
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
