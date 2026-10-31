"""Network 1: the read-generation likelihood p(Y | L, x), learned from a sample without instability.

Architecture
  encoder  x (read + locus features, standardised)  ->  h (128)            once per read
  head     [h, l(L)]                                 ->  logits over DeltaBins (Y - L) + tail parameters
  output   a normalised distribution of the observed length Y for each candidate true length L:
           log_softmax over 2D+1 per-bp bins (|Y - L| <= D) and two tail bins (the mass beyond -D and
           beyond +D). How that mass falls off with distance depends on the tail law:
             pareto   (default) k = |Y - L| - D >= 1 follows a discretised generalised Pareto per side,
                      P(k) = S(k-1) - S(k),  S(x) = (1 + xi x / sigma)^(-1/xi),
                      with sigma > 0 and xi >= 0 predicted per (read, L): a learned decay, no upper limit;
             uniform  (models saved before the Pareto tails) the mass spread evenly over D < |Y - L| <= T.

Training (stable sample, TRGT alleles as truth)
  Each read could come from either germline allele. With w1 its phase weight (het SNV evidence; 0.5
  without evidence), the loss is the negative log of the mixture

      J = - sum_i log [ w1_i p(Y_i | L1, x_i) + (1 - w1_i) p(Y_i | L2, x_i) ]

  so reads never need a hard allele assignment, and homozygous loci reduce to one term.
  Validation is held out by chromosome (default chr20, chr21, chr22) so reads of one locus never sit
  on both sides of the split. Early stopping on validation loss.

Inference
  log_curves(reads, grid): the encoder runs once per read, the head once per (read, L). The result is
  the read x grid matrix of log-likelihoods Stage 2 works on.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ALL_FEATURES, FIRST_MODEL_NAMES, METH_NAMES, X_NAMES, X_NAMES_NO_METH, DeltaBins, Scaler, raw_x
from .noise import NoiseModel


TAILS = ("pareto", "uniform")
SIGMA_MIN, XI_MIN = 0.5, 1e-3          # tail scale (bp) and shape floors, for numerical stability


def gpd_logpmf(k: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """log P(K = k) of a discretised generalised Pareto on k = 1, 2, ...: P(k) = S(k-1) - S(k) with
    survival S(x) = (1 + xi x / sigma)^(-1/xi). Sums to 1 over k >= 1; xi -> 0 is a geometric tail."""
    k = k.clamp(min=1).to(sigma.dtype)
    log_s_prev = -torch.log1p(xi * (k - 1) / sigma) / xi
    log_s = -torch.log1p(xi * k / sigma) / xi
    return log_s_prev + torch.log(-torch.expm1(log_s - log_s_prev).clamp(max=-1e-30))


class Net1(nn.Module):
    def __init__(self, nx: int, nbins: int, hidden: int = 128, nl: int = 3, dropout: float = 0.1,
                 tail: str = "pareto"):
        super().__init__()
        if tail not in TAILS:
            raise ValueError(f"tail must be one of {TAILS}")
        self.nbins, self.tail = nbins, tail
        extra = 4 if tail == "pareto" else 0                # log-scale and shape, left and right
        self.enc = nn.Sequential(nn.Linear(nx, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden + nl, hidden), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, nbins + extra))

    def dist(self, h: torch.Tensor, lf: torch.Tensor):
        """h [B, H], lf [B, K, 3] -> (log bin probabilities [B, K, nbins], tail parameters [B, K, 4] or None:
        sigma_left, xi_left, sigma_right, xi_right)."""
        hk = h[:, None, :].expand(-1, lf.shape[1], -1)
        out = self.head(torch.cat([hk, lf], -1))
        lp = F.log_softmax(out[..., :self.nbins], -1)
        if self.tail != "pareto":
            return lp, None
        raw = out[..., self.nbins:]
        sig = F.softplus(raw[..., 0::2]) + SIGMA_MIN
        xi = F.softplus(raw[..., 1::2]) + XI_MIN
        return lp, torch.stack([sig[..., 0], xi[..., 0], sig[..., 1], xi[..., 1]], -1)

    def logp(self, h: torch.Tensor, lf: torch.Tensor) -> torch.Tensor:
        """Log bin probabilities [B, K, nbins] (the tail bins hold each side's total mass)."""
        return self.dist(h, lf)[0]

    def delta_logp(self, lp: torch.Tensor, tp, delta: torch.Tensor, bins: DeltaBins) -> torch.Tensor:
        """log p(Delta) for integer Delta = Y - L. lp [B, K, nbins], tp [B, K, 4] or None, delta [B, K] or
        [B, K, M] (M deltas per (read, L)) -> same shape as delta."""
        D = bins.D
        extra = delta.dim() > lp.dim() - 1
        if extra:
            lp, tp = lp[..., None, :], (tp[..., None, :] if tp is not None else None)
        idx = (delta.clamp(-D, D) + D + 1).long()
        core = lp.expand(*delta.shape, lp.shape[-1]).gather(-1, idx[..., None]).squeeze(-1)
        left_mass, right_mass = lp[..., 0], lp[..., -1]
        if tp is None:                                      # uniform tails up to T, floor beyond
            w = -math.log(bins.T - D)
            out = torch.where(delta < -D, left_mass + w, torch.where(delta > D, right_mass + w, core))
            return torch.where(delta.abs() <= bins.T, out, torch.full_like(out, bins.floor))
        left = left_mass + gpd_logpmf(-D - delta, tp[..., 0], tp[..., 1])
        right = right_mass + gpd_logpmf(delta - D, tp[..., 2], tp[..., 3])
        return torch.where(delta < -D, left, torch.where(delta > D, right, core))

    def forward(self, x, lf):
        return self.logp(self.enc(x), lf)


def _device(name: str | None = None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tensors(df: pd.DataFrame, scaler: Scaler, bins: DeltaBins, names):
    X = scaler.x(raw_x(df, names))
    La = np.stack([df["L1"].to_numpy(), df["L2"].to_numpy()], 1)
    lf = scaler.l_feats(La, df["ref_len"].to_numpy()[:, None], df["period"].to_numpy()[:, None])
    delta = df["Y"].to_numpy()[:, None] - La                 # [n, 2] integer Y - L per allele
    w = df["w1_train"].to_numpy()
    logw = np.log(np.clip(np.stack([w, 1 - w], 1), 1e-12, 1))
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
    return [t(X), t(lf), t(delta, torch.long), t(logw)]


def mixture_nll(net: Net1, X, lf, delta, logw, bins: DeltaBins) -> torch.Tensor:
    lp, tp = net.dist(net.enc(X), lf)                        # [B, 2, nbins], [B, 2, 4]
    ll = net.delta_logp(lp, tp, delta, bins)                 # [B, 2]
    return -torch.logsumexp(ll + logw, -1).mean()


def train(df: pd.DataFrame, out: str, bins: DeltaBins | None = None, val_chroms=("chr20", "chr21", "chr22"),
          epochs: int = 40, batch: int = 4096, lr: float = 1e-3, weight_decay: float = 1e-5,
          hidden: int = 128, patience: int = 10, device: str | None = None, seed: int = 0, tail: str = "pareto",
          methylation: bool = True, meth_mask: float = 0.4, log=print):
    """methylation: use the 5mC features; meth_mask: share of training reads per batch whose methylation inputs
    are set to missing, so the model also predicts well for reads without 5mC calls."""
    torch.manual_seed(seed); np.random.seed(seed)
    bins = bins or DeltaBins(5 if tail == "pareto" else 50)
    is_val = df["chrom"].isin(val_chroms).to_numpy()
    if is_val.sum() == 0 or is_val.all():           # no such chromosomes: hold out 10% of loci
        h = pd.util.hash_pandas_object(df["locus"], index=False).to_numpy()
        is_val = (h % 10) == 0
    tr, va = df[~is_val], df[is_val]
    names = X_NAMES if methylation else X_NAMES_NO_METH
    scaler = Scaler().fit(raw_x(tr, names), np.concatenate([tr["L1"].to_numpy(), tr["L2"].to_numpy()]))
    dev = _device(device)
    T_tr = _tensors(tr, scaler, bins, names)
    T_va = _tensors(va, scaler, bins, names)
    n, n_val = len(tr), len(va)
    meth_frac = float(np.mean(raw_x(tr, ['mC_avail'])[:, 0])) if 'mC_avail' in names else 0.0
    del tr, va                                            # the tensors hold all training needs from here
    # GPU-resident training data when it fits comfortably: batches are then index lookups on the device
    nbytes = sum(t.numel() * t.element_size() for t in T_tr + T_va)
    resident = dev.type == "cuda" and nbytes < 0.6 * torch.cuda.get_device_properties(dev).total_memory
    if resident:
        T_tr = [t.to(dev) for t in T_tr]
        T_va = [t.to(dev) for t in T_va]
    log(f"training data {nbytes / 1e9:.1f} GB, {'on the GPU' if resident else 'in host memory'}")
    net = Net1(len(names), bins.n, hidden, tail=tail).to(dev)
    # the standardised "missing" value of each methylation input: the training median, and mC_avail = 0
    m_idx = [names.index(m) for m in METH_NAMES if m in names]
    missing = None
    if m_idx and meth_mask > 0:
        raw_missing = np.array([0.0 if names[i] == "mC_avail" else scaler.med[i] for i in m_idx])
        missing = torch.as_tensor((raw_missing - scaler.mu[m_idx]) / scaler.sd[m_idx], dtype=torch.float32,
                                  device=dev)
        m_cols = torch.as_tensor(m_idx, device=dev)
        log(f"methylation inputs {', '.join(names[i] for i in m_idx)}: masked for {meth_mask:.0%} of reads per batch; "
            f"{meth_frac:.1%} of training reads carry 5mC calls")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    best, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev if resident else "cpu")
        tot = 0.0
        for k in range(0, n, batch):
            b = perm[k:k + batch]
            Xb, *rest = [t[b] if resident else t[b].to(dev) for t in T_tr]
            if resident:
                Xb = Xb.clone()                           # the masking below writes into the batch
            if missing is not None:                    # methylation optional: hide it for a share of reads
                rows = (torch.rand(len(b), device=dev) < meth_mask).nonzero(as_tuple=True)[0]
                if len(rows):
                    Xb[rows[:, None], m_cols[None, :]] = missing
            loss = mixture_nll(net, Xb, *rest, bins)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        vl = evaluate(net, T_va, bins, dev, batch)
        sched.step(vl)
        log(f"epoch {ep + 1}: train nll {tot / max(n, 1):.4f}  val nll {vl:.4f}  lr {opt.param_groups[0]['lr']:.2g}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state or net.state_dict())
    model = TorchNoise(net.cpu(), scaler, bins, dict(hidden=hidden, val_nll=best, n_train=n, n_val=n_val,
                                                     tail=tail, meth_mask=meth_mask if m_idx else 0.0),
                       features=names)
    model.save(out)
    return model


@torch.no_grad()
def evaluate(net, T, bins, dev, batch=8192) -> float:
    net.eval()
    n = len(T[0])
    if n == 0:
        return float("nan")
    tot = 0.0
    for k in range(0, n, batch):
        tot += mixture_nll(net, *[t[k:k + batch].to(dev) for t in T], bins).item() * len(T[0][k:k + batch])
    return tot / n


class TorchNoise(NoiseModel):
    def __init__(self, net: Net1, scaler: Scaler, bins: DeltaBins, meta: dict | None = None, device=None,
                 features=None):
        self.net, self.scaler, self.bins, self.meta = net.eval(), scaler, bins, meta or {}
        self.features = list(features or FIRST_MODEL_NAMES)        # the inputs this model was trained on
        self.dev = _device(device)
        self.net.to(self.dev)

    @torch.no_grad()
    def log_curves(self, reads: pd.DataFrame, grid: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        X = torch.as_tensor(self.scaler.x(raw_x(reads, self.features)), device=self.dev)
        h = self.net.enc(X)                                                   # [n, H]
        grid = np.asarray(grid)
        lf = self.scaler.l_feats(grid[None, :], reads["ref_len"].to_numpy()[:, None],
                                 reads["period"].to_numpy()[:, None])         # [n, G, 3]
        delta = reads["Y"].to_numpy()[:, None] - grid[None, :]               # [n, G]
        n, G = delta.shape
        out = np.empty((n, G), dtype=np.float32)
        step = max(1, chunk // max(n, 1))
        for g0 in range(0, G, step):
            g1 = min(G, g0 + step)
            lp, tp = self.net.dist(h, torch.as_tensor(lf[:, g0:g1], device=self.dev))
            d = torch.as_tensor(delta[:, g0:g1], device=self.dev)
            out[:, g0:g1] = self.net.delta_logp(lp, tp, d, self.bins).cpu().numpy()
        return out

    def save(self, path: str):
        torch.save(dict(kind="torch", state=self.net.state_dict(), scaler=self.scaler.state(),
                        bins=self.bins.state(), meta=self.meta, features=self.features, tail=self.net.tail), path)

    @classmethod
    def load(cls, path: str, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        feats = ck.get("features") or FIRST_MODEL_NAMES
        unknown = [f for f in feats if f not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"model uses features this version does not know: {unknown}")
        bins = DeltaBins(**ck["bins"])
        net = Net1(len(feats), bins.n, ck["meta"].get("hidden", 128), tail=ck.get("tail", "uniform"))
        net.load_state_dict(ck["state"])
        return cls(net, Scaler.from_state(ck["scaler"]), bins, ck["meta"], device, features=feats)
