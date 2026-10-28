"""Network 1: the read-generation likelihood p(Y | L, x), learned from a sample without instability.

Architecture
  encoder  x (read + locus features, standardised)  ->  h (128)            once per read
  head     [h, l(L)]                                 ->  logits over DeltaBins (Y - L)
  output   log_softmax over the bins: a normalised distribution of the observed length Y for
           each candidate true length L; tail bins are spread uniformly over D < |Y - L| <= T.

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

from .features import X_NAMES, DeltaBins, Scaler, raw_x
from .noise import NoiseModel


class Net1(nn.Module):
    def __init__(self, nx: int, nbins: int, hidden: int = 128, nl: int = 3, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nx, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden + nl, hidden), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, nbins))

    def logp(self, h: torch.Tensor, lf: torch.Tensor) -> torch.Tensor:
        """h [B, H], lf [B, K, 3] -> log probabilities [B, K, nbins]."""
        hk = h[:, None, :].expand(-1, lf.shape[1], -1)
        return F.log_softmax(self.head(torch.cat([hk, lf], -1)), -1)

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


def _tensors(df: pd.DataFrame, scaler: Scaler, bins: DeltaBins):
    X = scaler.x(raw_x(df))
    La = np.stack([df["L1"].to_numpy(), df["L2"].to_numpy()], 1)
    lf = scaler.l_feats(La, df["ref_len"].to_numpy()[:, None], df["period"].to_numpy()[:, None])
    idx, corr, valid = bins.lookup(df["Y"].to_numpy()[:, None] - La)
    w = df["w1_train"].to_numpy()
    logw = np.log(np.clip(np.stack([w, 1 - w], 1), 1e-12, 1))
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)
    return [t(X), t(lf), t(idx, torch.long), t(corr), t(valid, torch.bool), t(logw)]


def mixture_nll(net: Net1, X, lf, idx, corr, valid, logw, floor: float) -> torch.Tensor:
    lp = net(X, lf)                                          # [B, 2, nbins]
    ll = lp.gather(-1, idx[..., None]).squeeze(-1) + corr    # [B, 2]
    ll = torch.where(valid, ll, torch.full_like(ll, floor))
    return -torch.logsumexp(ll + logw, -1).mean()


def train(df: pd.DataFrame, out: str, bins: DeltaBins | None = None, val_chroms=("chr20", "chr21", "chr22"),
          epochs: int = 40, batch: int = 4096, lr: float = 1e-3, weight_decay: float = 1e-5,
          hidden: int = 128, patience: int = 4, device: str | None = None, seed: int = 0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    bins = bins or DeltaBins()
    is_val = df["chrom"].isin(val_chroms).to_numpy()
    if is_val.sum() == 0 or is_val.all():           # no such chromosomes: hold out 10% of loci
        h = pd.util.hash_pandas_object(df["locus"], index=False).to_numpy()
        is_val = (h % 10) == 0
    tr, va = df[~is_val], df[is_val]
    scaler = Scaler().fit(raw_x(tr), np.concatenate([tr["L1"].to_numpy(), tr["L2"].to_numpy()]))
    dev = _device(device)
    T_tr = _tensors(tr, scaler, bins)
    T_va = _tensors(va, scaler, bins)
    net = Net1(len(X_NAMES), bins.n, hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    best, best_state, bad = math.inf, None, 0
    n = len(tr)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            b = perm[k:k + batch]
            loss = mixture_nll(net, *[t[b].to(dev) for t in T_tr], bins.floor)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        vl = evaluate(net, T_va, bins, dev, batch)
        sched.step(vl)
        log(f"epoch {ep + 1}: train nll {tot / max(n, 1):.4f}  val nll {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state or net.state_dict())
    model = TorchNoise(net.cpu(), scaler, bins, dict(hidden=hidden, val_nll=best, n_train=n, n_val=len(va)))
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
        tot += mixture_nll(net, *[t[k:k + batch].to(dev) for t in T], bins.floor).item() * len(T[0][k:k + batch])
    return tot / n


class TorchNoise(NoiseModel):
    def __init__(self, net: Net1, scaler: Scaler, bins: DeltaBins, meta: dict | None = None, device=None):
        self.net, self.scaler, self.bins, self.meta = net.eval(), scaler, bins, meta or {}
        self.dev = _device(device)
        self.net.to(self.dev)

    @torch.no_grad()
    def log_curves(self, reads: pd.DataFrame, grid: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        X = torch.as_tensor(self.scaler.x(raw_x(reads)), device=self.dev)
        h = self.net.enc(X)                                                   # [n, H]
        grid = np.asarray(grid)
        lf = self.scaler.l_feats(grid[None, :], reads["ref_len"].to_numpy()[:, None],
                                 reads["period"].to_numpy()[:, None])         # [n, G, 3]
        idx, corr, valid = self.bins.lookup(reads["Y"].to_numpy()[:, None] - grid[None, :])
        n, G = idx.shape
        out = np.empty((n, G), dtype=np.float32)
        step = max(1, chunk // max(n, 1))
        for g0 in range(0, G, step):
            g1 = min(G, g0 + step)
            lp = self.net.logp(h, torch.as_tensor(lf[:, g0:g1], device=self.dev))   # [n, g, nbins]
            ii = torch.as_tensor(idx[:, g0:g1], device=self.dev)
            out[:, g0:g1] = lp.gather(-1, ii[..., None]).squeeze(-1).cpu().numpy()
        out = out + corr
        return np.where(valid, out, self.bins.floor)

    def save(self, path: str):
        torch.save(dict(kind="torch", state=self.net.state_dict(), scaler=self.scaler.state(),
                        bins=self.bins.state(), meta=self.meta, features=X_NAMES), path)

    @classmethod
    def load(cls, path: str, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("features") != X_NAMES:
            raise ValueError("model was trained with a different feature set")
        bins = DeltaBins(**ck["bins"])
        net = Net1(len(X_NAMES), bins.n, ck["meta"].get("hidden", 128))
        net.load_state_dict(ck["state"])
        return cls(net, Scaler.from_state(ck["scaler"]), bins, ck["meta"], device)
