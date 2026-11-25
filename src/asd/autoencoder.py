"""Train the autoencoders on normal sounds; score clips; build error maps."""
import numpy as np
import torch
from torch import nn

from .config import CONTEXT, N_MELS, PATCH, PATCH_HOP, SEED
from .models import ConvAE, DenseAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def patch_starts(n_frames):
    s = list(range(0, n_frames - PATCH + 1, PATCH_HOP))
    if s[-1] != n_frames - PATCH:
        s.append(n_frames - PATCH)  # cover the last frames too
    return s


class TrainingFault(RuntimeError):
    """Training produced an implausible gradient or loss: discard the run."""


def check_epoch(ep, norms, loss, prev_loss):
    """Guard against silent hardware faults (seen on an overheating GPU)."""
    if not (torch.isfinite(norms).all() and np.isfinite(loss)):
        raise TrainingFault(f"epoch {ep}: non-finite gradient or loss")
    if ep == 1:  # large gradients are legitimate at the very start
        return
    ratio = float(norms.max() / norms.median())
    # Healthy dense runs on a T4 reach ~150x (rare loud batches); the GPU fault gave ~1e8x.
    if ratio > 1e4:
        raise TrainingFault(f"epoch {ep}: gradient spike, {ratio:.0f}x the median, loss {loss:.4f}")
    if loss > 1.5 * prev_loss:
        raise TrainingFault(f"epoch {ep}: loss jumped {prev_loss:.3f} -> {loss:.3f}")


class AEDetector:
    """kind='dense': 5-frame vectors in dB (official baseline).
    kind='conv': 128x64 patches, standardised per mel band on training normals."""

    def __init__(self, kind):
        self.kind = kind
        self.net = (DenseAE() if kind == "dense" else ConvAE()).to(DEVICE)
        self.mean = torch.zeros(N_MELS, 1)
        self.std = torch.ones(N_MELS, 1)
        self.history = []

    def _norm(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def fit(self, x, x_val, epochs=None, log=print):
        dense = self.kind == "dense"
        torch.manual_seed(SEED)
        self.net = type(self.net)().to(DEVICE)  # reproducible initialisation
        if not dense:
            self.mean = torch.tensor(x.mean((0, 2)))[:, None]
            self.std = torch.tensor(x.std((0, 2)))[:, None].clamp_min(1e-3)
        win, bs = (CONTEXT, 512) if dense else (PATCH, 64)
        epochs = epochs or (100 if dense else 80)
        data = self._norm(torch.from_numpy(x).to(DEVICE))
        n, n_win = len(data), data.shape[2] - win + 1
        # dense: every window each epoch; conv: on average 8 random crops per clip
        per_epoch = n * (n_win if dense else 8)
        steps = -(-per_epoch // bs)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        sched = None if dense else torch.optim.lr_scheduler.OneCycleLR(
            opt, 2e-3, total_steps=epochs * steps)
        offs = torch.arange(win, device=DEVICE)
        params, prev_loss = list(self.net.parameters()), None
        for ep in range(1, epochs + 1):
            self.net.train()
            norms = []
            total = torch.zeros((), device=DEVICE)
            for idx in torch.randperm(n * n_win, device=DEVICE)[:per_epoch].split(bs):
                c, t = idx // n_win, idx % n_win
                b = data[c[:, None], :, t[:, None] + offs]  # [B, win, mels]
                b = b.flatten(1) if dense else b.transpose(1, 2)[:, None]
                loss = nn.functional.mse_loss(self.net(b), b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norms.append(nn.utils.get_total_norm([p.grad for p in params]))
                opt.step()
                if sched:
                    sched.step()
                total += loss.detach() * len(idx)
            norms = torch.stack(norms).cpu()
            train_loss = float(total) / per_epoch
            check_epoch(ep, norms, train_loss, prev_loss)
            prev_loss = train_loss
            rec = {"epoch": ep, "train": train_loss, "grad_max": float(norms.max()),
                   "grad_median": float(norms.median())}
            if ep % 10 == 0 or ep == 1:
                rec["val"] = float(self.score(x_val).mean())
                log(rec)
            self.history.append(rec)
        return self

    @torch.no_grad()
    def error_map(self, x):
        """[mels, frames] dB -> squared error of the same shape (overlaps averaged)."""
        self.net.eval()
        d = self._norm(torch.as_tensor(x, device=DEVICE))
        f = d.shape[1]
        dense = self.kind == "dense"
        win = CONTEXT if dense else PATCH
        starts = list(range(f - win + 1)) if dense else patch_starts(f)
        b = torch.stack([d[:, s:s + win] for s in starts])  # [P, mels, win]
        if dense:
            rec = self.net(b.transpose(1, 2).flatten(1)).view(len(b), win, -1).transpose(1, 2)
        else:
            rec = self.net(b[:, None])[:, 0]
        err, cnt = torch.zeros_like(d), torch.zeros(f, device=d.device)
        for e, s in zip((rec - b) ** 2, starts):
            err[:, s:s + win] += e
            cnt[s:s + win] += 1
        return (err / cnt).cpu().numpy()

    @torch.no_grad()
    def score(self, x):
        """Mean reconstruction error per clip."""
        self.net.eval()
        out = []
        for part in torch.from_numpy(x).split(32):
            d = self._norm(part.to(DEVICE))
            if self.kind == "dense":
                w = d.unfold(2, CONTEXT, 1).permute(0, 2, 3, 1).flatten(2)  # [B, T, 5*mels]
                rec = self.net(w.flatten(0, 1)).view_as(w)
                out.append(((rec - w) ** 2).mean((1, 2)))
            else:
                w = torch.stack([d[:, :, s:s + PATCH] for s in patch_starts(d.shape[2])], 1)
                rec = self.net(w.flatten(0, 1)[:, None])[:, 0].view_as(w)
                out.append(((rec - w) ** 2).mean((1, 2, 3)))
        return torch.cat(out).cpu().numpy().astype(np.float64)

    def save(self, path):
        torch.save({"kind": self.kind, "net": self.net.state_dict(),
                    "mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path):
        s = torch.load(path, map_location=DEVICE, weights_only=True)
        d = cls(s["kind"])
        d.net.load_state_dict(s["net"])
        d.mean, d.std = s["mean"].cpu(), s["std"].cpu()
        return d
