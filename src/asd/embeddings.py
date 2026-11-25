"""Clip embeddings from the pre-trained PANNs CNN14 (16 kHz), scored by kNN or GMM."""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .config import CACHE, DATA, SEED
from .features import load_audio

CHECKPOINT = DATA / "panns" / "Cnn14_16k.pth"


class ConvBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.conv1 = nn.Conv2d(a, b, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(b, b, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(b), nn.BatchNorm2d(b)

    def forward(self, x, pool):
        x = F.relu_(self.bn2(self.conv2(F.relu_(self.bn1(self.conv1(x))))))
        return F.avg_pool2d(x, pool)


class Cnn14(nn.Module):
    """CNN14 trunk; attribute names match the official checkpoint."""

    def __init__(self):
        super().__init__()
        self.register_buffer("melW", torch.zeros(257, 64))
        self.register_buffer("window", torch.hann_window(512))
        self.bn0 = nn.BatchNorm2d(64)
        ch = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", ConvBlock(ch[i], ch[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)

    def forward(self, wav):
        spec = torch.stft(wav, 512, 160, window=self.window, return_complex=True).abs() ** 2
        x = 10 * torch.log10((spec.transpose(1, 2) @ self.melW).clamp_min(1e-10))  # [B, T, 64]
        x = self.bn0(x[:, None].transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            x = getattr(self, f"conv_block{i}")(x, 1 if i == 6 else 2)
        x = x.mean(3)
        x = x.amax(2) + x.mean(2)
        return F.relu_(self.fc1(x))


def load_cnn14(device="cpu"):
    # The file stores numpy metadata: allow only those types, keep the safe loader.
    allowed = [(np.zeros(0).__reduce__()[0], "numpy.core.multiarray._reconstruct"),
               np.ndarray, np.dtype, *{type(np.dtype(t)) for t in ("f4", "f8", "i4", "i8")}]
    with torch.serialization.safe_globals(allowed):
        sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    net = Cnn14()
    own = {k: sd[k] for k in net.state_dict() if k in sd}
    own["melW"] = sd["logmel_extractor.melW"]
    own["window"] = net.window
    net.load_state_dict(own)  # strict: every trunk weight must come from the checkpoint
    return net.eval().to(device)


def cached_embeddings(name, df):
    npy = CACHE / f"{name}_panns.npy"
    if npy.exists():
        e = np.load(npy)
        assert len(e) == len(df)
        return e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, out = load_cnn14(device), []
    if device == "cuda":  # the GPU must agree with the CPU before we trust it
        wav = torch.from_numpy(np.stack([load_audio(p) for p in df.path[:4]]))
        with torch.no_grad():
            a, b = load_cnn14("cpu")(wav), net(wav.cuda()).cpu()
        assert torch.allclose(a, b, rtol=1e-2, atol=1e-2), "GPU and CPU embeddings differ"
    with torch.no_grad():
        for i in tqdm(range(0, len(df), 16), desc=name, mininterval=20):
            wav = np.stack([load_audio(p) for p in df.path[i:i + 16]])
            out.append(net(torch.from_numpy(wav).to(device)).cpu().numpy())
    e = np.concatenate(out)
    np.save(npy, e)
    return e


class KNNDetector:
    """Mean distance to the k nearest normal clips (standardised, cosine)."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, e):
        self.sc = StandardScaler().fit(e)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(self.sc.transform(e))
        return self

    def score(self, e):
        return self.nn.kneighbors(self.sc.transform(e))[0].mean(1)


class GMMDetector:
    """Negative log-likelihood under a full-covariance GMM in PCA space."""

    def __init__(self, dim=32, n=2):
        self.dim, self.n = dim, n

    def fit(self, e):
        self.sc = StandardScaler().fit(e)
        z = self.sc.transform(e)
        self.pca = PCA(min(self.dim, len(e) // 4), random_state=SEED, whiten=True).fit(z)
        self.gmm = GaussianMixture(self.n, reg_covar=1e-3, random_state=SEED).fit(self.pca.transform(z))
        return self

    def score(self, e):
        return -self.gmm.score_samples(self.pca.transform(self.sc.transform(e)))
