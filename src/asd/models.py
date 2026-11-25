from torch import nn

from .config import CONTEXT, N_MELS, PATCH


class DenseAE(nn.Module):
    """DCASE 2020 Task 2 baseline: 640-128x4-8-128x4-640."""

    def __init__(self, dim=N_MELS * CONTEXT, hidden=128, bottleneck=8):
        super().__init__()
        sizes = [dim] + [hidden] * 4 + [bottleneck] + [hidden] * 4
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class ConvAE(nn.Module):
    """Strided convolutional AE on [1, 128, 64] patches, narrow linear bottleneck."""

    def __init__(self, ch=(16, 32, 64, 64), bottleneck=40):
        super().__init__()
        enc, c = [], 1
        for k in ch:
            enc += [nn.Conv2d(c, k, 4, 2, 1), nn.BatchNorm2d(k), nn.LeakyReLU(0.1)]
            c = k
        self.shape = (c, N_MELS >> len(ch), PATCH >> len(ch))
        flat = c * self.shape[1] * self.shape[2]
        self.enc = nn.Sequential(*enc, nn.Flatten(), nn.Linear(flat, bottleneck))
        self.up = nn.Linear(bottleneck, flat)
        dec = []
        for k in (*ch[-2::-1], 1):
            dec.append(nn.ConvTranspose2d(c, k, 4, 2, 1))
            if k != 1:
                dec += [nn.BatchNorm2d(k), nn.LeakyReLU(0.1)]
            c = k
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(self.up(self.enc(x)).view(-1, *self.shape))
