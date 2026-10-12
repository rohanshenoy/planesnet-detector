"""
S2Net: classifies a Sentinel-2 chip (10 m pixels) as aircraft in flight or not.

A moving aircraft shows in Sentinel-2 as blue, green and red copies up to about 45 px
(450 m) apart, because the bands are recorded a fraction of a second apart. PlaneNet
sees a 60 m window built for 3 m PlanesNet chips, so it never sees the whole signature.
S2Net looks at a 48x48 px window (480 m) centred on the candidate, so all three copies
are in view, and is trained on real Sentinel-2 aircraft (data/s2_flying).
"""

import torch
from torch import nn

WINDOW = 48


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class S2Net(nn.Module):
    """(N, 3, 48, 48) RGB in [0, 1] -> (N,) logits."""

    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(_block(3, width), _block(width, 2 * width), _block(2 * width, 4 * width))  # 48 -> 6
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * width, 1))

    def forward(self, x):
        f = self.features(x)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)  # average and max over the 6x6 map
        return self.head(pooled).squeeze(1)


def save(model, path, **meta):
    torch.save({'state_dict': model.state_dict(), 'width': model.features[0][0].out_channels, **meta}, path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = S2Net(width=ck.get('width', 32)).to(device)
    model.load_state_dict(ck['state_dict'])
    return model.eval()
