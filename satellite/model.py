"""
Fully convolutional aircraft classifier for 20x20 satellite chips.

The network uses only unpadded convolutions and aligned pooling, so running it
on a whole scene produces exactly the same scores as classifying every 20x20
window with a stride of 4 px, in a single forward pass instead of one call per
window.
"""

import torch
from torch import nn

WINDOW = 20
STRIDE = 4


def _block(cin, cout):
    return [nn.Conv2d(cin, cout, 3), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class PlaneNet(nn.Module):
    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            *_block(3, width), *_block(width, width), nn.MaxPool2d(2),          # 20 -> 8
            *_block(width, 2 * width), *_block(2 * width, 2 * width), nn.MaxPool2d(2),  # 8 -> 2
        )
        self.head = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 2), nn.ReLU(inplace=True),        # 2 -> 1
            nn.Dropout2d(dropout),
            nn.Conv2d(4 * width, 1, 1),
        )

    def forward(self, x):
        """x: (N, 3, H, W) in [0, 1]. Returns logits (N, 1, (H-20)//4+1, (W-20)//4+1)."""
        return self.head(self.features(x))


def save(model, path, **meta):
    torch.save({'state_dict': model.state_dict(), 'width': model.features[0].out_channels,
                **meta}, path)


def load(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PlaneNet(width=ckpt['width'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()
