"""model.py -- standard U-Net and the training loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """4-level U-Net. Output: logits of shape (B, H, W)."""

    def __init__(self, in_ch, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]       # 32, 64, 128, 256, 512
        self.enc = nn.ModuleList([DoubleConv(in_ch, chs[0])] +
                                 [DoubleConv(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([DoubleConv(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x).squeeze(1)


def masked_bce_dice(logits, y):
    """BCE + Dice, ignoring no-data pixels (label == -1)."""
    valid = (y >= 0).float()
    t = (y == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (t * valid).sum() + 1)
    return bce + dice
