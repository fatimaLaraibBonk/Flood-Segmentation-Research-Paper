"""metrics.py -- pixel counts, segmentation metrics, bootstrap CIs (shared by train/evaluation)."""
import numpy as np
import torch
from torch.utils.data import DataLoader


def metrics_from_counts(c):
    """c[..., :] = (TP, FP, FN, TN) summed over pixels -> pooled metrics."""
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return dict(iou=tp / (tp + fp + fn + eps),
                f1=2 * prec * rec / (prec + rec + eps),
                precision=prec, recall=rec,
                accuracy=(tp + tn) / (tp + fp + fn + tn + eps))


@torch.no_grad()
def predict(model, ds, device, bs=16):
    """Returns per-image counts (N, 4) and binary prediction masks (N, H, W)."""
    model.eval()
    counts, preds = [], []
    for x, y in DataLoader(ds, batch_size=bs, shuffle=False):
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x)) > 0.5
        valid, gt = y >= 0, y == 1
        tp = (pred & gt & valid).sum((1, 2))
        fp = (pred & ~gt & valid).sum((1, 2))
        fn = (~pred & gt & valid).sum((1, 2))
        tn = (~pred & ~gt & valid).sum((1, 2))
        counts.append(torch.stack([tp, fp, fn, tn], 1).cpu())
        preds.append(pred.cpu())
    return torch.cat(counts).numpy().astype(np.float64), torch.cat(preds).numpy()


def bootstrap_ci(counts, n_boot=1000, seed=0):
    """Per-image bootstrap: resample images, recompute pooled metrics -> 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(counts)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = metrics_from_counts(counts[idx].sum(axis=1))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in m.items()}


def drop_ci(counts_random, counts_geo, metric="iou", n_boot=1000, seed=0):
    """95% CI of (random - geo)/random*100, resampling each test set independently."""
    rng = np.random.default_rng(seed)
    ir = rng.integers(0, len(counts_random), size=(n_boot, len(counts_random)))
    ig = rng.integers(0, len(counts_geo), size=(n_boot, len(counts_geo)))
    r = metrics_from_counts(counts_random[ir].sum(axis=1))[metric]
    g = metrics_from_counts(counts_geo[ig].sum(axis=1))[metric]
    d = (r - g) / np.maximum(r, 1e-9) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
