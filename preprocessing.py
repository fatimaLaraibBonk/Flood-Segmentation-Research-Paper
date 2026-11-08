"""
preprocessing.py -- data indexing, cleaning, caching, splitting, normalisation.

Expected raw layout (Sen1Floods11 hand-labeled subset):
  <data_root>/S1Hand/<Country>_<id>_S1Hand.tif       2 bands: VV, VH (dB)
  <data_root>/LabelHand/<Country>_<id>_LabelHand.tif  -1 = no data, 0 = land, 1 = water

Running this file builds the cache and reports chip counts and split sizes:
  python preprocessing.py --data_root data --holdouts Pakistan
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CHANNEL_SETS = {"VV": [0], "VV+VH": [0, 1]}   # band 1 = VV, band 2 = VH
DB_MIN, DB_MAX = -50.0, 1.0                    # clip range for dB values


# ----------------------------------------------------------------------------
# Indexing / loading / cleaning
# ----------------------------------------------------------------------------
def build_index(root):
    root = Path(root)
    rows = []
    for p in sorted((root / "S1Hand").glob("*_S1Hand.tif")):
        chip = p.name.replace("_S1Hand.tif", "")
        lbl = root / "LabelHand" / f"{chip}_LabelHand.tif"
        if lbl.exists():
            rows.append(dict(chip=chip, country=chip.split("_")[0],
                             s1=str(p), label=str(lbl)))
    if not rows:
        raise FileNotFoundError(f"No S1Hand/LabelHand pairs found under {root}")
    return pd.DataFrame(rows)


def load_chip(s1_path, lbl_path, size):
    with rasterio.open(s1_path) as f:
        x = f.read().astype(np.float32)          # (2, H, W)
    with rasterio.open(lbl_path) as f:
        y = f.read(1).astype(np.float32)         # (H, W)

    bad = ~np.isfinite(x).all(axis=0)            # NaN/inf in either band
    x = np.nan_to_num(x, nan=DB_MIN, posinf=DB_MAX, neginf=DB_MIN)
    x = np.clip(x, DB_MIN, DB_MAX)
    y[bad] = -1                                  # corrupt pixels become no-data

    # fixed input size: area interpolation for the image, nearest for the label
    xt = F.interpolate(torch.from_numpy(x)[None], size=(size, size), mode="area")[0]
    yt = F.interpolate(torch.from_numpy(y)[None, None], size=(size, size), mode="nearest")[0, 0]
    return xt.numpy(), yt.numpy().astype(np.int8)


def load_or_build_cache(data_root, cache_dir, size):
    """Load all chips once, save to cache_dir, reuse on later runs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz, idx_csv = cache_dir / f"chips_{size}.npz", cache_dir / "chip_index.csv"
    if npz.exists() and idx_csv.exists():
        d = np.load(npz)
        return pd.read_csv(idx_csv), d["X"], d["Y"]

    df = build_index(data_root)
    X, Y = [], []
    for i, r in enumerate(df.itertuples()):
        x, y = load_chip(r.s1, r.label, size)
        X.append(x)
        Y.append(y)
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(df)} chips")
    X, Y = np.stack(X), np.stack(Y)
    np.savez_compressed(npz, X=X, Y=Y)
    df.to_csv(idx_csv, index=False)
    return df, X, Y


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------
def _train_val(idx, val_frac, rng):
    idx = np.array(idx)
    rng.shuffle(idx)
    n_val = int(round(val_frac * len(idx)))
    return idx[n_val:], idx[:n_val]


def random_split(n, seed, test_frac=0.20, val_frac=0.10):
    """~72% train / 8% val / 20% test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test, rest = idx[:n_test], idx[n_test:]
    train, val = _train_val(rest, val_frac, rng)
    return train, val, test


def geographic_split(df, holdout, seed, val_frac=0.10):
    """Whole held-out country = test; remaining chips split 90/10 train/val."""
    rng = np.random.default_rng(seed)
    is_test = (df["country"] == holdout).to_numpy()
    if is_test.sum() == 0:
        raise ValueError(f"Country '{holdout}' not found. "
                         f"Available: {sorted(df['country'].unique())}")
    test = np.where(is_test)[0]
    train, val = _train_val(np.where(~is_test)[0], val_frac, rng)
    return train, val, test


def get_split(df, protocol, seed):
    """protocol is 'random' or 'geo-<Country>' (e.g. 'geo-Pakistan')."""
    if protocol == "random":
        return random_split(len(df), seed)
    if protocol.startswith("geo-"):
        return geographic_split(df, protocol[4:], seed)
    raise ValueError(f"Unknown protocol: {protocol}")


def describe_split(protocol, df, train, val, test):
    n = len(df)
    return dict(protocol=protocol, n_train=len(train), n_val=len(val), n_test=len(test),
                pct_train=round(100 * len(train) / n, 1),
                pct_val=round(100 * len(val) / n, 1),
                pct_test=round(100 * len(test) / n, 1),
                test_countries=",".join(sorted(df.iloc[test]["country"].unique())))


# ----------------------------------------------------------------------------
# Normalisation (TRAIN statistics only) and Dataset
# ----------------------------------------------------------------------------
def train_stats(X, Y, train_idx):
    """Per-channel mean/std over valid (label >= 0) training pixels."""
    xs, ys = X[train_idx], Y[train_idx]
    valid = ys >= 0
    mean = np.array([xs[:, c][valid].mean() for c in range(xs.shape[1])], dtype=np.float32)
    std = np.array([xs[:, c][valid].std() for c in range(xs.shape[1])], dtype=np.float32)
    return mean, np.maximum(std, 1e-6)


class ChipDS(Dataset):
    """Normalised chips for one split. Flips / rot90 only when augment=True (train)."""

    def __init__(self, X, Y, idx, mean, std, chans, augment=False):
        mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        x = (X[idx] - mean[None, :, None, None]) / std[None, :, None, None]
        self.x = torch.from_numpy(x[:, chans]).float()
        self.y = torch.from_numpy(Y[idx].astype(np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        if self.augment:
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = random.randint(0, 3)
            if k:
                x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        return x, y


# ----------------------------------------------------------------------------
# CLI: build cache + document the dataset and splits
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--holdouts", nargs="+", default=["Pakistan"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df, X, Y = load_or_build_cache(a.data_root, a.cache, a.size)
    print("\nChips per country:\n", df.country.value_counts().to_string())
    flood = float((Y == 1).sum() / max((Y >= 0).sum(), 1))
    nodata = float((Y < 0).mean())
    print(f"\n{len(df)} chips of {a.size}x{a.size} | flood fraction {flood:.3f} | no-data fraction {nodata:.3f}")

    rows = [describe_split("random", df, *get_split(df, "random", a.seed))]
    for h in a.holdouts:
        rows.append(describe_split(f"geo-{h}", df, *get_split(df, f"geo-{h}", a.seed)))
    out = pd.DataFrame(rows)
    out.to_csv(Path(a.cache) / "split_counts.csv", index=False)
    print("\nSplit summary:\n", out.to_string(index=False))
