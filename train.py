"""
train.py -- train the same U-Net under the random split and the geographic hold-out split.

Each run is saved to  <runs>/<protocol>_<channels>_seed<k>/  containing:
  model.pt      best-validation-IoU weights
  meta.json     protocol, channels, seed, split indices, normalisation stats, hyper-parameters
  history.json  per-epoch train loss and validation IoU

Usage:
  python train.py --data_root data --holdouts Pakistan --seeds 0 1 2
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import metrics_from_counts, predict
from model import UNet, masked_bce_dice
from preprocessing import (CHANNEL_SETS, ChipDS, get_split, load_or_build_cache,
                           train_stats)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dir_name(protocol, chans_name, seed):
    return f"{protocol}_{chans_name.replace('+', '')}_seed{seed}"


def fit(model, train_ds, val_ds, args, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_iou, best_state, history = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = masked_bce_dice(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        val_counts, _ = predict(model, val_ds, device)
        val_iou = float(metrics_from_counts(val_counts.sum(0))["iou"])
        history.append(dict(epoch=ep, train_loss=total / len(train_ds), val_iou=val_iou))
        if val_iou > best_iou:                      # model selection on VALIDATION only
            best_iou, best_state = val_iou, copy.deepcopy(model.state_dict())
        if ep == 1 or ep % 5 == 0:
            print(f"    epoch {ep:3d} | loss {total / len(train_ds):.4f} | val IoU {val_iou:.4f}")
    model.load_state_dict(best_state)
    return model, history, best_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--holdouts", nargs="+", default=["Pakistan"],
                    help="country(ies) held out entirely for the geographic test set")
    ap.add_argument("--channels", nargs="+", default=["VV+VH", "VV"], choices=list(CHANNEL_SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="U-Net base channels")
    ap.add_argument("--overwrite", action="store_true", help="retrain even if a run exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    df, X, Y = load_or_build_cache(args.data_root, args.cache, args.size)
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)

    protocols = ["random"] + [f"geo-{h}" for h in args.holdouts]
    for protocol in protocols:
        for chans_name in args.channels:
            for seed in args.seeds:
                rdir = runs / run_dir_name(protocol, chans_name, seed)
                if (rdir / "model.pt").exists() and not args.overwrite:
                    print(f"skip (exists): {rdir.name}")
                    continue
                rdir.mkdir(parents=True, exist_ok=True)

                chans = CHANNEL_SETS[chans_name]
                train, val, test = get_split(df, protocol, seed)
                mean, std = train_stats(X, Y, train)           # TRAIN-only statistics
                tr = ChipDS(X, Y, train, mean, std, chans, augment=True)
                va = ChipDS(X, Y, val, mean, std, chans)

                print(f"\n[{protocol} | {chans_name} | seed {seed}] "
                      f"train {len(train)} / val {len(val)} / test {len(test)}")
                set_seed(seed)
                model = UNet(in_ch=len(chans), base=args.base).to(device)
                model, history, best_val = fit(model, tr, va, args, device)

                torch.save(model.state_dict(), rdir / "model.pt")
                json.dump(history, open(rdir / "history.json", "w"))
                json.dump(dict(protocol=protocol, channels=chans_name, seed=seed,
                               best_val_iou=best_val,
                               train_idx=train.tolist(), val_idx=val.tolist(), test_idx=test.tolist(),
                               mean=mean.tolist(), std=std.tolist(),
                               hparams=dict(size=args.size, epochs=args.epochs,
                                            batch_size=args.batch_size, lr=args.lr, base=args.base)),
                          open(rdir / "meta.json", "w"))
                print(f"  saved {rdir}  (best val IoU {best_val:.4f})")


if __name__ == "__main__":
    main()
