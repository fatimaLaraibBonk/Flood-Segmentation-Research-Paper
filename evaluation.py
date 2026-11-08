"""
evaluation.py -- evaluate every trained run on its own test split and compare protocols.

For each run in <runs>/ it rebuilds the test set from the stored indices and train-only
normalisation stats, then reports IoU / F1 / precision / recall (+ accuracy) with per-image
bootstrap 95% CIs. It then computes the relative generalisation drop
    (random - geographic) / random * 100
and writes tables and figures to <out>/.

Usage:
  python evaluation.py --cache cache --runs runs --out results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from metrics import bootstrap_ci, drop_ci, metrics_from_counts, predict
from model import UNet
from preprocessing import CHANNEL_SETS, ChipDS, load_or_build_cache

METRICS = ["iou", "f1", "precision", "recall"]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def save_prediction_figure(ds, preds, path, title, n=4):
    """Columns: VV | ground truth | prediction | error map (TP white, FP red, FN blue)."""
    n = min(n, len(ds))
    sel = np.linspace(0, len(ds) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(sel):
        vv, gt, pr = ds.x[i, 0].numpy(), ds.y[i].numpy(), preds[i]
        err = np.zeros(gt.shape + (3,))
        valid = gt >= 0
        err[(pr == 1) & (gt == 0) & valid] = [1, 0, 0]
        err[(pr == 0) & (gt == 1) & valid] = [0, 0.3, 1]
        err[(pr == 1) & (gt == 1)] = [1, 1, 1]
        panels = [(vv, "gray", "VV (normalised)"),
                  (np.ma.masked_where(gt < 0, gt), "Blues", "Ground truth"),
                  (pr, "Blues", "Prediction"),
                  (err, None, "TP white / FP red / FN blue")]
        for a, (im, cm, t) in zip(axes[r], panels):
            a.imshow(im, cmap=cm) if cm else a.imshow(im)
            a.set_title(t, fontsize=9)
            a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bar_figure(results, out):
    sub = results[results.channels == "VV+VH"]
    protos = list(sub.protocol.unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(protos)
    for j, p in enumerate(protos):
        s = sub[sub.protocol == p]
        ax.bar(np.arange(len(METRICS)) + j * w, [s[m].mean() for m in METRICS], w,
               yerr=[s[m].std(ddof=0) for m in METRICS], capsize=3, label=p)
    ax.set_xticks(np.arange(len(METRICS)) + w * (len(protos) - 1) / 2)
    ax.set_xticklabels(["IoU", "F1", "Precision", "Recall"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (mean ± std over seeds)")
    ax.set_title("Random split vs geographic hold-out (VV+VH)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_metrics_random_vs_geo.png", dpi=150)
    plt.close(fig)


def ablation_figure(results, out):
    piv = results.groupby(["protocol", "channels"])["iou"].agg(["mean", "std"]).reset_index()
    protos, chans = list(piv.protocol.unique()), list(piv.channels.unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(chans)
    for j, c in enumerate(chans):
        s = piv[piv.channels == c].set_index("protocol").reindex(protos)
        ax.bar(np.arange(len(protos)) + j * w, s["mean"], w, yerr=s["std"].fillna(0), capsize=3, label=c)
    ax.set_xticks(np.arange(len(protos)) + w * (len(chans) - 1) / 2)
    ax.set_xticklabels(protos)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test IoU")
    ax.set_title("Input ablation: VV vs VV+VH")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_ablation_channels.png", dpi=150)
    plt.close(fig)


def curves_figure(histories, out):
    first = min(k[2] for k in histories)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (proto, chans, seed), h in histories.items():
        if chans == "VV+VH" and seed == first:
            ax.plot([e["epoch"] for e in h], [e["val_iou"] for e in h], label=proto)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation IoU")
    ax.set_title("Validation IoU during training (VV+VH)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_training_curves.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, X, Y = load_or_build_cache(None, args.cache, args.size)   # cache must already exist

    run_dirs = sorted(d for d in Path(args.runs).iterdir() if (d / "model.pt").exists())
    if not run_dirs:
        raise FileNotFoundError(f"No trained runs found in {args.runs}. Run train.py first.")
    first_seed = min(json.load(open(d / "meta.json"))["seed"] for d in run_dirs)

    rows, counts_store, histories, per_image = [], {}, {}, []
    for d in run_dirs:
        meta = json.load(open(d / "meta.json"))
        protocol, chans_name, seed = meta["protocol"], meta["channels"], meta["seed"]
        chans = CHANNEL_SETS[chans_name]
        test_idx = np.array(meta["test_idx"])
        te = ChipDS(X, Y, test_idx, meta["mean"], meta["std"], chans)

        model = UNet(in_ch=len(chans), base=meta["hparams"]["base"]).to(device)
        model.load_state_dict(torch.load(d / "model.pt", map_location=device))
        counts, preds = predict(model, te, device)
        counts_store[(protocol, chans_name, seed)] = counts
        histories[(protocol, chans_name, seed)] = json.load(open(d / "history.json"))

        m = {k: float(v) for k, v in metrics_from_counts(counts.sum(0)).items()}
        ci = bootstrap_ci(counts, args.n_boot, seed)
        row = dict(protocol=protocol, channels=chans_name, seed=seed,
                   best_val_iou=meta["best_val_iou"], n_test=len(test_idx), **m)
        for k, (lo, hi) in ci.items():
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
        rows.append(row)
        print(f"{d.name:32s} IoU {m['iou']:.3f} [{ci['iou'][0]:.3f}, {ci['iou'][1]:.3f}] "
              f"F1 {m['f1']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}")

        # per-image scores (for picking failure cases)
        img_m = metrics_from_counts(counts)
        for j, ci_ in enumerate(test_idx):
            per_image.append(dict(protocol=protocol, channels=chans_name, seed=seed,
                                  chip=df.iloc[ci_].chip, country=df.iloc[ci_].country,
                                  iou=float(img_m["iou"][j]), f1=float(img_m["f1"][j])))

        if seed == first_seed and chans_name == "VV+VH":
            save_prediction_figure(te, preds, out / f"pred_{protocol}_{chans_name.replace('+', '')}.png",
                                   f"{protocol} | {chans_name} | seed {seed}")

    results = pd.DataFrame(rows).sort_values(["protocol", "channels", "seed"])
    results.to_csv(out / "results_per_run.csv", index=False)
    pd.DataFrame(per_image).to_csv(out / "per_image_scores.csv", index=False)

    summary = results.groupby(["protocol", "channels"])[METRICS + ["accuracy"]].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "results_summary.csv")

    # --- generalisation drop (per seed, with bootstrap CI for IoU) and averaged ---
    drop_rows = []
    for (chans, seed), g in results.groupby(["channels", "seed"]):
        rnd = g[g.protocol == "random"]
        if rnd.empty:
            continue
        for proto in [p for p in g.protocol.unique() if p != "random"]:
            geo = g[g.protocol == proto]
            if geo.empty:
                continue
            row = dict(channels=chans, seed=seed, geo_protocol=proto)
            for m in METRICS:
                r, ge = float(rnd[m].iloc[0]), float(geo[m].iloc[0])
                row[f"{m}_random"], row[f"{m}_geo"] = r, ge
                row[f"{m}_drop_pct"] = (r - ge) / max(r, 1e-9) * 100
            lo, hi = drop_ci(counts_store[("random", chans, seed)],
                             counts_store[(proto, chans, seed)], "iou", args.n_boot, seed)
            row["iou_drop_ci_lo"], row["iou_drop_ci_hi"] = lo, hi
            drop_rows.append(row)
    drop_seed = pd.DataFrame(drop_rows)
    drop_seed.to_csv(out / "generalisation_drop_per_seed.csv", index=False)
    drop_mean = (drop_seed.groupby(["channels", "geo_protocol"])
                 [[c for c in drop_seed.columns if c.endswith(("_random", "_geo", "_drop_pct"))]]
                 .agg(["mean", "std"]).round(4))
    drop_mean.to_csv(out / "generalisation_drop_summary.csv")

    bar_figure(results, out)
    ablation_figure(results, out)
    curves_figure(histories, out)

    print("\n=== Summary (mean/std over seeds) ===\n", summary.to_string())
    print("\n=== Relative generalisation drop, % (mean/std over seeds) ===\n", drop_mean.to_string())
    print(f"\nAll outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
