"""
NeuroCamo - Training History Plots  (for the review panel)
==========================================================
Parses the training logs in checkpoints/*.txt and renders panel-ready graphs
showing accuracy BEFORE vs AFTER training, epoch-by-epoch learning curves,
loss curves and the SNN sparsity trend.

No torch / GPU needed — pure log parsing + matplotlib. Re-run any time after
more training to refresh the figures:

    venv\\Scripts\\python.exe evaluation\\plot_training_history.py

Outputs (300-dpi PNG) -> results/training_graphs/
"""

import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
OUT = os.path.join(ROOT, "results", "training_graphs")

# log file  ->  human-readable run name
RUNS = [
    ("military_train_log.txt",     "Military 12-class"),
    ("mixed_train_log.txt",        "Mixed (COD10K + Military)"),
    ("military_neg_train_log.txt", "Military + Negatives"),
]

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+done.*?Train\s+([\d.]+)\s+Acc\s+([\d.]+)%.*?"
    r"Val\s+([\d.]+)\s+Acc\s+([\d.]+)%.*?Sparsity\s+([\d.]+)%")

COL = {"train": "#1f77b4", "val": "#d62728", "sp": "#2e7d32"}
def _read(path):
    raw = open(path, "rb").read()
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("latin-1", "ignore")


def parse_run(fname):
    """-> dict with epochs, train_loss/acc, val_loss/acc, sparsity (lists)."""
    path = os.path.join(CKPT, fname)
    if not os.path.exists(path):
        return None
    d = dict(epoch=[], tr_loss=[], tr_acc=[], val_loss=[], val_acc=[], sp=[])
    for m in EPOCH_RE.finditer(_read(path)):
        e, trl, tra, vl, va, sp = m.groups()
        d["epoch"].append(int(e))
        d["tr_loss"].append(float(trl)); d["tr_acc"].append(float(tra))
        d["val_loss"].append(float(vl)); d["val_acc"].append(float(va))
        d["sp"].append(float(sp))
    return d if d["epoch"] else None


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def plot_before_after(runs):
    """Headline: val accuracy at epoch 1 (before learning) vs best achieved."""
    names = [n for n, _ in runs]
    before = [r["val_acc"][0] for _, r in runs]
    after = [max(r["val_acc"]) for _, r in runs]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w/2, before, w, label="Before (epoch 1)", color="#9e9e9e")
    b2 = ax.bar(x + w/2, after, w, label="After (best val)", color="#2e7d32")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f"{b.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Validation accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("NeuroCamo SNN — Detection accuracy before vs after training",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "01_before_after_accuracy.png")


def plot_accuracy_curves(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(6*len(runs), 5), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, runs):
        ax.plot(r["epoch"], r["tr_acc"], "-o", color=COL["train"], label="Train")
        ax.plot(r["epoch"], r["val_acc"], "-s", color=COL["val"], label="Val")
        best = max(r["val_acc"]); be = r["epoch"][int(np.argmax(r["val_acc"]))]
        ax.axhline(best, ls="--", color=COL["val"], alpha=0.4)
        ax.annotate(f"best {best:.1f}%", (be, best), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=COL["val"], fontweight="bold")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Learning curves — accuracy per epoch", fontsize=13, fontweight="bold")
    _save(fig, "02_accuracy_curves.png")


def plot_loss_curves(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(6*len(runs), 5), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, runs):
        ax.plot(r["epoch"], r["tr_loss"], "-o", color=COL["train"], label="Train loss")
        ax.plot(r["epoch"], r["val_loss"], "-s", color=COL["val"], label="Val loss")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("Loss")
    fig.suptitle("Learning curves — loss per epoch", fontsize=13, fontweight="bold")
    _save(fig, "03_loss_curves.png")


def plot_sparsity(runs):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, r in runs:
        ax.plot(r["epoch"], r["sp"], "-o", label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hidden-layer spike sparsity (%)")
    ax.set_title("SNN spike sparsity across training (energy-efficiency indicator)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, "04_sparsity.png")


def plot_dashboard(runs):
    """One-slide summary: before/after bars + all val-accuracy curves."""
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax0 = fig.add_subplot(gs[0])
    names = [n for n, _ in runs]
    before = [r["val_acc"][0] for _, r in runs]
    after = [max(r["val_acc"]) for _, r in runs]
    x = np.arange(len(names)); w = 0.38
    ax0.bar(x - w/2, before, w, label="Before", color="#9e9e9e")
    bb = ax0.bar(x + w/2, after, w, label="After (best)", color="#2e7d32")
    for b in bb:
        ax0.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
    ax0.set_ylim(0, 105); ax0.set_ylabel("Val accuracy (%)")
    ax0.set_title("Before vs after training", fontweight="bold")
    ax0.legend(); ax0.grid(axis="y", alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    for name, r in runs:
        ax1.plot(r["epoch"], r["val_acc"], "-o", label=name)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val accuracy (%)")
    ax1.set_title("Validation accuracy per epoch", fontweight="bold")
    ax1.grid(alpha=0.3); ax1.legend()

    fig.suptitle("NeuroCamo — Training Results Summary", fontsize=15, fontweight="bold")
    _save(fig, "00_summary_dashboard.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for fname, name in RUNS:
        r = parse_run(fname)
        if r:
            runs.append((name, r))
            print(f"[{name}] {len(r['epoch'])} epochs | "
                  f"val {r['val_acc'][0]:.1f}% -> best {max(r['val_acc']):.1f}%")
        else:
            print(f"[skip] {fname} — no completed epochs found")
    if not runs:
        print("No training data parsed. Nothing to plot."); return
    plot_dashboard(runs)
    plot_before_after(runs)
    plot_accuracy_curves(runs)
    plot_loss_curves(runs)
    plot_sparsity(runs)
    print(f"\nDone. Graphs in {OUT}")


if __name__ == "__main__":
    main()
