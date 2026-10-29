"""
NeuroCamo - Efficiency Comparison  (power + detection, vs ANN/CNN)
==================================================================
Panel/paper graphs comparing NeuroCamo against conventional ANN/CNN models on
two axes:

  POWER  — theoretical energy/inference using the SAME model the SNN literature
           uses (Horowitz 2014, 45 nm CMOS: E_MAC = 4.6 pJ, E_AC = 0.9 pJ; and
           E_SNN = E_AC * firing_rate * T * FLOPs, per Zhou et al. 2023,
           Spikformer). We report the HONEST current operating point AND the
           design-target sparsity, with the ANN-parity break-even line.

  DETECTION / FOOTPRINT — parameter count vs conventional detectors. NeuroCamo
           is ~0.54 M params; reference detector sizes are published values on
           their own datasets (shown for scale, NOT a same-dataset benchmark).

No torch/GPU needed — analytic FLOPs from the documented backbone geometry.
FLOPs/sparsity/T are constants below; edit them if the architecture changes.

    venv\\Scripts\\python.exe evaluation\\efficiency_compare.py
Outputs (300-dpi PNG) -> results/efficiency_graphs/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "efficiency_graphs")

# ── constants (published) ─────────────────────────────────────────────────────
E_MAC = 4.6e-12          # J, 45nm CMOS multiply-accumulate  (Horowitz 2014)
E_AC = 0.9e-12           # J, 45nm CMOS accumulate            (Horowitz 2014)
T = 16                   # SNN timesteps
SPARSITY = 0.44          # measured mean hidden sparsity (from training logs)
FR = 1.0 - SPARSITY      # firing rate
TARGET_SPARSITY = 0.85   # design target (scnn_backbone.py comment)
FR_TARGET = 1.0 - TARGET_SPARSITY

# ── analytic backbone MACs (from models/scnn_backbone.py geometry) ─────────────
def _conv_macs(inC, outC, k, outH, outW):
    return outH * outW * outC * inC * k * k

FLOPS = (_conv_macs(3, 32, 3, 128, 128)      # conv1 @128x128
         + _conv_macs(32, 64, 3, 64, 64)     # conv2 @64x64
         + _conv_macs(64, 128, 3, 32, 32))   # conv3 @32x32

NC_PARAMS_M = 0.54       # NeuroCamo total params (538,676)

# reference detector/backbone sizes (published, different datasets — for scale)
REF_MODELS = [
    ("NeuroCamo\n(SNN, ours)", NC_PARAMS_M, "#2e7d32"),
    ("YOLO11n",       2.6,  "#607d8b"),
    ("YOLOv8n",       3.2,  "#607d8b"),
    ("YOLOv5s",       7.2,  "#607d8b"),
    ("SEW-ResNet18\n(SNN)", 11.7, "#8d6e63"),
    ("Spikformer-8-512\n(SNN)", 29.7, "#8d6e63"),
    ("SINet (COD ANN)", 48.9, "#455a64"),
]

def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def energy_ann():
    return E_MAC * FLOPS


def energy_snn(fr, t=T):
    return E_AC * fr * t * FLOPS


def plot_energy_bars():
    """Honest energy/inference: ANN vs SNN at current vs target sparsity."""
    labels = ["Equivalent\nANN (CNN)",
              f"NeuroCamo SNN\nnow ({int(SPARSITY*100)}% sparse)",
              f"NeuroCamo SNN\ntarget ({int(TARGET_SPARSITY*100)}% sparse)"]
    vals = [energy_ann()*1e3, energy_snn(FR)*1e3, energy_snn(FR_TARGET)*1e3]
    cols = ["#c62828", "#f9a825", "#2e7d32"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f} mJ",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(vals[0], ls="--", color="#c62828", alpha=0.5)
    ax.set_ylabel("Theoretical energy / inference (mJ)")
    ax.set_title("Power per inference — SNN vs equivalent ANN\n"
                 "(45 nm model, backbone conv ops)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, 0.94,
            "SNN beats ANN only once spikes are sparse enough (see break-even plot)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555",
            style="italic")
    _save(fig, "01_energy_per_inference.png")


def plot_breakeven():
    """Energy ratio (SNN/ANN) vs sparsity at T, with parity + operating points."""
    spars = np.linspace(0.0, 0.98, 200)
    fr = 1.0 - spars
    ratio = energy_snn(fr) / energy_ann()      # = (E_AC/E_MAC)*fr*T
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(spars*100, ratio, color="#1f77b4", lw=2.2, label=f"SNN/ANN energy (T={T})")
    ax.axhline(1.0, ls="--", color="k", alpha=0.6, label="ANN parity (=1.0)")
    # break-even sparsity
    be_fr = E_MAC/E_AC/T
    be_sp = (1-be_fr)*100
    ax.axvline(be_sp, ls=":", color="gray", alpha=0.7)
    ax.annotate(f"break-even\n{be_sp:.0f}% sparse", (be_sp, 1.0),
                textcoords="offset points", xytext=(6, 40), fontsize=9, color="gray")
    # current
    r_now = energy_snn(FR)/energy_ann()
    ax.scatter([SPARSITY*100], [r_now], color="#f9a825", s=90, zorder=5)
    ax.annotate(f"now: {SPARSITY*100:.0f}% sparse\n{r_now:.2f}x ANN",
                (SPARSITY*100, r_now), textcoords="offset points", xytext=(10, 6),
                fontsize=9, color="#b26a00", fontweight="bold")
    # target
    r_t = energy_snn(FR_TARGET)/energy_ann()
    ax.scatter([TARGET_SPARSITY*100], [r_t], color="#2e7d32", s=90, zorder=5)
    ax.annotate(f"target: {TARGET_SPARSITY*100:.0f}% sparse\n{r_t:.2f}x ANN "
                f"({energy_ann()/energy_snn(FR_TARGET):.1f}x greener)",
                (TARGET_SPARSITY*100, r_t), textcoords="offset points", xytext=(-30, 30),
                fontsize=9, color="#2e7d32", fontweight="bold")
    ax.fill_between(spars*100, 0, 1.0, where=(ratio <= 1.0), color="#2e7d32", alpha=0.08)
    ax.set_xlabel("Hidden-layer spike sparsity (%)")
    ax.set_ylabel("SNN energy  /  ANN energy")
    ax.set_title("Where the SNN wins: energy vs spike sparsity",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(2.2, ratio.max()*1.05)); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    _save(fig, "02_breakeven_sparsity.png")


def plot_timestep_lever():
    """Energy ratio vs timesteps T for current & target firing rate."""
    ts = np.arange(1, 33)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fr, name, c in [(FR, f"now (fr={FR:.2f})", "#f9a825"),
                        (FR_TARGET, f"target (fr={FR_TARGET:.2f})", "#2e7d32")]:
        ax.plot(ts, (E_AC*fr*ts*FLOPS)/energy_ann(), "-o", ms=3, color=c, label=name)
    ax.axhline(1.0, ls="--", color="k", alpha=0.6, label="ANN parity")
    ax.axvline(T, ls=":", color="gray", alpha=0.6)
    ax.annotate(f"current T={T}", (T, ax.get_ylim()[1]*0.9), fontsize=9, color="gray")
    ax.set_xlabel("SNN timesteps T"); ax.set_ylabel("SNN energy / ANN energy")
    ax.set_title("Second lever: fewer timesteps also cut SNN energy",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, "03_timestep_lever.png")


def plot_param_footprint():
    """Detection-model footprint: params vs conventional detectors (log scale)."""
    names = [m[0] for m in REF_MODELS]
    vals = [m[1] for m in REF_MODELS]
    cols = [m[2] for m in REF_MODELS]
    order = np.argsort(vals)
    names = [names[i] for i in order]; vals = [vals[i] for i in order]; cols = [cols[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.65)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:g}M",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Parameters (millions, log scale)")
    ax.set_title("Detection-model footprint — NeuroCamo vs conventional detectors\n"
                 "(published sizes; datasets differ — shown for scale)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    _save(fig, "04_parameter_footprint.png")


def plot_dashboard():
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    # energy bars
    ax0 = fig.add_subplot(gs[0])
    labels = ["ANN", f"SNN now\n{int(SPARSITY*100)}%sp", f"SNN target\n{int(TARGET_SPARSITY*100)}%sp"]
    vals = [energy_ann()*1e3, energy_snn(FR)*1e3, energy_snn(FR_TARGET)*1e3]
    bb = ax0.bar(labels, vals, color=["#c62828", "#f9a825", "#2e7d32"], width=0.6)
    for b, v in zip(bb, vals):
        ax0.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax0.set_ylabel("Energy / inference (mJ)")
    ax0.set_title("Power per inference", fontweight="bold"); ax0.grid(axis="y", alpha=0.3)
    # footprint
    ax1 = fig.add_subplot(gs[1])
    names = [m[0].replace("\n", " ") for m in REF_MODELS]; vals2 = [m[1] for m in REF_MODELS]
    cols = [m[2] for m in REF_MODELS]
    order = np.argsort(vals2)
    names = [names[i] for i in order]; vals2 = [vals2[i] for i in order]; cols = [cols[i] for i in order]
    ax1.barh(range(len(vals2)), vals2, color=cols)
    ax1.set_xscale("log"); ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=8)
    for i, v in enumerate(vals2):
        ax1.text(v, i, f" {v:g}M", va="center", fontsize=8, fontweight="bold")
    ax1.set_xlabel("Parameters (M, log)"); ax1.set_title("Model footprint", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, which="both")
    fig.suptitle("NeuroCamo — Efficiency vs Conventional ANN/CNN Models",
                 fontsize=15, fontweight="bold")
    _save(fig, "00_efficiency_dashboard.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Backbone FLOPs = {FLOPS/1e6:.1f}M MACs | ANN {energy_ann()*1e3:.3f} mJ | "
          f"SNN now {energy_snn(FR)*1e3:.3f} mJ ({energy_snn(FR)/energy_ann():.2f}x) | "
          f"SNN target {energy_snn(FR_TARGET)*1e3:.3f} mJ "
          f"({energy_ann()/energy_snn(FR_TARGET):.1f}x greener)")
    plot_dashboard()
    plot_energy_bars()
    plot_breakeven()
    plot_timestep_lever()
    plot_param_footprint()
    print(f"\nDone. Graphs in {OUT}")


if __name__ == "__main__":
    main()
