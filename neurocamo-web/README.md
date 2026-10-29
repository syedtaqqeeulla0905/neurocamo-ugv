# NeuroCamo Web

Local browser front end for the NeuroCamo spiking neural network — the same
camouflage detector as `neurocamo/utils/test_gui.py`, served as a web page
instead of a Tkinter window, with the layer-3 spike spectrogram added.

Visual language (near-black ground, single mint accent, DM Mono body / Outfit
display, film grain, hairline rules) is carried over from Echolyx so the two
demos read as one product family. Nothing in `echolyx-web/` is touched or
imported — the tokens are duplicated in `static/styles.css`.

## Run

```bash
E:/major_project/neurocamo/venv/Scripts/python.exe E:/major_project/neurocamo-web/app.py
```

Or double-click `run.bat`. Then open <http://127.0.0.1:8080>.

The server reuses the existing `neurocamo/venv` (torch, snntorch, opencv, flask
are already installed there). Port 8080 is the default; override with
`NEUROCAMO_PORT`.

It binds to `127.0.0.1`, so it is reachable only from this machine. `/api/detect`
has no authentication — if you set `NEUROCAMO_HOST=0.0.0.0` to reach it from a
phone on the same Wi-Fi, anyone on that network can post images to the model.

## What it shows

| Panel | Content |
| --- | --- |
| Verdict | Gated confidence dial, verdict word, six readouts (raw SNN score, texture, sparsity, latency, active channels) |
| Input / Detection | Your frame beside the annotated result; the box is drawn only above the 25% gate, exactly as in the desktop tool |
| Layer-3 spike spectrogram | 128 semantic channels × 16 Poisson timesteps, cell brightness = that channel's firing rate over the 16×16 grid. Hover for `ch / t / rate`; the three legend chips filter by firing density |
| Population firing rate | Mean layer-3 spike rate per timestep — the network's membrane charge-up curve |
| Scene gate | The three heuristics (plain backdrop, open sky, texture floor) that multiply the raw SNN score by 0 or 1, with pass/block per check |

The spectrogram is the SNN analogue of an audio spectrogram: rows are channels
rather than frequency bins, columns are timesteps rather than frames. It uses the
same banded colormap idea as Echolyx — hue encodes which density band a cell
falls in (sparse / active / saturated), luminance encodes rate.

## Detection logic

`inference.py` is a direct port of `test_gui.py`, so verdicts match the desktop
tool. The only structural change: the backbone, ASAB and head are called
separately instead of via `model(spikes)`, which keeps the layer-3 spike train
`(T, B, 128, 16, 16)` available for the spectrogram.

Thresholds, unchanged: `conf ≥ 0.60` → DETECTED, `≥ 0.25` → POSSIBLE TARGET,
below → NO CAMO DETECTED, where `conf = snn_conf × gate_weight` and the gate is
0 for plain backdrops, visible sky, or texture below 300.

## Files

```
app.py              Flask server: static shell + /api/status + /api/detect
inference.py        model bundle, ported detection logic, spike spectrogram
static/index.html   single page
static/styles.css   Echolyx design tokens and components
static/app.js       upload, fetch, canvas spectrogram + rate curve + dial
static/samples/     two bundled frames (one COD10K target, one ordinary photo)
run.bat             launches app.py with the neurocamo venv
```

## Notes

- Checkpoint: `neurocamo/checkpoints/best_model_v2.pth`, overridable with
  `NEUROCAMO_CHECKPOINT`. If the model source has gained parameters the
  checkpoint never trained, the server loads tolerantly and the page shows which
  tensors kept init defaults instead of silently using untrained weights.
- Uploads are capped at 25 MB and the long edge is resized to 1600 px before the
  Laplacian pass; the SNN sees a 128×128 crop either way.
- `debug=False` deliberately: the Flask reloader would load the SNN twice.
