/* ============================================================
   NeuroCamo Web — client
   Single page, no build step. Renders the verdict, the two frames,
   and the layer-3 spike spectrogram on a canvas using the same
   banded colormap idea as the Echolyx spectrogram: hue encodes
   which density band a cell falls in, luminance encodes rate.
   ============================================================ */

const BAND_RGB = {
  normal: [0, 229, 160],
  stress: [245, 163, 0],
  fault: [255, 68, 85],
};
const BG_RGB = [8, 8, 11];
const OFF_RGB = [120, 124, 140];

/* band cut points as a fraction of the frame's peak firing rate */
const CUT_STRESS = 0.33;
const CUT_FAULT = 0.66;

const $ = (id) => document.getElementById(id);

const els = {
  pip: $("pip"),
  pipText: $("pipText"),
  deck: $("deck"),
  file: $("file"),
  deckIdle: $("deckIdle"),
  deckLoaded: $("deckLoaded"),
  thumb: $("thumb"),
  fname: $("fname"),
  fmeta: $("fmeta"),
  run: $("run"),
  alertSlot: $("alertSlot"),
  idle: $("idle"),
  results: $("results"),
  dial: $("dial"),
  verdictWord: $("verdictWord"),
  verdictSub: $("verdictSub"),
  readout: $("readout"),
  inImg: $("inImg"),
  outImg: $("outImg"),
  inMeta: $("inMeta"),
  outMeta: $("outMeta"),
  specWrap: $("specWrap"),
  spec: $("spec"),
  probe: $("probe"),
  specMeta: $("specMeta"),
  rate: $("rate"),
  meters: $("meters"),
  gateRows: $("gateRows"),
  legend: $("legend"),
  bandChips: $("bandChips"),
};

const state = {
  file: null,
  objectUrl: null,
  result: null,
  busy: false,
  active: { normal: true, stress: true, fault: true },
  bitmap: null,
  hover: null,
};

/* ── status ────────────────────────────────────────────── */
function setPip(cls, text) {
  els.pip.className = "pip " + cls;
  els.pipText.textContent = text;
}

async function loadStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    $("mCkpt").textContent = s.checkpoint_loaded ? s.checkpoint : "none (random weights)";
    $("mDev").textContent = s.accelerator;
    $("mEpoch").textContent = s.epoch || "—";
    $("mLoss").textContent = s.val_loss ? s.val_loss.toFixed(4) : "—";
    $("mParams").textContent = s.params.toLocaleString() + " params";
    $("mSteps").textContent = "T = " + s.timesteps;
    if (s.checkpoint_loaded) {
      setPip("live", (s.device === "cuda" ? "GPU" : "CPU") + " ready");
    } else {
      setPip("warm", "No checkpoint loaded");
      modelAlert("No checkpoint found — the model is running with random weights.");
    }
    if (s.defaulted_keys && s.defaulted_keys.length) {
      // the model source gained parameters the checkpoint never trained
      setPip("warm", "Partial checkpoint");
      modelAlert(
        "The checkpoint predates the current model definition. Untrained " +
          "(init-default) parameters: <b>" +
          s.defaulted_keys.join(", ") +
          "</b> — verdicts may differ from the trained model."
      );
    }
  } catch (e) {
    setPip("down", "Server unreachable");
  }
}

/* transient per-request errors — cleared on every run */
function alert_(kind, msg) {
  els.alertSlot.innerHTML = msg
    ? `<div class="alert ${kind === "warn" ? "warn" : ""}">${msg}</div>`
    : "";
}

/* persistent model-state warning — survives across runs */
function modelAlert(msg) {
  $("modelAlert").innerHTML = msg ? `<div class="alert warn">${msg}</div>` : "";
}

/* ── file selection ────────────────────────────────────── */
function acceptFile(f) {
  if (!f) return;
  if (!/\.(jpe?g|png|bmp|webp)$/i.test(f.name)) {
    alert_("err", "Unsupported file type. Use JPG, PNG, BMP or WEBP.");
    return;
  }
  if (f.size > 25 * 1024 * 1024) {
    alert_("err", "That file is over the 25 MB limit.");
    return;
  }
  alert_("", "");
  state.file = f;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(f);

  els.thumb.src = state.objectUrl;
  els.fname.textContent = f.name;
  els.fmeta.textContent = (f.size / 1024).toFixed(0) + " KB";
  els.deckIdle.classList.add("hidden");
  els.deckLoaded.classList.remove("hidden");
  els.deck.classList.add("loaded");
  els.run.disabled = false;
}

els.file.addEventListener("change", (e) => acceptFile(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  els.deck.addEventListener(ev, (e) => {
    e.preventDefault();
    els.deck.classList.add("hot");
  })
);
["dragleave", "drop"].forEach((ev) =>
  els.deck.addEventListener(ev, (e) => {
    e.preventDefault();
    els.deck.classList.remove("hot");
  })
);
els.deck.addEventListener("drop", (e) => acceptFile(e.dataTransfer.files[0]));

/* the two bundled sample frames — one COD10K camo target, one ordinary photo */
$("sampleChips").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-sample]");
  if (!btn || state.busy) return;
  const path = btn.dataset.sample;
  try {
    const blob = await (await fetch(path)).blob();
    acceptFile(new File([blob], path.split("/").pop(), { type: blob.type }));
    runDetect();
  } catch (err) {
    alert_("err", "Could not load that sample frame.");
  }
});

/* ── run detection ─────────────────────────────────────── */
els.run.addEventListener("click", runDetect);

async function runDetect() {
  if (!state.file || state.busy) return;
  state.busy = true;
  els.run.disabled = true;
  els.run.textContent = "Encoding spikes…";
  els.deck.classList.add("scan");
  setPip("warm", "Running 16-timestep pass");
  alert_("", "");

  const body = new FormData();
  body.append("image", state.file);

  try {
    const r = await fetch("/api/detect", { method: "POST", body });
    const text = await r.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (parseErr) {
      throw new Error(`Server error (${r.status}). Check backend console log.`);
    }

    if (!r.ok) throw new Error(data.error || "Request failed (" + r.status + ")");
    state.result = data;
    els.inImg.src = state.objectUrl;
    render(data);
    setPip("live", "Pass complete · " + data.latency_ms.toFixed(0) + " ms");
  } catch (e) {
    alert_("err", e.message);
    setPip("down", "Inference failed");
  } finally {
    state.busy = false;
    state.file && (els.run.disabled = false);
    els.run.textContent = "Run detection";
    els.deck.classList.remove("scan");
  }
}

/* ── Phone IP Camera Live Stream Handling ───────────────── */
const srcFileBtn = $("srcFileBtn");
const srcIpBtn = $("srcIpBtn");
const fileSourceGroup = $("fileSourceGroup");
const ipSourceGroup = $("ipSourceGroup");
const ipUrlInput = $("ipUrlInput");
const toggleStreamBtn = $("toggleStreamBtn");

let isStreaming = false;
let streamTimer = null;

srcFileBtn.addEventListener("click", () => {
  srcFileBtn.classList.add("active");
  srcIpBtn.classList.remove("active");
  fileSourceGroup.classList.remove("hidden");
  ipSourceGroup.classList.add("hidden");
  els.run.classList.remove("hidden");
  stopStream();
});

srcIpBtn.addEventListener("click", () => {
  srcIpBtn.classList.add("active");
  srcFileBtn.classList.remove("active");
  fileSourceGroup.classList.add("hidden");
  ipSourceGroup.classList.remove("hidden");
  els.run.classList.add("hidden");
});

// Presets
$("presetIpWebcam").addEventListener("click", () => {
  let val = ipUrlInput.value.trim() || "http://192.168.1.50:8080";
  val = val.replace(/\/(video|shot\.jpg|photo\.jpg)?$/, "");
  ipUrlInput.value = val + "/shot.jpg";
});

$("presetIpVideo").addEventListener("click", () => {
  let val = ipUrlInput.value.trim() || "http://192.168.1.50:8080";
  val = val.replace(/\/(video|shot\.jpg|photo\.jpg)?$/, "");
  ipUrlInput.value = val + "/video";
});

$("presetDroidCam").addEventListener("click", () => {
  let val = ipUrlInput.value.trim() || "http://192.168.1.50:4747";
  val = val.replace(/\/(video|shot\.jpg|photo\.jpg)?$/, "");
  ipUrlInput.value = val + "/video";
});

const detectFrameBtn = $("detectFrameBtn");

toggleStreamBtn.addEventListener("click", () => {
  if (isStreaming) {
    stopStream();
  } else {
    startStream();
  }
});

detectFrameBtn.addEventListener("click", runStreamDetection);

function setLiveOnlyMode(isLiveOnly) {
  const verdictPanel = $("verdictPanel");
  const detectionPanel = $("detectionPanel");
  const specPanel = $("specPanel");
  const bottomMetricsGrid = $("bottomMetricsGrid");

  if (isLiveOnly) {
    els.results.classList.add("live-only-full");
    if (verdictPanel) verdictPanel.classList.add("hidden");
    if (detectionPanel) detectionPanel.classList.add("hidden");
    if (specPanel) specPanel.classList.add("hidden");
    if (bottomMetricsGrid) bottomMetricsGrid.classList.add("hidden");
  } else {
    els.results.classList.remove("live-only-full");
    if (verdictPanel) verdictPanel.classList.remove("hidden");
    if (detectionPanel) detectionPanel.classList.remove("hidden");
    if (specPanel) specPanel.classList.remove("hidden");
    if (bottomMetricsGrid) bottomMetricsGrid.classList.remove("hidden");
  }
}

function startStream() {
  let url = ipUrlInput.value.trim();
  if (!url) {
    alert_("err", "Please enter your Phone's IP Camera URL.");
    return;
  }

  // Auto-format IP Webcam stream URL if user entered root address
  if (url.startsWith("http")) {
    if (!anyExt(url)) {
      if (!url.endsWith("/")) url += "/";
      url += "video";
    }
  }

  const feedUrl = "/api/video_feed?url=" + encodeURIComponent(url);

  isStreaming = true;
  toggleStreamBtn.textContent = "Stop Live Camera Feed";
  toggleStreamBtn.classList.add("streaming");
  detectFrameBtn.classList.remove("hidden");
  setPip("live", "LIVE Phone Camera (30 FPS)");
  alert_("", "");

  els.inImg.onerror = () => {
    alert_("err", `Could not connect to camera at '${url}'. Make sure <b>IP Webcam</b> app is open on your phone, <b>'Start server'</b> is tapped, and both phone & PC are on the same Wi-Fi network.`);
    stopStream();
  };

  // Reveal stage, hide metrics/graphs, and show pure live video stream
  els.idle.classList.add("hidden");
  els.results.classList.remove("hidden");
  setLiveOnlyMode(true);
  els.inImg.src = feedUrl;
  els.inMeta.textContent = "Live Stream (30 FPS)";
}

function anyExt(u) {
  return [".jpg", ".jpeg", ".png", "/shot.jpg", "/video", "/photo.jpg", "/mempkg"].some(ext => u.toLowerCase().endsWith(ext));
}

function stopStream() {
  isStreaming = false;
  toggleStreamBtn.textContent = "Connect Live Camera Feed";
  toggleStreamBtn.classList.remove("streaming");
  detectFrameBtn.classList.add("hidden");
  setPip("live", "Stream Stopped");
  els.inImg.src = "";
  els.outImg.src = "";
  setLiveOnlyMode(false);
}

async function runStreamDetection() {
  let url = ipUrlInput.value.trim();
  if (!url) return;

  detectFrameBtn.disabled = true;
  detectFrameBtn.textContent = "Running SNN Inference...";
  setPip("warm", "Running SNN Pass on Live Frame");
  alert_("", "");

  try {
    const r = await fetch("/api/detect_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream_url: url })
    });
    
    const text = await r.text();
    let data = {};
    try {
      data = JSON.parse(text);
    } catch (parseErr) {
      data = { error: `Server error (${r.status})` };
    }

    if (!r.ok) {
      throw new Error(data.error || `HTTP ${r.status} error from server`);
    }

    state.result = data;
    if (data.input_png) els.inImg.src = data.input_png;
    render(data);
    setPip("live", "Pass complete · " + data.latency_ms.toFixed(0) + " ms");
  } catch (e) {
    alert_("err", e.message);
    setPip("down", "Stream Connection Error");
  } finally {
    detectFrameBtn.disabled = false;
    detectFrameBtn.textContent = "Run SNN Detection on Current Frame";
  }
}

/* ── render ────────────────────────────────────────────── */
function pct(v) {
  return (v * 100).toFixed(1) + "%";
}

function render(d) {
  els.idle.classList.add("hidden");
  els.results.classList.remove("hidden");
  setLiveOnlyMode(false);

  els.verdictWord.textContent = d.verdict;
  els.verdictWord.className = "verdict-word b-" + d.band;
  els.verdictSub.textContent = d.gate_reason;

  els.readout.innerHTML = [
    ["Confidence", pct(d.confidence), "gated"],
    ["Raw SNN", pct(d.snn_confidence), "pre-gate"],
    ["Texture", d.texture_score.toFixed(0), "laplacian var"],
    ["Sparsity", pct(d.sparsity), "silent neurons"],
    ["Latency", d.latency_ms.toFixed(0) + " ms", "encode + 16 steps"],
    ["Active ch.", d.spectrogram.active_channels + " / " + d.spectrogram.n_channels, "rate > 1%"],
  ]
    .map(
      ([k, v, s]) =>
        `<div><dt>${k}</dt><dd>${v}<br /><small>${s}</small></dd></div>`
    )
    .join("");

  drawDial(d);

  els.inImg.src = state.objectUrl;
  els.outImg.src = d.result_png;
  els.inMeta.textContent = d.image_size[0] + " × " + d.image_size[1];
  els.outMeta.textContent = d.bbox_drawn
    ? "box " + d.bbox_px.join(", ")
    : "no box — below 25% gate";

  els.specMeta.textContent =
    d.spectrogram.n_channels +
    " ch × " +
    d.spectrogram.n_steps +
    " steps · peak " +
    pct(d.spectrogram.peak_rate);

  buildBitmap(d.spectrogram);
  drawSpec();
  drawRate(d.spectrogram.per_step_rate);
  drawGate(d);
}

/* ── confidence dial ───────────────────────────────────── */
function drawDial(d) {
  const c = els.dial;
  const ctx = c.getContext("2d");
  const S = c.width;
  const cx = S / 2;
  const cy = S / 2;
  const R = S * 0.4;
  const lw = S * 0.075;
  const A0 = Math.PI * 0.75;
  const SPAN = Math.PI * 1.5;
  const [r, g, b] = BAND_RGB[d.band];

  ctx.clearRect(0, 0, S, S);

  ctx.lineCap = "round";
  ctx.lineWidth = lw;
  ctx.strokeStyle = "#13131a";
  ctx.beginPath();
  ctx.arc(cx, cy, R, A0, A0 + SPAN);
  ctx.stroke();

  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.beginPath();
  ctx.arc(cx, cy, R, A0, A0 + SPAN * Math.max(0.004, d.confidence));
  ctx.stroke();

  // threshold ticks at the two decision boundaries
  ctx.lineWidth = 2;
  for (const t of [d.thresholds.possible, d.thresholds.detected]) {
    const a = A0 + SPAN * t;
    ctx.strokeStyle = "rgba(238,238,242,0.35)";
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * (R - lw / 2), cy + Math.sin(a) * (R - lw / 2));
    ctx.lineTo(cx + Math.cos(a) * (R + lw / 2), cy + Math.sin(a) * (R + lw / 2));
    ctx.stroke();
  }

  ctx.fillStyle = "#eeeef2";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `700 ${S * 0.19}px Outfit, sans-serif`;
  ctx.fillText((d.confidence * 100).toFixed(0) + "%", cx, cy - S * 0.02);
  ctx.fillStyle = "#5e5e6e";
  ctx.font = `${S * 0.065}px "DM Mono", monospace`;
  ctx.fillText("CONFIDENCE", cx, cy + S * 0.13);
}

/* ── spike spectrogram ─────────────────────────────────── */
const PAD_L = 52;
const PAD_B = 20;
const PAD_T = 8;
const PAD_R = 8;
const PLOT_H = 260;

function bandOfValue(v) {
  if (v >= CUT_FAULT) return "fault";
  if (v >= CUT_STRESS) return "stress";
  return "normal";
}

/* Native-resolution heatmap (n_steps x n_channels), scaled by the GPU on draw
   so paint cost stays flat regardless of display size. */
function buildBitmap(spec) {
  const { n_channels, n_steps, data } = spec;
  const off = document.createElement("canvas");
  off.width = n_steps;
  off.height = n_channels;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(n_steps, n_channels);

  for (let row = 0; row < n_channels; row++) {
    // channel 0 at the bottom, like a frequency axis
    const y = n_channels - 1 - row;
    for (let col = 0; col < n_steps; col++) {
      const v = data[row * n_steps + col] / 255;
      const band = bandOfValue(v);
      const on = state.active[band];
      const [br, bg, bb] = on ? BAND_RGB[band] : OFF_RGB;
      // mild gamma: layer-3 rates cluster low, a linear ramp reads almost black
      const t = Math.pow(v, 0.85);
      const hot = Math.max(0, t - 0.85) / 0.15;
      const i = (y * n_steps + col) * 4;
      img.data[i] = BG_RGB[0] + (br - BG_RGB[0]) * t + (255 - br) * hot * 0.7;
      img.data[i + 1] = BG_RGB[1] + (bg - BG_RGB[1]) * t + (255 - bg) * hot * 0.7;
      img.data[i + 2] = BG_RGB[2] + (bb - BG_RGB[2]) * t + (255 - bb) * hot * 0.7;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  state.bitmap = off;
}

function drawSpec() {
  if (!state.bitmap || !state.result) return;
  const spec = state.result.spectrogram;
  const canvas = els.spec;
  const width = Math.max(360, Math.round(els.specWrap.clientWidth));
  const height = PLOT_H + PAD_T + PAD_B;
  const plotW = width - PAD_L - PAD_R;
  const dpr = Math.min(2, window.devicePixelRatio || 1);

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = "100%";
  canvas.style.height = height + "px";

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(state.bitmap, PAD_L, PAD_T, plotW, PLOT_H);

  ctx.font = '9px "DM Mono", monospace';
  ctx.textBaseline = "middle";

  // channel axis
  ctx.fillStyle = "#5e5e6e";
  ctx.textAlign = "right";
  const step = Math.max(1, Math.round(spec.n_channels / 4));
  for (let ch = 0; ch < spec.n_channels; ch += step) {
    const y = PAD_T + PLOT_H - (ch / (spec.n_channels - 1)) * PLOT_H;
    ctx.fillText("ch " + ch, PAD_L - 6, y);
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD_L, Math.round(y) + 0.5);
    ctx.lineTo(PAD_L + plotW, Math.round(y) + 0.5);
    ctx.stroke();
  }

  // timestep axis
  ctx.fillStyle = "#5e5e6e";
  ctx.textAlign = "center";
  for (let t = 0; t < spec.n_steps; t += Math.max(1, Math.round(spec.n_steps / 8))) {
    const x = PAD_L + ((t + 0.5) / spec.n_steps) * plotW;
    ctx.fillText("t" + t, x, PAD_T + PLOT_H + 10);
  }

  if (state.hover) {
    ctx.strokeStyle = "rgba(238,238,242,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(state.hover.x, PAD_T);
    ctx.lineTo(state.hover.x, PAD_T + PLOT_H);
    ctx.moveTo(PAD_L, state.hover.y);
    ctx.lineTo(PAD_L + plotW, state.hover.y);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.strokeRect(PAD_L + 0.5, PAD_T + 0.5, plotW - 1, PLOT_H - 1);
  state.specGeom = { width, height, plotW };
}

/* hover probe over the spectrogram */
els.spec.addEventListener("mousemove", (e) => {
  if (!state.result || !state.specGeom) return;
  const spec = state.result.spectrogram;
  const { width, height, plotW } = state.specGeom;
  const rect = els.spec.getBoundingClientRect();
  const scale = width / rect.width;
  const x = (e.clientX - rect.left) * scale;
  const y = (e.clientY - rect.top) * scale;

  if (x < PAD_L || x > PAD_L + plotW || y < PAD_T || y > PAD_T + PLOT_H) {
    clearProbe();
    return;
  }

  const col = Math.min(
    spec.n_steps - 1,
    Math.max(0, Math.floor(((x - PAD_L) / plotW) * spec.n_steps))
  );
  const ch = Math.min(
    spec.n_channels - 1,
    Math.max(0, Math.round(((PAD_T + PLOT_H - y) / PLOT_H) * (spec.n_channels - 1)))
  );
  const rel = spec.data[ch * spec.n_steps + col] / 255;
  const rate = rel * spec.peak_rate;

  state.hover = { x, y };
  drawSpec();

  els.probe.classList.remove("hidden");
  els.probe.innerHTML =
    `<b>ch ${ch}</b> · t${col}<br />rate ${(rate * 100).toFixed(1)}% · ` +
    bandOfValue(rel).toUpperCase();
  const left = Math.min(Math.max(x + 10, PAD_L), width - 150);
  els.probe.style.left = (left / width) * 100 + "%";
  els.probe.style.top = ((y + 10) / height) * 100 + "%";
});

els.spec.addEventListener("mouseleave", clearProbe);

function clearProbe() {
  if (!state.hover) return;
  state.hover = null;
  els.probe.classList.add("hidden");
  drawSpec();
}

/* ── population rate curve ─────────────────────────────── */
function drawRate(series) {
  const canvas = els.rate;
  const wrap = canvas.parentElement;
  const width = Math.max(280, Math.round(wrap.clientWidth));
  const height = 150;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const L = 44,
    R = 10,
    T = 12,
    B = 20;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = "100%";
  canvas.style.height = height + "px";

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const plotW = width - L - R;
  const plotH = height - T - B;
  const peak = Math.max(...series, 1e-6);
  const xOf = (i) => L + (series.length === 1 ? 0.5 : i / (series.length - 1)) * plotW;
  const yOf = (v) => T + plotH - (v / peak) * plotH;

  ctx.font = '9px "DM Mono", monospace';
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (let k = 0; k <= 2; k++) {
    const v = (peak * k) / 2;
    const y = yOf(v);
    ctx.fillStyle = "#5e5e6e";
    ctx.fillText((v * 100).toFixed(1) + "%", L - 6, y);
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.beginPath();
    ctx.moveTo(L, Math.round(y) + 0.5);
    ctx.lineTo(L + plotW, Math.round(y) + 0.5);
    ctx.stroke();
  }

  // area under the curve, then the stroke, then per-step dots
  ctx.beginPath();
  ctx.moveTo(xOf(0), T + plotH);
  series.forEach((v, i) => ctx.lineTo(xOf(i), yOf(v)));
  ctx.lineTo(xOf(series.length - 1), T + plotH);
  ctx.closePath();
  ctx.fillStyle = "rgba(0,229,160,0.10)";
  ctx.fill();

  ctx.beginPath();
  series.forEach((v, i) => (i ? ctx.lineTo(xOf(i), yOf(v)) : ctx.moveTo(xOf(i), yOf(v))));
  ctx.strokeStyle = "#00e5a0";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  series.forEach((v, i) => {
    ctx.beginPath();
    ctx.arc(xOf(i), yOf(v), 2, 0, Math.PI * 2);
    ctx.fillStyle = "#00e5a0";
    ctx.fill();
  });

  ctx.fillStyle = "#5e5e6e";
  ctx.textAlign = "center";
  for (let i = 0; i < series.length; i += Math.max(1, Math.round(series.length / 8))) {
    ctx.fillText("t" + i, xOf(i), T + plotH + 10);
  }

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.strokeRect(L + 0.5, T + 0.5, plotW - 1, plotH - 1);
}

/* ── scene gate panel ──────────────────────────────────── */
function drawGate(d) {
  const texMax = 1200;
  const rows = [
    {
      name: "Texture",
      band: "stress",
      frac: Math.min(1, d.texture_score / texMax),
      tick: d.thresholds.texture_gate / texMax,
      tickLabel: "gate 300",
      val: d.texture_score.toFixed(0),
      sub: "laplacian var",
    },
    {
      name: "Raw SNN",
      band: "normal",
      frac: d.snn_confidence,
      tick: d.thresholds.detected,
      tickLabel: "0.60",
      val: pct(d.snn_confidence),
      sub: "objectness",
    },
    {
      name: "Sparsity",
      band: "fault",
      frac: d.sparsity,
      tick: 0.85,
      tickLabel: "target 85%",
      val: pct(d.sparsity),
      sub: "silent neurons",
    },
  ];

  els.meters.innerHTML = rows
    .map(
      (r) => `
      <div class="meter-row" style="color: var(--band-${r.band})">
        <span class="meter-name">${r.name}</span>
        <span class="meter-track">
          <span class="meter-fill" style="width:${(r.frac * 100).toFixed(1)}%;background:currentColor"></span>
          <span class="meter-tick" style="left:${(r.tick * 100).toFixed(1)}%" data-label="${r.tickLabel}"></span>
        </span>
        <span class="meter-val">${r.val}<small>${r.sub}</small></span>
      </div>`
    )
    .join("");

  const checks = [
    [
      "Plain backdrop",
      d.plain_background ? "top corners uniform" : "top corners varied",
      !d.plain_background,
    ],
    ["Open sky", d.open_sky ? "bright + smooth top strip" : "no sky signature", !d.open_sky],
    [
      "Texture ≥ 300",
      d.texture_score.toFixed(0),
      d.texture_score >= d.thresholds.texture_gate,
    ],
    [
      "Gate weight",
      d.texture_weight.toFixed(1) + " ×",
      d.texture_weight > 0,
    ],
  ];

  els.gateRows.innerHTML = checks
    .map(
      ([k, v, pass]) => `
      <tr${pass ? "" : ' class="picked"'}>
        <td>${k}</td><td>${v}</td>
        <td class="${pass ? "v-pass" : "v-block"}">${pass ? "PASS" : "BLOCK"}</td>
      </tr>`
    )
    .join("");
}

/* ── band toggles ──────────────────────────────────────── */
function toggleBand(band) {
  state.active[band] = !state.active[band];
  document
    .querySelectorAll(`[data-band="${band}"]`)
    .forEach((el) => el.setAttribute("aria-pressed", String(state.active[band])));
  if (state.result) {
    buildBitmap(state.result.spectrogram);
    drawSpec();
  }
}

[els.legend, els.bandChips].forEach((host) =>
  host.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-band]");
    if (btn) toggleBand(btn.dataset.band);
  })
);

/* ── resize ────────────────────────────────────────────── */
let rt = null;
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    if (!state.result) return;
    drawSpec();
    drawRate(state.result.spectrogram.per_step_rate);
  }, 120);
});

loadStatus();
