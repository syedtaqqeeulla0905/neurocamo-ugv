/* ==========================================================================
   NEUROCAMO GCS & SNN ANALYTICS SUITE — CLIENT CONTROLLER
   Full-featured SPA for Tactical Camouflage Reconnaissance & Spiking Analytics
   ========================================================================== */

const BAND_COLORS = {
  sparse: [0, 229, 160],     // Emerald
  active: [245, 163, 0],     // Amber
  saturated: [255, 68, 85],  // Crimson
};

const state = {
  activeView: "view-recon",
  samples: [],
  currentSampleId: null,
  currentResult: null,
  spikeFilter: "all",
  batteryCapacityMah: 5200,
  serialPaused: false,
  graphs: [],
  activeGraphIndex: 0,
};

// ── DOM Element Cache ────────────────────────────────────────────────────────
const el = {
  navTabs: document.querySelectorAll(".nav-tab"),
  views: document.querySelectorAll(".view"),
  sysStatusText: document.getElementById("sysStatusText"),
  fileInput: document.getElementById("fileInput"),
  sampleCarousel: document.getElementById("sampleCarousel"),

  // Reconnaissance
  rawImg: document.getElementById("rawImg"),
  rawPlaceholder: document.getElementById("rawPlaceholder"),
  detImg: document.getElementById("detImg"),
  detPlaceholder: document.getElementById("detPlaceholder"),
  rawMeta: document.getElementById("rawMeta"),
  detMeta: document.getElementById("detMeta"),
  reconVerdict: document.getElementById("reconVerdict"),
  reconVerdictSub: document.getElementById("reconVerdictSub"),
  reconConf: document.getElementById("reconConf"),
  reconConfBar: document.getElementById("reconConfBar"),
  reconTex: document.getElementById("reconTex"),
  reconGateReason: document.getElementById("reconGateReason"),
  reconLatency: document.getElementById("reconLatency"),
  btnGoSpikes: document.getElementById("btnGoSpikes"),
  btnGoPower: document.getElementById("btnGoPower"),

  // SNN Core
  specCanvas: document.getElementById("specCanvas"),
  specProbe: document.getElementById("specProbe"),
  specHoverReadout: document.getElementById("specHoverReadout"),
  rateCanvas: document.getElementById("rateCanvas"),
  sparsityVal: document.getElementById("sparsityVal"),
  activeChannelsVal: document.getElementById("activeChannelsVal"),
  zeroMacVal: document.getElementById("zeroMacVal"),
  spikeChips: document.querySelectorAll("#spikeBandFilters .chip"),

  // Power
  pwrSnn: document.getElementById("pwrSnn"),
  pwrAnn: document.getElementById("pwrAnn"),
  battSlider: document.getElementById("battSlider"),
  battCapText: document.getElementById("battCapText"),
  autoSnn: document.getElementById("autoSnn"),
  autoAnn: document.getElementById("autoAnn"),
  graphTabs: document.getElementById("graphTabs"),
  graphTitle: document.getElementById("graphTitle"),
  graphDesc: document.getElementById("graphDesc"),
  graphImg: document.getElementById("graphImg"),

  // Telemetry
  compassCanvas: document.getElementById("compassCanvas"),
  telemetryBearing: document.getElementById("telemetryBearing"),
  telemetryCmd: document.getElementById("telemetryCmd"),
  motorAPwm: document.getElementById("motorAPwm"),
  motorABar: document.getElementById("motorABar"),
  motorBPwm: document.getElementById("motorBPwm"),
  motorBBar: document.getElementById("motorBBar"),
  serialConsole: document.getElementById("serialConsole"),
  btnToggleSerial: document.getElementById("btnToggleSerial"),
  btnClearSerial: document.getElementById("btnClearSerial"),
};

// ── Application Initialization ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupEventListeners();
  drawCompass(0.0);
  drawRateCurve([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

  await loadSystemStatus();
  await loadCuratedSamples();
  await loadPublicationGraphs();
  startTelemetryTicker();
});

// ── 1. Tab Navigation ────────────────────────────────────────────────────────
function setupNavigation() {
  el.navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetView = tab.getAttribute("data-view");
      switchView(targetView);
    });
  });

  el.btnGoSpikes.addEventListener("click", () => switchView("view-spikes"));
  el.btnGoPower.addEventListener("click", () => switchView("view-power"));
}

function switchView(viewId) {
  state.activeView = viewId;

  el.navTabs.forEach((tab) => {
    if (tab.getAttribute("data-view") === viewId) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  el.views.forEach((view) => {
    if (view.id === viewId) {
      view.classList.add("active");
    } else {
      view.classList.remove("active");
    }
  });

  // Re-draw canvases if their dimensions adjusted
  if (viewId === "view-spikes" && state.currentResult) {
    renderSpectrogram(state.currentResult.spectrogram);
    drawRateCurve(state.currentResult.spectrogram.per_step_rate);
  } else if (viewId === "view-telemetry" && state.currentResult) {
    const bearing = state.currentResult.telemetry?.bearing_deg || 0.0;
    drawCompass(bearing);
  }
}

// ── 2. System Status & Samples Loading ───────────────────────────────────────
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ready) {
      el.sysStatusText.textContent = `JETSON ${data.device.toUpperCase()} // READY`;
    }
  } catch (err) {
    el.sysStatusText.textContent = "STANDALONE MODE";
  }
}

async function loadCuratedSamples() {
  try {
    const res = await fetch("/api/samples");
    state.samples = await res.json();

    el.sampleCarousel.innerHTML = "";
    state.samples.forEach((sample, idx) => {
      const pill = document.createElement("button");
      pill.className = "sample-pill" + (idx === 0 ? " active" : "");
      pill.setAttribute("data-id", sample.id);
      pill.innerHTML = `
        <span class="sample-pill-cat">${sample.category}</span>
        <span class="sample-pill-title">${sample.title}</span>
      `;
      pill.addEventListener("click", () => {
        document.querySelectorAll(".sample-pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        triggerSampleInference(sample.id);
      });
      el.sampleCarousel.appendChild(pill);
    });

    // Auto-trigger the first verified sample
    if (state.samples.length > 0) {
      triggerSampleInference(state.samples[0].id);
    }
  } catch (err) {
    console.error("Failed to load samples:", err);
  }
}

async function triggerSampleInference(sampleId) {
  state.currentSampleId = sampleId;
  setProcessingState(true);

  try {
    const res = await fetch(`/api/detect_sample/${sampleId}`);
    const data = await res.json();
    if (data.error) {
      alert("Inference Error: " + data.error);
      return;
    }
    applyInferenceResult(data);
  } catch (err) {
    console.error("Sample detection failed:", err);
  } finally {
    setProcessingState(false);
  }
}

// ── 3. Custom Image Upload ───────────────────────────────────────────────────
function setupEventListeners() {
  el.fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Deselect carousel pills
    document.querySelectorAll(".sample-pill").forEach((p) => p.classList.remove("active"));

    const formData = new FormData();
    formData.append("image", file);

    setProcessingState(true);
    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.error) {
        alert("Upload Error: " + data.error);
        return;
      }
      applyInferenceResult(data);
    } catch (err) {
      alert("Failed to run detection on uploaded image: " + err);
    } finally {
      setProcessingState(false);
    }
  });

  // Spike Band Filters
  el.spikeChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      el.spikeChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.spikeFilter = chip.getAttribute("data-band");
      if (state.currentResult?.spectrogram) {
        renderSpectrogram(state.currentResult.spectrogram);
      }
    });
  });

  // Battery Slider
  el.battSlider.addEventListener("input", (e) => {
    const mah = parseInt(e.target.value, 10);
    state.batteryCapacityMah = mah;
    updateBatteryAutonomy(mah);
  });

  // Serial Console Controls
  el.btnToggleSerial.addEventListener("click", () => {
    state.serialPaused = !state.serialPaused;
    el.btnToggleSerial.textContent = state.serialPaused ? "Resume Feed" : "Pause Feed";
  });

  el.btnClearSerial.addEventListener("click", () => {
    el.serialConsole.innerHTML = "";
  });

  // Spike Spectrogram Hover Probe
  el.specCanvas.addEventListener("mousemove", onSpecHover);
  el.specCanvas.addEventListener("mouseleave", () => {
    el.specProbe.style.display = "none";
    el.specHoverReadout.textContent = "Hover over the spike matrix to inspect channel firing rates.";
  });
}

function setProcessingState(busy) {
  if (busy) {
    el.reconVerdict.textContent = "PROCESSING SNN SPIKES...";
    el.reconVerdict.style.color = "var(--color-cyan)";
  }
}

// ── 4. Apply Detection Results to All Views ──────────────────────────────────
function applyInferenceResult(data) {
  state.currentResult = data;

  // View 1: Reconnaissance
  el.rawImg.src = data.input_png || "";
  el.rawImg.classList.add("visible");
  el.rawPlaceholder.style.display = "none";

  el.detImg.src = data.result_png || "";
  el.detImg.classList.add("visible");
  el.detPlaceholder.style.display = "none";

  const sz = data.image_size || [0, 0];
  el.rawMeta.textContent = `${sz[0]}×${sz[1]} px · ${data.filename || "Frame"}`;

  const confPct = (data.confidence * 100).toFixed(1);
  const detected = data.bbox_drawn;

  if (detected) {
    el.reconVerdict.textContent = "TARGET IDENTIFIED (ENGAGED)";
    el.reconVerdict.style.color = "var(--color-emerald)";
    el.reconVerdictSub.textContent = `Dense mask objectness verified: ${confPct}%`;
  } else {
    el.reconVerdict.textContent = "SECTOR CLEAR (NO CAMOUFLAGE)";
    el.reconVerdict.style.color = "var(--text-muted)";
    el.reconVerdictSub.textContent = data.gate_reason || "No camouflage patterns detected";
  }

  el.reconConf.textContent = `${confPct}%`;
  el.reconConfBar.style.width = `${Math.min(100, data.confidence * 100)}%`;

  el.reconTex.textContent = `${data.texture_score.toFixed(1)}`;
  el.reconGateReason.textContent = `Gate: ${data.plain_background ? "Plain BG" : "Complex Scene"}`;

  el.reconLatency.textContent = `${data.latency_ms.toFixed(1)} ms`;
  const fpsEq = data.latency_ms > 0 ? (1000.0 / data.latency_ms).toFixed(0) : "60";
  el.reconFps.textContent = `Inference rate: ~${fpsEq} FPS`;

  // View 2: Neuromorphic Core
  if (data.spectrogram) {
    renderSpectrogram(data.spectrogram);
    drawRateCurve(data.spectrogram.per_step_rate);
    el.activeChannelsVal.textContent = `${data.spectrogram.active_channels} / 128`;
  }
  const sparsityPct = (data.sparsity * 100).toFixed(1);
  el.sparsityVal.textContent = `${sparsityPct}%`;
  el.zeroMacVal.textContent = `${sparsityPct}%`;

  // View 3: Power Benchmarks
  const snnPwr = data.telemetry?.power_snn_w || 0.91;
  const annPwr = data.telemetry?.power_ann_w || 8.70;
  el.pwrSnn.textContent = `${snnPwr.toFixed(2)} W`;
  el.pwrAnn.textContent = `${annPwr.toFixed(2)} W`;

  // View 4: Telemetry
  const tel = data.telemetry || {};
  const bearing = tel.bearing_deg || 0.0;
  drawCompass(bearing);
  el.telemetryBearing.textContent = (bearing >= 0 ? "+" : "") + bearing.toFixed(1) + "°";
  el.telemetryCmd.textContent = tel.command || "PATROL SCAN";

  const m1 = tel.motor_a || 0;
  const m2 = tel.motor_b || 0;
  el.motorAPwm.textContent = `${m1} / 255`;
  el.motorBPwm.textContent = `${m2} / 255`;
  el.motorABar.style.width = `${(m1 / 255) * 100}%`;
  el.motorBBar.style.width = `${(m2 / 255) * 100}%`;

  // Add serial packet to console
  if (tel.packet) {
    appendSerialLog(tel.packet);
  }
}

// ── 5. Canvas: Layer-3 Spike Spectrogram ─────────────────────────────────────
function renderSpectrogram(spec) {
  const canvas = el.specCanvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const nChannels = spec.n_channels || 128;
  const nSteps = spec.n_steps || 16;
  const data = spec.data; // Quantized 0-255 flat array [row0(t0..t15), row1, ...]

  const cellW = w / nSteps;
  const cellH = h / nChannels;

  const filter = state.spikeFilter;

  for (let c = 0; c < nChannels; c++) {
    for (let t = 0; t < nSteps; t++) {
      const idx = c * nSteps + t;
      const val = data[idx] / 255.0; // 0.0 - 1.0

      if (val < 0.02) continue; // Skip near-zero background

      let band = "sparse";
      if (val > 0.66) band = "saturated";
      else if (val > 0.33) band = "active";

      // Apply band filter
      if (filter !== "all" && filter !== band) {
        ctx.fillStyle = "rgba(40, 48, 64, 0.4)";
        ctx.fillRect(t * cellW, c * cellH, cellW - 0.5, cellH - 0.5);
        continue;
      }

      const rgb = BAND_COLORS[band];
      const alpha = Math.max(0.2, val);
      ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
      ctx.fillRect(t * cellW, c * cellH, cellW - 0.5, cellH - 0.5);
    }
  }

  // Draw Subtle Timestep Dividers
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  for (let t = 0; t <= nSteps; t++) {
    ctx.beginPath();
    ctx.moveTo(t * cellW, 0);
    ctx.lineTo(t * cellW, h);
    ctx.stroke();
  }
}

function onSpecHover(e) {
  if (!state.currentResult?.spectrogram) return;

  const spec = state.currentResult.spectrogram;
  const rect = el.specCanvas.getBoundingClientRect();
  const scaleX = el.specCanvas.width / rect.width;
  const scaleY = el.specCanvas.height / rect.height;

  const mouseX = (e.clientX - rect.left) * scaleX;
  const mouseY = (e.clientY - rect.top) * scaleY;

  const nChannels = spec.n_channels || 128;
  const nSteps = spec.n_steps || 16;

  const stepIdx = Math.floor(mouseX / (el.specCanvas.width / nSteps));
  const chanIdx = Math.floor(mouseY / (el.specCanvas.height / nChannels));

  if (stepIdx >= 0 && stepIdx < nSteps && chanIdx >= 0 && chanIdx < nChannels) {
    const rawVal = spec.data[chanIdx * nSteps + stepIdx];
    const rateNorm = (rawVal / 255.0) * (spec.peak_rate || 1.0);

    const txt = `CH ${chanIdx} | t=${stepIdx} | Rate: ${rateNorm.toFixed(3)} (Peak: ${spec.peak_rate.toFixed(3)})`;
    el.specHoverReadout.textContent = txt;

    el.specProbe.style.display = "block";
    el.specProbe.style.left = `${e.clientX - rect.left + 15}px`;
    el.specProbe.style.top = `${e.clientY - rect.top + 10}px`;
    el.specProbe.textContent = `Ch ${chanIdx} [t${stepIdx}]: ${(rateNorm * 100).toFixed(1)}%`;
  }
}

// ── 6. Canvas: Population Rate Curve (t0 -> t15) ─────────────────────────────
function drawRateCurve(rates) {
  const canvas = el.rateCanvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const padX = 35;
  const padY = 25;
  const drawW = w - padX * 2;
  const drawH = h - padY * 2;

  // Grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, drawW, drawH);

  const maxVal = Math.max(0.1, ...rates);

  // Plot Curve
  ctx.beginPath();
  const stepW = drawW / (rates.length - 1);

  rates.forEach((val, i) => {
    const x = padX + i * stepW;
    const y = padY + drawH - (val / maxVal) * drawH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = "var(--color-cyan)";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Gradient fill under curve
  ctx.lineTo(padX + drawW, padY + drawH);
  ctx.lineTo(padX, padY + drawH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padY, 0, padY + drawH);
  grad.addColorStop(0, "rgba(0, 210, 255, 0.25)");
  grad.addColorStop(1, "rgba(0, 210, 255, 0.0)");
  ctx.fillStyle = grad;
  ctx.fill();

  // Axis Labels
  ctx.fillStyle = "#7b889e";
  ctx.font = "10px monospace";
  ctx.fillText("t0", padX, h - 8);
  ctx.fillText("t15", padX + drawW - 15, h - 8);
  ctx.fillText(`${maxVal.toFixed(2)}`, 6, padY + 10);
}

// ── 7. Canvas: Radial Steering Compass ──────────────────────────────────────
function drawCompass(bearingDeg) {
  const canvas = el.compassCanvas;
  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const r = 90;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Outer ring
  ctx.strokeStyle = "rgba(0, 210, 255, 0.2)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // Inner ring
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.7, 0, Math.PI * 2);
  ctx.stroke();

  // Degree ticks
  for (let a = 0; a < 360; a += 30) {
    const rad = ((a - 90) * Math.PI) / 180;
    const x1 = cx + Math.cos(rad) * (r - 6);
    const y1 = cy + Math.sin(rad) * (r - 6);
    const x2 = cx + Math.cos(rad) * r;
    const y2 = cy + Math.sin(rad) * r;

    ctx.strokeStyle = a % 90 === 0 ? "var(--color-cyan)" : "rgba(255, 255, 255, 0.2)";
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  // Heading Indicators (N, E, S, W)
  ctx.fillStyle = "#7b889e";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("0° (FWD)", cx, cy - r - 12);

  // Target Bearing Needle
  const targetRad = ((bearingDeg - 90) * Math.PI) / 180;
  const nx = cx + Math.cos(targetRad) * (r - 10);
  const ny = cy + Math.sin(targetRad) * (r - 10);

  ctx.strokeStyle = "var(--color-emerald)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.stroke();

  // Target indicator tip
  ctx.fillStyle = "var(--color-emerald)";
  ctx.beginPath();
  ctx.arc(nx, ny, 5, 0, Math.PI * 2);
  ctx.fill();

  // Center Pivot
  ctx.fillStyle = "var(--color-cyan)";
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fill();
}

// ── 8. Battery Autonomy Calculator ───────────────────────────────────────────
function updateBatteryAutonomy(mah) {
  const wh = (mah * 12.0) / 1000.0;
  el.battCapText.textContent = `${mah} mAh (${wh.toFixed(1)} Wh)`;

  // SNN average active power: 0.91W, ANN power: 8.70W
  // Use 85% battery depth-of-discharge usable capacity
  const usableWh = wh * 0.85;
  const snnHrs = (usableWh / 0.91).toFixed(1);
  const annHrs = (usableWh / 8.70).toFixed(1);

  el.autoSnn.textContent = `${snnHrs} hrs`;
  el.autoAnn.textContent = `${annHrs} hrs`;
}

// ── 9. Publication Research Graphs Viewer ────────────────────────────────────
async function loadPublicationGraphs() {
  try {
    const res = await fetch("/api/graphs");
    state.graphs = await res.json();

    el.graphTabs.innerHTML = "";
    state.graphs.forEach((g, idx) => {
      const btn = document.createElement("button");
      btn.className = "graph-tab-btn" + (idx === 0 ? " active" : "");
      btn.textContent = g.title.split(":")[0];
      btn.addEventListener("click", () => {
        document.querySelectorAll(".graph-tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        showGraph(idx);
      });
      el.graphTabs.appendChild(btn);
    });

    if (state.graphs.length > 0) {
      showGraph(0);
    }
  } catch (err) {
    console.error("Failed to load graphs:", err);
  }
}

function showGraph(index) {
  state.activeGraphIndex = index;
  const g = state.graphs[index];
  if (!g) return;

  el.graphTitle.textContent = g.title;
  el.graphDesc.textContent = g.desc;
  el.graphImg.src = `/api/graph_img/${g.category}/${g.file}`;
}

// ── 10. Real-time UGV Serial Telemetry Stream ────────────────────────────────
function appendSerialLog(packet) {
  if (state.serialPaused) return;

  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0];

  const line = document.createElement("div");
  line.className = "console-line text-emerald";
  line.textContent = `[${timeStr}] ${packet}`;

  el.serialConsole.appendChild(line);
  if (el.serialConsole.children.length > 50) {
    el.serialConsole.removeChild(el.serialConsole.children[0]);
  }
  el.serialConsole.scrollTop = el.serialConsole.scrollHeight;
}

function startTelemetryTicker() {
  let seq = 482;
  setInterval(() => {
    if (state.serialPaused || !state.currentResult) return;

    seq++;
    const tel = state.currentResult.telemetry || {};
    const brg = tel.bearing_deg || 0.0;
    const m1 = tel.motor_a || 0;
    const m2 = tel.motor_b || 0;
    const cmd = tel.detected ? "ENGAGE" : "PATROL";
    const pwr = tel.detected ? "0.91W" : "0.35W";

    const packet = `$UGV,SEQ:${seq.toString().padStart(4, "0")},CMD:${cmd},M1:${m1},M2:${m2},BRG:${brg >= 0 ? "+" : ""}${brg.toFixed(1)},PWR:${pwr}*4B`;
    appendSerialLog(packet);
  }, 2000);
}
