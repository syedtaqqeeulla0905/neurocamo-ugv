/* ==========================================================================
   NEUROCAMO GCS & SNN ANALYTICS SUITE — CLIENT CONTROLLER
   Live WebSocket Hardware Telemetry + Apache ECharts + Multi-Session Profiler
   ========================================================================== */

const BAND_COLORS = {
  sparse: [0, 229, 160],     // Emerald
  active: [245, 163, 0],     // Amber
  saturated: [255, 68, 85],  // Crimson
};

const PHASE_LABELS = {
  idle_baseline: { text: "IDLE BASELINE", cls: "idle" },
  snn_inference: { text: "NEUROCAMO SNN", cls: "snn" },
  cooldown: { text: "COOLDOWN", cls: "cooldown" },
  yolo_inference: { text: "YOLOV8-NANO", cls: "yolo" },
  post_idle: { text: "POST IDLE", cls: "idle" },
  complete: { text: "COMPLETE", cls: "complete" },
  standby: { text: "STANDBY", cls: "idle" },
  error: { text: "ERROR", cls: "yolo" },
};

const state = {
  activeView: "view-recon",
  samples: [],
  currentSampleId: "camo_marine",
  currentResult: null,
  spikeFilter: "all",
  yoloConf: 0.15,
  envFilter: "normal",
  streamQuality: "high",
  temporalStep: 16,
  temporalTimer: null,
  isPlayingTemporal: false,
  currentOverlay: "box", // "box", "saliency", "mask"
  isWebcamActive: false,
  isRealSenseActive: false,
  webcamStream: null,
  webcamInterval: null,
  rsTimeout: null,
  isProcessingCamFrame: false,
  batteryCapacityMah: 5200,

  // Real hardware telemetry & multi-session state (zero fake/seeded data)
  ws: null,
  wsConnected: false,
  wsReconnectDelayMs: 500,
  wsReconnectTimer: null,
  miniCollapsed: false,
  echartsAvailable: false,
  echartsMain: null,
  echartsMini: null,

  sessions: [],        // Ordered newest-first summary objects
  sessionMap: {},      // session_id -> { summary, samples: [], sampleKeys: Set, active: bool, currentPhase: str }
  activeSessionId: null,
  viewedSessionId: null,

  sensorInfo: {
    source: null,
    is_estimated: false,
    sensor_detail: "Probing hardware rail...",
  },
  latestMeasuredSnnW: null,
  latestMeasuredYoloW: null,

  // Real completed-session history arrays for latency oscilloscope
  latencyHistory: [],
  yoloLatencyHistory: [],
  currentEnergy: null,
};

// ── Safe DOM Helpers ─────────────────────────────────────────────────────────
function getEl(id) {
  return document.getElementById(id);
}

function setTxt(id, val) {
  const node = typeof id === "string" ? document.getElementById(id) : id;
  if (node) {
    node.textContent = val;
  }
}

function setStyle(id, prop, val) {
  const node = typeof id === "string" ? document.getElementById(id) : id;
  if (node && node.style) {
    node.style[prop] = val;
  }
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Application Initialization ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupEventListeners();
  initPowerCharts();

  drawRateCurve([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  drawLatencyOscilloscope([], []);
  drawDynamicEnergyChart(null);
  updateBatteryAutonomy(state.batteryCapacityMah);

  // Connect WebSocket FIRST and wait briefly so initial sample streams live from t=0.0s
  await connectPowerWebSocket({ waitOpenMs: 450 });
  await loadSystemStatus();
  await loadLiveSensorBadge();
  await loadSessionHistory();
  await loadCuratedSamples();
});

// ── 1. Tab & Dropdown Mode Navigation ────────────────────────────────────────
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetView = tab.getAttribute("data-view");
      switchView(targetView);
    });
  });

  const modeSelect = getEl("modeSelect");
  if (modeSelect) {
    modeSelect.addEventListener("change", (e) => {
      switchView(e.target.value);
    });
  }

  const btnGoCompare = getEl("btnGoCompare");
  if (btnGoCompare) btnGoCompare.addEventListener("click", () => switchView("view-compare"));

  const btnGoSpikes = getEl("btnGoSpikes");
  if (btnGoSpikes) btnGoSpikes.addEventListener("click", () => switchView("view-spikes"));

  const btnGoPower = getEl("btnGoPower");
  if (btnGoPower) btnGoPower.addEventListener("click", () => switchView("view-power"));

  const btnMiniToPower = getEl("btnMiniToPower");
  if (btnMiniToPower) btnMiniToPower.addEventListener("click", () => switchView("view-power"));

  const btnToggleMiniChart = getEl("btnToggleMiniChart");
  if (btnToggleMiniChart) {
    btnToggleMiniChart.addEventListener("click", () => {
      setMiniStripCollapsed(!state.miniCollapsed);
    });
  }
}

function setMiniStripCollapsed(collapsed) {
  state.miniCollapsed = Boolean(collapsed);
  setStyle("livePowerStripBody", "display", state.miniCollapsed ? "none" : "block");
  setTxt("btnToggleMiniChart", state.miniCollapsed ? "Show Waveform" : "Hide");
  if (!state.miniCollapsed && state.echartsMini) {
    requestAnimationFrame(() => {
      state.echartsMini.resize();
      renderViewedPowerCharts();
    });
  }
}

function switchView(viewId) {
  state.activeView = viewId;

  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.getAttribute("data-view") === viewId);
  });

  const modeSelect = getEl("modeSelect");
  if (modeSelect && (viewId === "view-recon" || viewId === "view-compare")) {
    modeSelect.value = viewId;
  }

  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });

  // Hide compact live strip in view-power because view-power hosts the full-size ECharts profiler
  setStyle("livePowerStrip", "display", viewId === "view-power" ? "none" : "block");

  requestAnimationFrame(() => {
    if (viewId === "view-power") {
      if (state.echartsMain) state.echartsMain.resize();
      renderViewedPowerCharts();
      drawLatencyOscilloscope(state.latencyHistory, state.yoloLatencyHistory);
      drawDynamicEnergyChart(state.currentEnergy);
    } else {
      if (state.echartsMini && !state.miniCollapsed) state.echartsMini.resize();
      renderViewedPowerCharts();
    }

    if (viewId === "view-spikes" && state.currentResult) {
      renderSpectrogram(state.currentResult.spectrogram);
      if (state.currentResult.spectrogram?.per_step_rate) {
        drawRateCurve(state.currentResult.spectrogram.per_step_rate);
      }
    } else if ((viewId === "view-compare" || viewId === "view-recon") && state.currentResult) {
      applyInferenceResult(state.currentResult, { skipSessionMerge: true });
    }
  });
}

// ── 2. Hardware Source & Phase Badge Formatters ──────────────────────────────
function formatSourceTruth(source, isEstimated, sensorDetail) {
  const src = String(source || "").trim();
  const isEst =
    Boolean(isEstimated) ||
    !src ||
    src.toLowerCase().includes("psutil") ||
    src.toLowerCase().includes("estimated");
  if (!src) {
    return {
      label: "SENSOR: PROBING...",
      shortLabel: "PROBING...",
      isEstimated: false,
      cls: "",
      detail: sensorDetail || "Probing hardware telemetry source...",
    };
  }
  if (isEst) {
    return {
      label: "ESTIMATED: psutil CPU Telemetry",
      shortLabel: "ESTIMATED: psutil",
      isEstimated: true,
      cls: "estimated",
      detail: sensorDetail || "Estimated from real psutil CPU & memory counters (no direct rail sensor)",
    };
  }
  return {
    label: `MEASURED: ${src}`,
    shortLabel: `MEASURED: ${src}`,
    isEstimated: false,
    cls: "measured",
    detail: sensorDetail || `Direct hardware rail telemetry via ${src}`,
  };
}

function updateSourceBadges(source, isEstimated, sensorDetail) {
  if (source) {
    state.sensorInfo = {
      source,
      is_estimated: Boolean(isEstimated),
      sensor_detail: sensorDetail || "",
    };
  }
  const info = formatSourceTruth(
    state.sensorInfo.source,
    state.sensorInfo.is_estimated,
    state.sensorInfo.sensor_detail
  );

  const badges = [
    { wrap: "hwSourceBadge", txt: "hwSourceText" },
    { wrap: "miniSourceBadge", txt: "miniSourceBadge" },
    { wrap: "mainSourceBadge", txt: "mainSourceBadge" },
    { wrap: "powerViewSourceBadge", txt: "powerViewSourceBadge" },
  ];
  badges.forEach(({ wrap, txt }) => {
    const wrapEl = getEl(wrap);
    if (wrapEl) {
      wrapEl.classList.remove("measured", "estimated");
      if (info.cls) wrapEl.classList.add(info.cls);
      wrapEl.title = info.detail;
    }
    setTxt(txt, info.label);
  });

  if (state.sensorInfo.sensor_detail) {
    setTxt("pwrSensorFooter", `${info.shortLabel} (${state.sensorInfo.sensor_detail})`);
  } else {
    setTxt("pwrSensorFooter", info.label);
  }
}

function updatePhaseBadges(phaseKey) {
  const cfg = PHASE_LABELS[phaseKey] || { text: String(phaseKey || "STANDBY").toUpperCase(), cls: "idle" };
  ["miniPhaseBadge", "mainPhaseBadge", "powerViewPhaseBadge"].forEach((id) => {
    const el = getEl(id);
    if (el) {
      el.className = `phase-badge ${cfg.cls}`;
      el.textContent = cfg.text;
    }
  });
}

function updateWsBadge(statusClass, text) {
  const badge = getEl("wsLinkBadge");
  if (badge) {
    badge.classList.remove("connected", "connecting", "disconnected");
    badge.classList.add(statusClass);
  }
  setTxt("wsLinkText", text);
}

// ── 3. WebSocket Client (/ws/power) ──────────────────────────────────────────
function connectPowerWebSocket(options = {}) {
  const waitOpenMs = Number(options.waitOpenMs || 0);
  if (
    state.ws &&
    (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)
  ) {
    return Promise.resolve(state.ws.readyState === WebSocket.OPEN);
  }

  if (state.wsReconnectTimer) {
    clearTimeout(state.wsReconnectTimer);
    state.wsReconnectTimer = null;
  }

  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${window.location.host}/ws/power`;
  updateWsBadge("connecting", "WS: CONNECTING");

  return new Promise((resolve) => {
    let settled = false;
    let openWaitTimer = null;
    const finishWait = (ok) => {
      if (settled) return;
      settled = true;
      if (openWaitTimer) clearTimeout(openWaitTimer);
      resolve(ok);
    };

    if (waitOpenMs > 0) {
      openWaitTimer = setTimeout(() => finishWait(false), waitOpenMs);
    } else {
      finishWait(false);
    }

    let ws;
    try {
      ws = new WebSocket(wsUrl);
    } catch (err) {
      console.warn("WebSocket initialization failed:", err);
      updateWsBadge("disconnected", "WS: UNAVAILABLE");
      scheduleWsReconnect();
      finishWait(false);
      return;
    }

    state.ws = ws;

    ws.onopen = () => {
      state.wsConnected = true;
      state.wsReconnectDelayMs = 500;
      updateWsBadge("connected", "WS: LIVE");
      finishWait(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
      } catch (err) {
        console.warn("Invalid WebSocket payload:", err);
      }
    };

    ws.onerror = () => {
      state.wsConnected = false;
      updateWsBadge("disconnected", "WS: RECONNECTING");
    };

    ws.onclose = () => {
      state.wsConnected = false;
      if (state.ws === ws) {
        state.ws = null;
      }
      updateWsBadge("disconnected", "WS: RECONNECTING");
      finishWait(false);
      scheduleWsReconnect();
    };
  });
}

function scheduleWsReconnect() {
  if (state.wsReconnectTimer) return;
  const delay = state.wsReconnectDelayMs;
  state.wsReconnectDelayMs = Math.min(5000, Math.round(state.wsReconnectDelayMs * 1.6));
  state.wsReconnectTimer = setTimeout(() => {
    state.wsReconnectTimer = null;
    connectPowerWebSocket();
  }, delay);
}

function handleWsMessage(msg) {
  if (!msg || !msg.type) return;

  switch (msg.type) {
    case "hello": {
      const caps = msg.capabilities || {};
      const latest = msg.latest || {};
      const src = msg.source || caps.source || latest.source;
      const isEst =
        msg.is_estimated !== undefined
          ? msg.is_estimated
          : caps.is_estimated !== undefined
            ? caps.is_estimated
            : latest.is_estimated;
      const detail = msg.sensor_detail || caps.sensor_detail || latest.sensor_detail;
      if (src) {
        updateSourceBadges(src, isEst, detail);
      }
      if (Number.isFinite(Number(latest.watts))) {
        setStyle("sysPowerStatus", "display", "inline");
        setTxt("globalSnnPower", `${Number(latest.watts).toFixed(2)} W`);
        setTxt("miniLiveWatts", `${Number(latest.watts).toFixed(2)} W`);
        setTxt("liveRailNowVal", `${Number(latest.watts).toFixed(2)} W`);
      }
      const act =
        msg.active_session ||
        (msg.active_session_id
          ? {
              session_id: msg.active_session_id,
              image_name: msg.image_name,
              started_at_utc: msg.started_at_utc,
            }
          : null);
      if (act && act.session_id && !state.sessionMap[act.session_id]) {
        ensureSessionRecord(act.session_id, {
          session_id: act.session_id,
          image_name: act.image_name || "Active Benchmark",
          started_at_utc: act.started_at_utc || new Date().toISOString(),
          source: src,
          is_estimated: isEst,
          sensor_detail: detail,
          active: true,
        });
      }
      break;
    }

    case "session_start":
      onWsSessionStart(msg);
      break;

    case "phase_change":
      onWsPhaseChange(msg);
      break;

    case "power_sample":
      onWsPowerSample(msg);
      break;

    case "session_complete":
      onWsSessionComplete(msg);
      break;

    case "error":
      onWsError(msg);
      break;

    default:
      break;
  }
}

function ensureSessionRecord(sessionId, meta = {}) {
  if (!sessionId) return null;
  let rec = state.sessionMap[sessionId];
  if (!rec) {
    rec = {
      summary: {
        session_id: sessionId,
        image_name: meta.image_name || "Image Benchmark",
        started_at_utc: meta.started_at_utc || new Date().toISOString(),
        completed_at_utc: meta.completed_at_utc || null,
        duration_s: meta.duration_s || 0,
        source: meta.source || state.sensorInfo.source,
        is_estimated:
          meta.is_estimated !== undefined ? meta.is_estimated : state.sensorInfo.is_estimated,
        sensor_detail: meta.sensor_detail || state.sensorInfo.sensor_detail,
        sample_count: meta.sample_count || 0,
        phases: meta.phases || {},
        comparison: meta.comparison || {},
      },
      samples: [],
      sampleKeys: new Set(),
      active: Boolean(meta.active),
      currentPhase: meta.currentPhase || "idle_baseline",
    };
    state.sessionMap[sessionId] = rec;
    upsertSessionSummary(rec.summary);
  } else if (meta && Object.keys(meta).length > 0) {
    rec.summary = {
      ...rec.summary,
      ...meta,
      phases: meta.phases ? { ...rec.summary.phases, ...meta.phases } : rec.summary.phases,
      comparison: meta.comparison
        ? { ...rec.summary.comparison, ...meta.comparison }
        : rec.summary.comparison,
    };
    upsertSessionSummary(rec.summary);
  }
  return rec;
}

function upsertSessionSummary(summary) {
  if (!summary || !summary.session_id) return;
  const idx = state.sessions.findIndex((s) => s.session_id === summary.session_id);
  if (idx >= 0) {
    state.sessions[idx] = { ...state.sessions[idx], ...summary };
  } else {
    state.sessions.unshift(summary);
  }
  if (state.sessions.length > 50) {
    state.sessions = state.sessions.slice(0, 50);
  }
}

function onWsSessionStart(msg) {
  const sid = msg.session_id;
  if (!sid) return;

  // Preserve all earlier sessions; create a fresh active session entry
  const rec = ensureSessionRecord(sid, {
    session_id: sid,
    image_name: msg.image_name || "Uploaded Frame",
    started_at_utc: msg.started_at_utc || new Date().toISOString(),
    source: msg.source,
    is_estimated: msg.is_estimated,
    sensor_detail: msg.sensor_detail,
    active: true,
    phases: {},
    comparison: {},
  });
  rec.samples = [];
  rec.sampleKeys = new Set();
  rec.active = true;
  rec.currentPhase = "idle_baseline";

  state.activeSessionId = sid;
  state.viewedSessionId = sid;

  if (msg.source) {
    updateSourceBadges(msg.source, msg.is_estimated, msg.sensor_detail);
  }
  updatePhaseBadges("idle_baseline");

  // Auto-expand mini live chart if user collapsed it so live benchmark is visible on upload
  if (state.miniCollapsed && state.activeView !== "view-power") {
    setMiniStripCollapsed(false);
  }

  setTxt("miniSessionName", rec.summary.image_name);
  setTxt("miniSnnAvgW", "-- W");
  setTxt("miniYoloAvgW", "-- W");
  setTxt("miniLiveWatts", "-- W");
  setTxt("liveSnnPwrVal", "-- W");
  setTxt("liveYoloPwrVal", "-- W");
  setTxt("liveSnnPeakVal", "-- W");
  setTxt("liveYoloPeakVal", "-- W");
  setTxt("liveSnnEnergyVal", "-- mJ");
  setTxt("liveYoloEnergyVal", "-- mJ");
  setTxt("livePwrRatioVal", "--×");
  setTxt("livePwrAdv", "Benchmark Running...");
  setTxt("liveEnergySavingsVal", "--% energy saved");
  setTxt("pwrRatioBadge", "Capturing Baseline...");

  renderSessionControls();
  renderViewedPowerCharts({ resetChart: true });
}

function onWsPhaseChange(msg) {
  const sid = msg.session_id || state.activeSessionId;
  const rec = ensureSessionRecord(sid);
  if (rec) {
    rec.currentPhase = msg.phase;
  }
  if (state.viewedSessionId === sid || state.activeSessionId === sid) {
    updatePhaseBadges(msg.phase);
    if (msg.phase === "snn_inference") {
      setTxt("pwrRatioBadge", "Measuring SNN Active Power...");
    } else if (msg.phase === "yolo_inference") {
      setTxt("pwrRatioBadge", "Measuring YOLOv8 Active Power...");
    } else if (msg.phase === "cooldown") {
      setTxt("pwrRatioBadge", "Inter-Model Cooldown...");
    }
  }
}

function ingestSampleIntoSession(rec, sample) {
  if (!rec || !sample || sample.watts === undefined || sample.watts === null) return false;
  const key =
    sample.seq !== undefined && sample.seq !== null
      ? `seq:${sample.seq}`
      : `t:${sample.elapsed_s}:${sample.phase}`;
  if (rec.sampleKeys.has(key)) return false;
  rec.sampleKeys.add(key);
  rec.samples.push(sample);
  rec.samples.sort((a, b) => (a.elapsed_s || 0) - (b.elapsed_s || 0));
  rec.summary.sample_count = rec.samples.length;
  return true;
}

function computePhaseLiveStats(samples, phaseName) {
  const pts = (samples || []).filter((s) => s.phase === phaseName && Number.isFinite(Number(s.watts)));
  if (pts.length === 0) return null;
  let sum = 0;
  let peak = -Infinity;
  for (const p of pts) {
    const w = Number(p.watts);
    sum += w;
    if (w > peak) peak = w;
  }
  return {
    avg: sum / pts.length,
    peak,
    latest: Number(pts[pts.length - 1].watts),
    count: pts.length,
  };
}

function onWsPowerSample(msg) {
  const sid = msg.session_id || state.activeSessionId;
  if (!sid) return;
  const rec = ensureSessionRecord(sid, {
    source: msg.source,
    is_estimated: msg.is_estimated,
    sensor_detail: msg.sensor_detail,
  });
  if (!rec) return;

  ingestSampleIntoSession(rec, msg);
  if (msg.phase) rec.currentPhase = msg.phase;
  if (msg.source) {
    updateSourceBadges(msg.source, msg.is_estimated, msg.sensor_detail);
  }

  // Update instantaneous hardware rail readouts
  const watts = Number(msg.watts);
  if (Number.isFinite(watts)) {
    setStyle("sysPowerStatus", "display", "inline");
    setTxt("globalSnnPower", `${watts.toFixed(2)} W`);
    setTxt("miniLiveWatts", `${watts.toFixed(2)} W`);
    setTxt("liveRailNowVal", `${watts.toFixed(2)} W`);
  }

  const cpuStr = msg.cpu_pct !== null && msg.cpu_pct !== undefined ? `${Number(msg.cpu_pct).toFixed(0)}%` : "--%";
  const gpuStr =
    msg.gpu_util_pct !== null && msg.gpu_util_pct !== undefined
      ? `${Number(msg.gpu_util_pct).toFixed(0)}%`
      : "N/A";
  setTxt("liveCpuGpuVal", `CPU: ${cpuStr} | GPU: ${gpuStr}`);

  // If currently viewing this session, update live SNN/YOLO metrics & ECharts incrementally
  if (state.viewedSessionId === sid) {
    updatePhaseBadges(msg.phase || rec.currentPhase);
    updateSessionReadouts(rec);
    renderViewedPowerCharts();
  }
}

function onWsSessionComplete(msg) {
  const sid = msg.session_id;
  const summary = msg.summary || {};
  const rec = ensureSessionRecord(sid, summary);
  if (rec) {
    rec.active = false;
    rec.currentPhase = "complete";
    rec.summary = { ...rec.summary, ...summary };
    upsertSessionSummary(rec.summary);
  }

  if (state.activeSessionId === sid) {
    state.activeSessionId = null;
  }

  if (state.viewedSessionId === sid && rec) {
    updatePhaseBadges("complete");
    updateSessionReadouts(rec);
    renderViewedPowerCharts();
  }

  const snnAvg = summary.phases?.snn_inference?.avg_watts;
  const yoloAvg = summary.phases?.yolo_inference?.avg_watts;
  if (Number.isFinite(snnAvg)) state.latestMeasuredSnnW = snnAvg;
  if (Number.isFinite(yoloAvg)) state.latestMeasuredYoloW = yoloAvg;
  updateBatteryAutonomy(state.batteryCapacityMah);

  renderSessionControls();
  rebuildLatencyHistoryFromSessions();
  // Sync exact persisted session store from backend
  loadSessionHistory();
}

function onWsError(msg) {
  console.error("WebSocket benchmark error:", msg);
  updatePhaseBadges("error");
  setTxt("mainChartMeta", `Benchmark Error: ${msg.message || "Unknown error"}`);
}

function formatEnergyValue(mjVal) {
  const mj = Number(mjVal);
  if (!Number.isFinite(mj)) return "-- mJ";
  if (mj >= 1000) return `${(mj / 1000).toFixed(2)} J`;
  return `${mj.toFixed(1)} mJ`;
}

function updateSessionReadouts(rec) {
  if (!rec) return;
  const sum = rec.summary || {};
  const phases = sum.phases || {};
  const cmp = sum.comparison || {};

  setTxt("miniSessionName", sum.image_name || rec.summary.session_id);
  setTxt(
    "mainChartMeta",
    `${sum.image_name || "Benchmark"} · ${sum.session_id} · ${rec.samples.length} hardware samples`
  );

  if (sum.source) {
    updateSourceBadges(sum.source, sum.is_estimated, sum.sensor_detail);
  }

  // Prefer finalized phase metrics, fall back to live running sample averages during execution
  const liveSnn = computePhaseLiveStats(rec.samples, "snn_inference");
  const liveYolo = computePhaseLiveStats(rec.samples, "yolo_inference");

  const snnAvg = phases.snn_inference?.avg_watts ?? liveSnn?.avg ?? null;
  const snnPeak = phases.snn_inference?.peak_watts ?? liveSnn?.peak ?? null;
  const snnTotalMj = phases.snn_inference?.total_energy_mj ?? null;
  const snnInfMj = phases.snn_inference?.energy_per_inference_mj ?? null;

  const yoloAvg = phases.yolo_inference?.avg_watts ?? liveYolo?.avg ?? null;
  const yoloPeak = phases.yolo_inference?.peak_watts ?? liveYolo?.peak ?? null;
  const yoloTotalMj = phases.yolo_inference?.total_energy_mj ?? null;
  const yoloInfMj = phases.yolo_inference?.energy_per_inference_mj ?? null;

  if (Number.isFinite(snnAvg)) {
    state.latestMeasuredSnnW = snnAvg;
    const snnTxt = `${snnAvg.toFixed(2)} W`;
    setTxt("miniSnnAvgW", snnTxt);
    setTxt("liveSnnPwrVal", snnTxt);
    setTxt("pwrSnn", snnTxt);
    setTxt("reconPower", snnTxt);
    setTxt("snnCardPower", snnTxt);
    setStyle("snnCardPowerPill", "display", "inline-flex");
    setTxt("cmpSnnPower", snnTxt);
    setTxt("hdrSnnPower", `${snnAvg.toFixed(2)}W`);
  } else {
    setTxt("miniSnnAvgW", "-- W");
    setTxt("liveSnnPwrVal", "-- W");
    setTxt("pwrSnn", "-- W");
  }

  if (Number.isFinite(yoloAvg)) {
    state.latestMeasuredYoloW = yoloAvg;
    const yoloTxt = `${yoloAvg.toFixed(2)} W`;
    setTxt("miniYoloAvgW", yoloTxt);
    setTxt("liveYoloPwrVal", yoloTxt);
    setTxt("pwrAnn", yoloTxt);
    setTxt("cmpYoloPower", yoloTxt);
    setTxt("hdrYoloPower", `${yoloAvg.toFixed(2)}W`);
  } else {
    setTxt("miniYoloAvgW", "-- W");
    setTxt("liveYoloPwrVal", "-- W");
    setTxt("pwrAnn", "-- W");
  }

  setTxt("liveSnnPeakVal", Number.isFinite(snnPeak) ? `${snnPeak.toFixed(2)} W` : "-- W");
  setTxt("liveYoloPeakVal", Number.isFinite(yoloPeak) ? `${yoloPeak.toFixed(2)} W` : "-- W");
  setTxt(
    "liveSnnEnergyVal",
    Number.isFinite(snnInfMj)
      ? `${formatEnergyValue(snnInfMj)}/inf`
      : Number.isFinite(snnTotalMj)
        ? formatEnergyValue(snnTotalMj)
        : "-- mJ"
  );
  setTxt(
    "liveYoloEnergyVal",
    Number.isFinite(yoloInfMj)
      ? `${formatEnergyValue(yoloInfMj)}/inf`
      : Number.isFinite(yoloTotalMj)
        ? formatEnergyValue(yoloTotalMj)
        : "-- mJ"
  );

  // Update proportional System Power Consumption bars dynamically for any platform (1W Jetson .. 45W GPU)
  const scaleMax = Math.max(5.0, (snnAvg || 0) * 1.15, (yoloAvg || 0) * 1.15);
  const snnPct = Number.isFinite(snnAvg) ? Math.max(4, Math.min(100, (snnAvg / scaleMax) * 100)) : 0;
  const annPct = Number.isFinite(yoloAvg) ? Math.max(4, Math.min(100, (yoloAvg / scaleMax) * 100)) : 0;
  setStyle("pwrSnnBar", "width", `${snnPct.toFixed(1)}%`);
  setStyle("pwrAnnBar", "width", `${annPct.toFixed(1)}%`);

  const ratio =
    cmp.power_ratio_yolo_over_snn ??
    (Number.isFinite(snnAvg) && Number.isFinite(yoloAvg) && snnAvg > 0 ? yoloAvg / snnAvg : null);
  const eSav = cmp.energy_savings_pct;
  if (Number.isFinite(ratio)) {
    setTxt("livePwrRatioVal", `${Number(ratio).toFixed(2)}×`);
    setTxt("livePwrAdv", `${Number(ratio).toFixed(2)}× Lower Draw`);
    setTxt("pwrRatioBadge", `${Number(ratio).toFixed(2)}× Power Efficiency`);
  } else {
    setTxt("livePwrRatioVal", "--×");
    setTxt("livePwrAdv", rec.active ? "Sampling..." : "Awaiting Trace");
  }

  setTxt(
    "liveEnergySavingsVal",
    Number.isFinite(eSav) ? `${Number(eSav).toFixed(1)}% energy saved` : "--% energy saved"
  );

  const lastSample = rec.samples.length > 0 ? rec.samples[rec.samples.length - 1] : null;
  if (lastSample && Number.isFinite(Number(lastSample.watts))) {
    setTxt("miniLiveWatts", `${Number(lastSample.watts).toFixed(2)} W`);
    setTxt("liveRailNowVal", `${Number(lastSample.watts).toFixed(2)} W`);
    const cpuStr =
      lastSample.cpu_pct !== null && lastSample.cpu_pct !== undefined
        ? `${Number(lastSample.cpu_pct).toFixed(0)}%`
        : "--%";
    const gpuStr =
      lastSample.gpu_util_pct !== null && lastSample.gpu_util_pct !== undefined
        ? `${Number(lastSample.gpu_util_pct).toFixed(0)}%`
        : "N/A";
    setTxt("liveCpuGpuVal", `CPU: ${cpuStr} | GPU: ${gpuStr}`);
  }
}

// ── 4. Apache ECharts Initialization, Data Mapping & Offline Fallback ────────
function initPowerCharts() {
  const echartsOk =
    typeof window.echarts === "object" && typeof window.echarts.init === "function";
  state.echartsAvailable = echartsOk;

  if (echartsOk) {
    setStyle("echartsFallbackBanner", "display", "none");
    setStyle("echartsMiniFallbackBanner", "display", "none");
    setStyle("echartsPowerMain", "display", "block");
    setStyle("echartsPowerMini", "display", "block");
    setStyle("powerTraceCanvas", "display", "none");
    setStyle("miniPowerFallbackCanvas", "display", "none");

    const mainEl = getEl("echartsPowerMain");
    const miniEl = getEl("echartsPowerMini");
    if (mainEl && !state.echartsMain) {
      state.echartsMain = window.echarts.init(mainEl, null, { renderer: "canvas" });
    }
    if (miniEl && !state.echartsMini) {
      state.echartsMini = window.echarts.init(miniEl, null, { renderer: "canvas" });
    }

    window.addEventListener("resize", () => {
      if (state.echartsMain && state.activeView === "view-power") {
        state.echartsMain.resize();
      }
      if (state.echartsMini && state.activeView !== "view-power" && !state.miniCollapsed) {
        state.echartsMini.resize();
      }
    });
  } else {
    console.warn("Apache ECharts CDN unavailable; activating HTML5 Canvas fallback.");
    setStyle("echartsFallbackBanner", "display", "block");
    setStyle("echartsMiniFallbackBanner", "display", "block");
    setStyle("echartsPowerMain", "display", "none");
    setStyle("echartsPowerMini", "display", "none");
    setStyle("powerTraceCanvas", "display", "block");
    setStyle("miniPowerFallbackCanvas", "display", "block");
  }

  renderViewedPowerCharts({ resetChart: true });
}

function buildPhaseSeriesData(samples) {
  const baselineData = [];
  const snnData = [];
  const yoloData = [];
  let snnMinT = null;
  let snnMaxT = null;
  let yoloMinT = null;
  let yoloMaxT = null;

  const sorted = [...(samples || [])].sort((a, b) => (a.elapsed_s || 0) - (b.elapsed_s || 0));
  let prevPt = null;
  let prevGroup = null;

  const toPoint = (s) => [
    Number(Number(s.elapsed_s || 0).toFixed(3)),
    Number(Number(s.watts || 0).toFixed(3)),
    s.phase || "idle_baseline",
    s.cpu_pct ?? null,
    s.gpu_util_pct ?? null,
    s.mem_pct ?? null,
    s.source || "",
    Boolean(s.is_estimated),
  ];

  const getGroup = (phase) => {
    if (phase === "snn_inference") return "snn";
    if (phase === "yolo_inference") return "yolo";
    return "base";
  };

  for (const s of sorted) {
    const pt = toPoint(s);
    const t = pt[0];
    const grp = getGroup(s.phase);

    if (grp !== prevGroup && prevPt !== null) {
      // Break previous series if returning to baseline later so distant baseline segments do not connect
      if (prevGroup === "base") {
        baselineData.push(null);
      } else if (prevGroup === "snn") {
        snnData.push(null);
      } else if (prevGroup === "yolo") {
        yoloData.push(null);
      }

      // Bridge seamlessly from the previous point into the newly entered series
      if (grp === "snn") {
        snnData.push(prevPt);
      } else if (grp === "yolo") {
        yoloData.push(prevPt);
      } else {
        baselineData.push(prevPt);
      }
    }

    if (grp === "snn") {
      snnData.push(pt);
      if (snnMinT === null || t < snnMinT) snnMinT = t;
      if (snnMaxT === null || t > snnMaxT) snnMaxT = t;
    } else if (grp === "yolo") {
      yoloData.push(pt);
      if (yoloMinT === null || t < yoloMinT) yoloMinT = t;
      if (yoloMaxT === null || t > yoloMaxT) yoloMaxT = t;
    } else {
      baselineData.push(pt);
    }

    prevPt = pt;
    prevGroup = grp;
  }

  const snnMarkAreas =
    snnMinT !== null && snnMaxT !== null && snnMaxT > snnMinT
      ? [[{ name: "SNN", xAxis: snnMinT }, { xAxis: snnMaxT }]]
      : [];
  const yoloMarkAreas =
    yoloMinT !== null && yoloMaxT !== null && yoloMaxT > yoloMinT
      ? [[{ name: "YOLOv8", xAxis: yoloMinT }, { xAxis: yoloMaxT }]]
      : [];

  return { baselineData, snnData, yoloData, snnMarkAreas, yoloMarkAreas, sorted };
}

function buildEChartsOption(session, compact) {
  const { baselineData, snnData, yoloData, snnMarkAreas, yoloMarkAreas, sorted } =
    buildPhaseSeriesData(session?.samples || []);
  const hasSamples = sorted.length > 0;

  const makeGrad = (r, g, b, topA, botA) => {
    if (window.echarts?.graphic?.LinearGradient) {
      return new window.echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: `rgba(${r}, ${g}, ${b}, ${topA})` },
        { offset: 1, color: `rgba(${r}, ${g}, ${b}, ${botA})` },
      ]);
    }
    return `rgba(${r}, ${g}, ${b}, 0.16)`;
  };

  return {
    backgroundColor: "transparent",
    animation: true,
    animationDuration: 140,
    animationDurationUpdate: 80,
    animationEasingUpdate: "linear",
    grid: compact
      ? { top: 28, right: 16, bottom: 24, left: 52 }
      : { top: 44, right: 24, bottom: 42, left: 60 },
    legend: {
      top: compact ? 2 : 8,
      right: 12,
      textStyle: { color: "#9fb0c7", fontSize: compact ? 10 : 11, fontFamily: "monospace" },
      itemWidth: 16,
      itemHeight: 8,
      data: ["Baseline / Gap", "NeuroCamo SNN", "YOLOv8-Nano"],
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "line",
        lineStyle: { color: "rgba(0, 210, 255, 0.55)", width: 1, type: "dashed" },
      },
      backgroundColor: "rgba(10, 14, 22, 0.96)",
      borderColor: "#2d384e",
      borderWidth: 1,
      textStyle: { color: "#e8ecf5", fontSize: 11, fontFamily: "monospace" },
      formatter: (params) => {
        if (!Array.isArray(params) || params.length === 0) return "";
        const valid = params.find((p) => Array.isArray(p.data) && p.data.length >= 2);
        if (!valid) return "";
        const [t, w, phase, cpu, gpu, mem, src, isEst] = valid.data;
        const pInfo = PHASE_LABELS[phase] || { text: phase || "UNKNOWN" };
        const truth = isEst ? `ESTIMATED (${src || "psutil"})` : `MEASURED (${src || "HW"})`;
        const cpuTxt = cpu !== null && cpu !== undefined ? `${Number(cpu).toFixed(1)}%` : "--";
        const gpuTxt = gpu !== null && gpu !== undefined ? `${Number(gpu).toFixed(1)}%` : "N/A";
        const memTxt = mem !== null && mem !== undefined ? `${Number(mem).toFixed(1)}%` : "--";
        return [
          `<div><strong>Time:</strong> ${Number(t).toFixed(2)}s</div>`,
          `<div><strong>Phase:</strong> ${escapeHtml(pInfo.text)}</div>`,
          `<div><strong>Power:</strong> <span style="color:#00e5a0;font-weight:700">${Number(w).toFixed(2)} W</span></div>`,
          `<div><strong>CPU / GPU:</strong> ${cpuTxt} / ${gpuTxt} (RAM ${memTxt})</div>`,
          `<div><strong>Source:</strong> ${escapeHtml(truth)}</div>`,
        ].join("");
      },
    },
    xAxis: {
      type: "value",
      name: compact ? "" : "Time (s)",
      nameLocation: "middle",
      nameGap: 25,
      nameTextStyle: { color: "#7b889e", fontSize: 11, fontFamily: "monospace" },
      min: 0,
      max: (val) =>
        val.max !== null && val.max !== undefined && isFinite(val.max)
          ? Number(Math.max(1.0, val.max + 0.15).toFixed(2))
          : 1.0,
      axisLine: { lineStyle: { color: "#2d384e" } },
      axisLabel: {
        color: "#7b889e",
        fontSize: 10,
        fontFamily: "monospace",
        formatter: (v) => `${Number(v).toFixed(1)}s`,
      },
      splitLine: { lineStyle: { color: "rgba(45, 56, 78, 0.4)", type: "dashed" } },
    },
    yAxis: {
      type: "value",
      name: compact ? "W" : "Power (W)",
      nameTextStyle: { color: "#7b889e", fontSize: 11, fontFamily: "monospace" },
      scale: true,
      min: (val) => {
        if (val.min === null || val.min === undefined || !isFinite(val.min)) return 0;
        const span = Math.max(0.25, (val.max || val.min) - val.min);
        return Math.max(0, Number((val.min - span * 0.18).toFixed(2)));
      },
      max: (val) => {
        if (val.max === null || val.max === undefined || !isFinite(val.max)) return 5;
        const span = Math.max(0.4, val.max - (val.min || 0));
        return Number((val.max + span * 0.22).toFixed(2));
      },
      axisLine: { lineStyle: { color: "#2d384e" } },
      axisLabel: {
        color: "#7b889e",
        fontSize: 10,
        fontFamily: "monospace",
        formatter: (v) => `${Number(v).toFixed(1)}W`,
      },
      splitLine: { lineStyle: { color: "rgba(45, 56, 78, 0.4)", type: "dashed" } },
    },
    graphic: hasSamples
      ? []
      : [
          {
            type: "text",
            left: "center",
            top: "middle",
            style: {
              text: "Awaiting live WebSocket power telemetry...",
              fill: "#7b889e",
              font: "12px monospace",
            },
          },
        ],
    series: [
      {
        name: "Baseline / Gap",
        type: "line",
        color: "#64748b",
        lineStyle: { width: 1.8, type: "dashed", color: "#64748b" },
        itemStyle: { color: "#64748b" },
        showSymbol: false,
        connectNulls: false,
        data: baselineData,
      },
      {
        name: "NeuroCamo SNN",
        type: "line",
        color: "#00e5a0",
        lineStyle: { width: 2.5, color: "#00e5a0" },
        itemStyle: { color: "#00e5a0" },
        symbol: "circle",
        symbolSize: compact ? 4 : 5,
        showSymbol: snnData.length <= 45,
        connectNulls: false,
        areaStyle: { color: makeGrad(0, 229, 160, 0.28, 0.02) },
        markArea: {
          silent: true,
          label: {
            show: true,
            position: "insideTop",
            color: "#00e5a0",
            fontSize: 10,
            fontFamily: "monospace",
          },
          itemStyle: {
            color: "rgba(0, 229, 160, 0.07)",
            borderColor: "rgba(0, 229, 160, 0.28)",
            borderWidth: 1,
          },
          data: snnMarkAreas,
        },
        data: snnData,
      },
      {
        name: "YOLOv8-Nano",
        type: "line",
        color: "#ff4455",
        lineStyle: { width: 2.5, color: "#ff4455" },
        itemStyle: { color: "#ff4455" },
        symbol: "circle",
        symbolSize: compact ? 4 : 5,
        showSymbol: yoloData.length <= 45,
        connectNulls: false,
        areaStyle: { color: makeGrad(255, 68, 85, 0.24, 0.02) },
        markArea: {
          silent: true,
          label: {
            show: true,
            position: "insideTop",
            color: "#ff4455",
            fontSize: 10,
            fontFamily: "monospace",
          },
          itemStyle: {
            color: "rgba(255, 68, 85, 0.07)",
            borderColor: "rgba(255, 68, 85, 0.28)",
            borderWidth: 1,
          },
          data: yoloMarkAreas,
        },
        data: yoloData,
      },
    ],
  };
}

function renderViewedPowerCharts(opts = {}) {
  const session = state.viewedSessionId ? state.sessionMap[state.viewedSessionId] : null;
  const reset = Boolean(opts.resetChart);

  if (state.echartsAvailable) {
    if (state.echartsMain && state.activeView === "view-power") {
      state.echartsMain.setOption(buildEChartsOption(session, false), reset);
    }
    if (state.echartsMini && state.activeView !== "view-power" && !state.miniCollapsed) {
      state.echartsMini.setOption(buildEChartsOption(session, true), reset);
    }
  } else {
    drawCanvasPowerTrace("powerTraceCanvas", session, false);
    drawCanvasPowerTrace("miniPowerFallbackCanvas", session, true);
  }

  // Always keep compact sparkline canvases updated with real samples (never fake waves)
  drawCanvasPowerTrace("reconPowerCanvas", session, true);
  drawCanvasPowerTrace("sysPowerCardCanvas", session, true);
}

function drawCanvasPowerTrace(canvasId, session, compact) {
  const canvas = getEl(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#06090e";
  ctx.fillRect(0, 0, w, h);

  const samples = [...(session?.samples || [])].sort(
    (a, b) => (a.elapsed_s || 0) - (b.elapsed_s || 0)
  );
  if (samples.length === 0) {
    ctx.fillStyle = "#7b889e";
    ctx.font = compact ? "10px monospace" : "12px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Awaiting live hardware telemetry...", w / 2, h / 2 + 4);
    ctx.textAlign = "left";
    return;
  }

  const padL = compact ? 6 : 50;
  const padR = compact ? 6 : 16;
  const padT = compact ? 5 : 22;
  const padB = compact ? 5 : 26;
  const drawW = Math.max(10, w - padL - padR);
  const drawH = Math.max(10, h - padT - padB);

  // Subtle grid
  ctx.strokeStyle = "rgba(0, 210, 255, 0.08)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 3; i++) {
    const gy = padT + (drawH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padL, gy);
    ctx.lineTo(padL + drawW, gy);
    ctx.stroke();
  }

  const wattsVals = samples.map((s) => Number(s.watts || 0));
  const rawMin = Math.min(...wattsVals);
  const rawMax = Math.max(...wattsVals);
  const spanW = Math.max(0.4, rawMax - rawMin);
  const minW = Math.max(0, rawMin - spanW * 0.18);
  const maxW = rawMax + spanW * 0.22;
  const maxT = Math.max(1.0, ...samples.map((s) => Number(s.elapsed_s || 0)));

  const mapX = (t) => padL + (Math.max(0, t) / maxT) * drawW;
  const mapY = (v) => padT + drawH - ((v - minW) / Math.max(0.01, maxW - minW)) * drawH;

  // Draw colored phase segments sequentially
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1];
    const b = samples[i];
    const p = b.phase || "idle_baseline";
    ctx.beginPath();
    ctx.moveTo(mapX(Number(a.elapsed_s || 0)), mapY(Number(a.watts || 0)));
    ctx.lineTo(mapX(Number(b.elapsed_s || 0)), mapY(Number(b.watts || 0)));
    if (p === "snn_inference") {
      ctx.setLineDash([]);
      ctx.strokeStyle = "#00e5a0";
      ctx.lineWidth = compact ? 2.0 : 2.5;
    } else if (p === "yolo_inference") {
      ctx.setLineDash([]);
      ctx.strokeStyle = "#ff4455";
      ctx.lineWidth = compact ? 2.0 : 2.5;
    } else {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 1.5;
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Latest sample point dot
  const last = samples[samples.length - 1];
  const lx = mapX(Number(last.elapsed_s || 0));
  const ly = mapY(Number(last.watts || 0));
  ctx.beginPath();
  ctx.arc(lx, ly, compact ? 3 : 4.5, 0, Math.PI * 2);
  ctx.fillStyle =
    last.phase === "yolo_inference"
      ? "#ff4455"
      : last.phase === "snn_inference"
        ? "#00e5a0"
        : "#94a3b8";
  ctx.fill();

  if (!compact) {
    ctx.fillStyle = "#7b889e";
    ctx.font = "10px monospace";
    ctx.fillText(`${maxW.toFixed(1)}W`, 4, padT + 10);
    ctx.fillText(`${minW.toFixed(1)}W`, 4, padT + drawH);
    ctx.fillText("0.0s", padL, h - 6);
    ctx.fillText(`${maxT.toFixed(1)}s`, padL + drawW - 34, h - 6);
  }
}

// ── 5. Multi-Session Benchmark History & Inspection ──────────────────────────
async function loadLiveSensorBadge() {
  try {
    const res = await fetch("/api/power/live");
    if (!res.ok) return;
    const data = await res.json();
    if (data.source) {
      updateSourceBadges(data.source, data.is_estimated, data.sensor_detail);
    }
    if (Number.isFinite(Number(data.watts))) {
      setStyle("sysPowerStatus", "display", "inline");
      setTxt("globalSnnPower", `${Number(data.watts).toFixed(2)} W`);
      setTxt("miniLiveWatts", `${Number(data.watts).toFixed(2)} W`);
      setTxt("liveRailNowVal", `${Number(data.watts).toFixed(2)} W`);
    }
  } catch (err) {
    console.warn("Could not read /api/power/live:", err);
  }
}

async function loadSessionHistory() {
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const data = await res.json();
    const list = Array.isArray(data.sessions) ? data.sessions : [];

    list.forEach((summary) => {
      ensureSessionRecord(summary.session_id, summary);
    });
    // Preserve backend ordering (newest first) while keeping any active in-progress session at top
    const activeSid = state.activeSessionId;
    const orderedIds = [];
    if (activeSid && state.sessionMap[activeSid]) orderedIds.push(activeSid);
    list.forEach((s) => {
      if (s.session_id && !orderedIds.includes(s.session_id)) {
        orderedIds.push(s.session_id);
      }
    });
    state.sessions = orderedIds.map((id) => state.sessionMap[id].summary);

    if (!state.viewedSessionId && state.sessions.length > 0) {
      await selectViewedSession(state.sessions[0].session_id);
    } else {
      renderSessionControls();
    }
    rebuildLatencyHistoryFromSessions();
  } catch (err) {
    console.warn("Could not fetch /api/sessions:", err);
  }
}

async function selectViewedSession(sessionId) {
  if (!sessionId) return;
  state.viewedSessionId = sessionId;
  let rec = state.sessionMap[sessionId];

  if (!rec || (!rec.active && rec.samples.length === 0)) {
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
      if (res.ok) {
        const full = await res.json();
        rec = ensureSessionRecord(sessionId, full);
        if (Array.isArray(full.samples)) {
          full.samples.forEach((s) => ingestSampleIntoSession(rec, s));
        }
      }
    } catch (err) {
      console.warn(`Failed to fetch session ${sessionId}:`, err);
    }
  }

  if (rec) {
    updatePhaseBadges(rec.active ? rec.currentPhase : "complete");
    updateSessionReadouts(rec);
  }
  renderSessionControls();
  renderViewedPowerCharts({ resetChart: true });
}

function renderSessionControls() {
  const select = getEl("sessionSelect");
  if (select) {
    select.innerHTML = "";
    if (state.sessions.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Awaiting first benchmark...";
      select.appendChild(opt);
    } else {
      state.sessions.forEach((s, idx) => {
        const opt = document.createElement("option");
        opt.value = s.session_id;
        const rec = state.sessionMap[s.session_id];
        const liveTag = rec?.active ? "[LIVE] " : `#${state.sessions.length - idx} `;
        const snnW = s.phases?.snn_inference?.avg_watts;
        const yoloW = s.phases?.yolo_inference?.avg_watts;
        const statsTag =
          Number.isFinite(snnW) && Number.isFinite(yoloW)
            ? ` (${snnW.toFixed(1)}W vs ${yoloW.toFixed(1)}W)`
            : "";
        opt.textContent = `${liveTag}${s.image_name || s.session_id}${statsTag}`;
        opt.selected = s.session_id === state.viewedSessionId;
        select.appendChild(opt);
      });
    }
  }

  setTxt(
    "sessionCountBadge",
    `${state.sessions.length} Session${state.sessions.length === 1 ? "" : "s"} Recorded`
  );
  renderSessionHistoryTable();
}

function renderSessionHistoryTable() {
  const tbody = getEl("sessionHistoryBody");
  if (!tbody) return;

  if (state.sessions.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="10" class="session-empty-cell">No benchmark sessions recorded yet. Upload an image or click a sample to start live hardware profiling.</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  state.sessions.forEach((s) => {
    const rec = state.sessionMap[s.session_id];
    const isActiveView = s.session_id === state.viewedSessionId;
    const tr = document.createElement("tr");
    if (isActiveView) tr.classList.add("active-session-row");

    const snn = s.phases?.snn_inference || {};
    const yolo = s.phases?.yolo_inference || {};
    const cmp = s.comparison || {};
    const truth = formatSourceTruth(s.source, s.is_estimated, s.sensor_detail);

    const snnAvg = Number.isFinite(snn.avg_watts) ? `${snn.avg_watts.toFixed(2)} W` : "--";
    const snnPeak = Number.isFinite(snn.peak_watts) ? `${snn.peak_watts.toFixed(2)} W` : "--";
    const snnEnergy = Number.isFinite(snn.total_energy_mj)
      ? `${formatEnergyValue(snn.total_energy_mj)} / ${formatEnergyValue(snn.energy_per_inference_mj)}`
      : "--";
    const snnLat = Number.isFinite(snn.mean_latency_ms)
      ? `${snn.mean_latency_ms.toFixed(1)} ms`
      : "--";

    const yoloAvg = Number.isFinite(yolo.avg_watts) ? `${yolo.avg_watts.toFixed(2)} W` : "--";
    const yoloPeak = Number.isFinite(yolo.peak_watts) ? `${yolo.peak_watts.toFixed(2)} W` : "--";
    const yoloEnergy = Number.isFinite(yolo.total_energy_mj)
      ? `${formatEnergyValue(yolo.total_energy_mj)} / ${formatEnergyValue(yolo.energy_per_inference_mj)}`
      : "--";
    const yoloLat = Number.isFinite(yolo.mean_latency_ms)
      ? `${yolo.mean_latency_ms.toFixed(1)} ms`
      : "--";

    const ratioTxt = Number.isFinite(cmp.power_ratio_yolo_over_snn)
      ? `${cmp.power_ratio_yolo_over_snn.toFixed(2)}× (${Number(cmp.energy_savings_pct || 0).toFixed(0)}% saved)`
      : rec?.active
        ? "LIVE..."
        : "--";

    const timeStr = s.started_at_utc
      ? new Date(s.started_at_utc).toLocaleTimeString([], { hour12: false })
      : "--";

    tr.innerHTML = `
      <td>
        <div class="session-img-name">${escapeHtml(s.image_name || "Upload")}</div>
        <div class="session-sub-meta">${escapeHtml(s.session_id)} · ${escapeHtml(timeStr)}</div>
      </td>
      <td><span class="source-truth-badge ${truth.cls}" title="${escapeHtml(truth.detail)}">${escapeHtml(truth.shortLabel)}</span></td>
      <td class="text-emerald">${snnAvg} / ${snnPeak}</td>
      <td>${snnEnergy}</td>
      <td>${snnLat}</td>
      <td class="text-crimson">${yoloAvg} / ${yoloPeak}</td>
      <td>${yoloEnergy}</td>
      <td>${yoloLat}</td>
      <td class="text-cyan">${escapeHtml(ratioTxt)}</td>
      <td><button class="btn btn-sm btn-secondary btn-inspect-session" type="button">${isActiveView ? "Viewing" : "Inspect"}</button></td>
    `;

    tr.addEventListener("click", () => {
      selectViewedSession(s.session_id);
    });
    tbody.appendChild(tr);
  });
}

function rebuildLatencyHistoryFromSessions() {
  const completed = state.sessions
    .filter(
      (s) =>
        Number.isFinite(s.phases?.snn_inference?.mean_latency_ms) &&
        Number.isFinite(s.phases?.yolo_inference?.mean_latency_ms)
    )
    .slice(0, 25)
    .reverse();

  state.latencyHistory = completed.map((s) =>
    Number(s.phases.snn_inference.mean_latency_ms.toFixed(1))
  );
  state.yoloLatencyHistory = completed.map((s) =>
    Number(s.phases.yolo_inference.mean_latency_ms.toFixed(1))
  );

  if (state.activeView === "view-power" || state.latencyHistory.length > 0) {
    drawLatencyOscilloscope(state.latencyHistory, state.yoloLatencyHistory);
  }
}

// ── 6. System Status & Curated Samples Loading ───────────────────────────────
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ready) {
      setTxt("sysStatusText", "EDGE READY");
    }
    const pm = data.power_monitor || {};
    const src = pm.source || data.power_source;
    if (src) {
      const isEst = pm.is_estimated !== undefined ? pm.is_estimated : data.power_is_estimated;
      const detail = pm.sensor_detail || data.power_sensor_detail;
      updateSourceBadges(src, isEst, detail);
    }
  } catch (err) {
    setTxt("sysStatusText", "STANDALONE");
  }
}

async function loadCuratedSamples() {
  try {
    const res = await fetch("/api/samples");
    state.samples = await res.json();

    const carousel = getEl("sampleCarousel");
    if (carousel) {
      carousel.innerHTML = "";
      state.samples.forEach((sample) => {
        const card = document.createElement("div");
        card.className = "sample-card" + (sample.id === state.currentSampleId ? " active" : "");
        card.setAttribute("data-id", sample.id);
        card.innerHTML = `
          <div class="sample-thumb-wrap">
            <img src="/api/sample_img/${sample.source}/${sample.file}" alt="${escapeHtml(sample.title)}" class="sample-thumb" loading="lazy" />
          </div>
          <div class="sample-info">
            <div class="sample-category">${escapeHtml(sample.category)}</div>
            <div class="sample-name">${escapeHtml(sample.title)}</div>
          </div>
        `;
        card.addEventListener("click", () => {
          document.querySelectorAll(".sample-card").forEach((c) => c.classList.remove("active"));
          card.classList.add("active");
          triggerSampleInference(sample.id);
        });
        carousel.appendChild(card);
      });
    }

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
    const res = await fetch(`/api/detect_sample/${sampleId}?yolo_conf=${state.yoloConf}`);
    if (!res.ok) {
      throw new Error(`Server returned status HTTP ${res.status}`);
    }
    const data = await res.json();
    if (data.error) {
      alert("Inference Error: " + data.error);
      return;
    }
    applyInferenceResult(data);
  } catch (err) {
    console.error("Sample detection failed:", err);
    setTxt("cmpSnnBadge", "OFFLINE // RETRY");
    setTxt("cmpYoloBadge", "OFFLINE // RETRY");
  } finally {
    setProcessingState(false);
  }
}

async function triggerYoloReeval(conf) {
  try {
    setTxt("cmpYoloBadge", "RE-EVALUATING YOLO...");
    setStyle("cmpYoloBadge", "color", "var(--color-amber)");
    setStyle("cmpYoloBadge", "borderColor", "var(--color-amber)");

    const res = await fetch("/api/yolo_reeval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conf: conf }),
    });
    if (!res.ok) return;
    const yolo = await res.json();
    if (state.currentResult) {
      state.currentResult.yolo_benchmark = yolo;
      updateYoloDisplay(yolo);
    }
  } catch (err) {
    console.error("YOLO re-evaluation error:", err);
  }
}

function updateYoloDisplay(yolo) {
  if (!yolo) return;
  const cmpYoloImg = getEl("cmpYoloImg");
  const cmpYoloPh = getEl("cmpYoloPlaceholder");
  if (cmpYoloImg && yolo.result_png) {
    if (cmpYoloImg.src !== yolo.result_png) cmpYoloImg.src = yolo.result_png;
    cmpYoloImg.style.display = "block";
    cmpYoloImg.classList.add("visible");
    if (cmpYoloPh) {
      cmpYoloPh.style.display = "none";
      cmpYoloPh.classList.add("hidden");
    }
  }

  const cmpYoloBadge = getEl("cmpYoloBadge");
  if (cmpYoloBadge) {
    if (!yolo.boxes_count || yolo.boxes_count === 0) {
      setTxt("cmpYoloBadge", `0 OBJECTS (CONF >= ${Math.round(state.yoloConf * 100)}%)`);
      cmpYoloBadge.classList.remove("detected");
      setStyle("cmpYoloBadge", "color", "var(--color-crimson)");
      setStyle("cmpYoloBadge", "borderColor", "rgba(255, 68, 85, 0.5)");
    } else {
      const top =
        yolo.detections && yolo.detections[0] ? yolo.detections[0] : { class: "Object", conf: 0 };
      setTxt(
        "cmpYoloBadge",
        `${yolo.boxes_count} DETECTED: ${top.class.toUpperCase()} (${top.conf}%)`
      );
      cmpYoloBadge.classList.add("detected");
      setStyle("cmpYoloBadge", "color", "var(--color-amber)");
      setStyle("cmpYoloBadge", "borderColor", "rgba(245, 163, 0, 0.5)");
    }
  }

  const yoloLat = Number.isFinite(yolo.latency_ms) ? yolo.latency_ms.toFixed(1) : "--";
  const yoloEnergy = Number.isFinite(yolo.energy_mj) ? `${yolo.energy_mj.toFixed(2)} mJ` : "-- mJ";
  setTxt("cmpYoloEnergy", yoloEnergy);
  setTxt("cmpYoloLatency", `${yoloLat} ms`);
}

// ── 7. Custom Image Upload & Event Listeners ─────────────────────────────────
function setupEventListeners() {
  const fileInput = getEl("fileInput");
  const cmpFileInput = getEl("cmpFileInput");

  const handleFileUpload = async (file) => {
    if (!file) return;

    document.querySelectorAll(".cmp-sample-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".sample-card").forEach((c) => c.classList.remove("active"));

    // Ensure WS is connected before uploading so user sees live waveform immediately
    if (!state.wsConnected) {
      await connectPowerWebSocket({ waitOpenMs: 300 });
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("yolo_conf", state.yoloConf);
    formData.append("stream_quality", state.streamQuality);

    setProcessingState(true);
    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data.error) {
        alert("Upload Error: " + data.error);
        return;
      }
      applyInferenceResult(data);
    } catch (err) {
      console.error("Detection error:", err);
      alert("Failed to run detection on uploaded image: " + err.message);
      setTxt("cmpSnnBadge", "UPLOAD FAILED");
      setTxt("cmpYoloBadge", "UPLOAD FAILED");
    } finally {
      setProcessingState(false);
      if (fileInput) fileInput.value = "";
      if (cmpFileInput) cmpFileInput.value = "";
    }
  };

  const btnUploadHeader = getEl("btnUploadHeader");
  if (btnUploadHeader && fileInput) {
    btnUploadHeader.addEventListener("click", () => fileInput.click());
  }
  if (fileInput) {
    fileInput.addEventListener("change", (e) => handleFileUpload(e.target.files[0]));
  }

  const btnUploadCompare = getEl("btnUploadCompare");
  if (btnUploadCompare && cmpFileInput) {
    btnUploadCompare.addEventListener("click", () => cmpFileInput.click());
  }
  if (cmpFileInput) {
    cmpFileInput.addEventListener("change", (e) => handleFileUpload(e.target.files[0]));
  }

  // Session History Selector Dropdown
  const sessionSelect = getEl("sessionSelect");
  if (sessionSelect) {
    sessionSelect.addEventListener("change", (e) => {
      if (e.target.value) {
        selectViewedSession(e.target.value);
      }
    });
  }

  // Comparison Sample Buttons
  document.querySelectorAll(".cmp-sample-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".cmp-sample-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const sampleId = btn.getAttribute("data-sample");
      triggerSampleInference(sampleId);
    });
  });

  // YOLO Sensitivity Slider
  const yoloSlider = getEl("yoloConfSlider");
  if (yoloSlider) {
    yoloSlider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      state.yoloConf = val / 100.0;
      setTxt("yoloConfVal", `${val}%`);
    });
    yoloSlider.addEventListener("change", async (e) => {
      const val = parseInt(e.target.value, 10);
      state.yoloConf = val / 100.0;
      await triggerYoloReeval(state.yoloConf);
    });
  }

  // Spike Band Filters
  document.querySelectorAll("#spikeBandFilters .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll("#spikeBandFilters .chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.spikeFilter = chip.getAttribute("data-band");
      if (state.currentResult?.spectrogram) {
        renderSpectrogram(state.currentResult.spectrogram);
      }
    });
  });

  // Battery Slider
  const battSlider = getEl("battSlider");
  if (battSlider) {
    battSlider.addEventListener("input", (e) => {
      const mah = parseInt(e.target.value, 10);
      state.batteryCapacityMah = mah;
      updateBatteryAutonomy(mah);
    });
  }

  // Spike Spectrogram Hover Probe
  const specCanvas = getEl("specCanvas");
  if (specCanvas) {
    specCanvas.addEventListener("mousemove", onSpecHover);
    specCanvas.addEventListener("mouseleave", () => {
      setStyle("specProbe", "display", "none");
      setTxt("specHoverReadout", "Hover over the spike matrix to inspect channel firing rates.");
    });
  }

  // Environmental Stress Testing Chips
  document.querySelectorAll(".env-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const filter = chip.getAttribute("data-filter");
      triggerEnvReeval(filter);
    });
  });

  // Stream Quality Chips
  document.querySelectorAll(".quality-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-quality");
      if (!q) return;
      state.streamQuality = q;
      document.querySelectorAll(".quality-chip").forEach((c) => {
        c.classList.toggle("active", c === chip);
      });
    });
  });

  // Temporal Scrubber & Play Controls
  const temporalSlider = getEl("temporalSlider");
  if (temporalSlider) {
    temporalSlider.addEventListener("input", (e) => {
      const step = parseInt(e.target.value, 10);
      updateTemporalDisplay(step);
    });
  }

  const btnPlayTemporal = getEl("btnPlayTemporal");
  if (btnPlayTemporal) {
    btnPlayTemporal.addEventListener("click", toggleTemporalPlay);
  }

  // Mission Dossier Export Controls
  const btnExportDossier = getEl("btnExportDossier");
  if (btnExportDossier) {
    btnExportDossier.addEventListener("click", openDossierModal);
  }

  const btnCloseDossier = getEl("btnCloseDossier");
  if (btnCloseDossier) {
    btnCloseDossier.addEventListener("click", () => {
      const modal = getEl("dossierModal");
      if (modal) modal.style.display = "none";
    });
  }

  const btnPrintDossier = getEl("btnPrintDossier");
  if (btnPrintDossier) {
    btnPrintDossier.addEventListener("click", () => window.print());
  }

  // Live Webcam & RealSense Buttons
  const btnLiveCam = getEl("btnLiveCam");
  if (btnLiveCam) {
    btnLiveCam.addEventListener("click", toggleLiveWebcam);
  }

  const btnRealSense = getEl("btnRealSense");
  if (btnRealSense) {
    btnRealSense.addEventListener("click", toggleRealSense);
  }

  // Tactical Overlay Mode Buttons
  document.querySelectorAll(".overlay-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.getAttribute("data-overlay");
      setTacticalOverlay(mode);
    });
  });

  // PTZ Slew-to-Cue Gimbal Actuator Button
  const btnSlewCue = getEl("btnSlewCue");
  if (btnSlewCue) {
    btnSlewCue.addEventListener("click", executeSlewToCue);
  }
}

function setProcessingState(busy) {
  const reconPwr = getEl("reconPower");
  const globalPwr = getEl("globalSnnPower");
  const cardPwr = getEl("snnCardPower");

  if (busy) {
    setTxt("cmpSnnBadge", "PROCESSING SNN SPIKES...");
    setStyle("cmpSnnBadge", "color", "var(--color-cyan)");
    setStyle("cmpSnnBadge", "borderColor", "var(--color-cyan)");
    setTxt("cmpYoloBadge", "PROCESSING YOLOV8...");
    setStyle("cmpYoloBadge", "color", "var(--color-amber)");
    setStyle("cmpYoloBadge", "borderColor", "var(--color-amber)");

    if (reconPwr) reconPwr.classList.add("pulsing-pwr");
    if (globalPwr) globalPwr.classList.add("pulsing-pwr");
    if (cardPwr) cardPwr.classList.add("pulsing-pwr");
  } else {
    if (reconPwr) reconPwr.classList.remove("pulsing-pwr");
    if (globalPwr) globalPwr.classList.remove("pulsing-pwr");
    if (cardPwr) cardPwr.classList.remove("pulsing-pwr");
  }
}

// ── Feature 1: Live Webcam Stream Capture & Neural Processing ────────────────
async function toggleLiveWebcam() {
  const btn = getEl("btnLiveCam");
  const btnTxt = getEl("btnLiveCamText");
  const video = getEl("webcamVideo");
  const canvas = getEl("webcamCaptureCanvas");

  if (state.isWebcamActive) {
    if (state.webcamInterval) clearInterval(state.webcamInterval);
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((track) => track.stop());
      state.webcamStream = null;
    }
    state.isWebcamActive = false;
    if (btn) btn.classList.remove("streaming");
    if (btnTxt) btnTxt.textContent = "Start Live Cam";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" },
      audio: false,
    });
    state.webcamStream = stream;
    if (video) {
      video.srcObject = stream;
      video.play();
    }
    state.isWebcamActive = true;
    if (btn) btn.classList.add("streaming");
    if (btnTxt) btnTxt.textContent = "Stop Live Cam";

    const ctx = canvas.getContext("2d");
    state.webcamInterval = setInterval(async () => {
      if (!state.isWebcamActive || state.isProcessingCamFrame) return;
      if (!video || video.readyState < 2) return;

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        async (blob) => {
          if (!blob || !state.isWebcamActive) return;
          state.isProcessingCamFrame = true;
          try {
            const formData = new FormData();
            formData.append("image", blob, "webcam_frame.jpg");
            formData.append("yolo_conf", state.yoloConf);
            formData.append("env_filter", state.envFilter);
            formData.append("is_live", "1");
            formData.append("stream_quality", state.streamQuality);

            const res = await fetch("/api/detect", { method: "POST", body: formData });
            if (res.ok) {
              const data = await res.json();
              applyInferenceResult(data);
            }
          } catch (err) {
            console.warn("Live cam frame dropped:", err);
          } finally {
            state.isProcessingCamFrame = false;
          }
        },
        "image/jpeg",
        0.65
      );
    }, 80);
  } catch (err) {
    console.warn("Browser webcam unavailable, switching to RealSense D435i...", err);
    state.isWebcamActive = true;
    if (btn) btn.classList.add("streaming");
    if (btnTxt) btnTxt.textContent = "Stop RealSense";
    const pollRS = async () => {
      if (!state.isWebcamActive || state.isProcessingCamFrame) return;
      state.isProcessingCamFrame = true;
      try {
        const res = await fetch("/api/detect_realsense", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            yolo_conf: state.yoloConf,
            env_filter: state.envFilter,
            is_live: true,
            stream_quality: state.streamQuality,
          }),
        });
        const data = await res.json();
        if (res.ok && !data.error) {
          applyInferenceResult(data);
        } else {
          throw new Error(data.error || "RealSense error");
        }
      } catch (rsErr) {
        console.error("RealSense feed error:", rsErr);
        clearInterval(state.webcamInterval);
        state.isWebcamActive = false;
        if (btn) btn.classList.remove("streaming");
        if (btnTxt) btnTxt.textContent = "Start Live Cam";
        alert("RealSense error: " + rsErr.message);
      } finally {
        state.isProcessingCamFrame = false;
      }
    };
    pollRS();
    state.webcamInterval = setInterval(pollRS, 90);
  }
}

async function toggleRealSense() {
  const btn = getEl("btnRealSense");
  const btnTxt = getEl("btnRealSenseText");

  if (state.isWebcamActive) {
    await toggleLiveWebcam();
  }

  if (state.isRealSenseActive) {
    if (state.rsTimeout) clearTimeout(state.rsTimeout);
    state.rsTimeout = null;
    state.isRealSenseActive = false;
    state.isProcessingCamFrame = false;
    if (btn) btn.classList.remove("streaming");
    if (btnTxt) btnTxt.textContent = "RealSense Live";
    return;
  }

  state.isRealSenseActive = true;
  if (btn) btn.classList.add("streaming");
  if (btnTxt) btnTxt.textContent = "Stop RealSense";

  const pollRS = async () => {
    if (!state.isRealSenseActive) return;
    if (state.isProcessingCamFrame) {
      if (state.isRealSenseActive) state.rsTimeout = setTimeout(pollRS, 25);
      return;
    }
    state.isProcessingCamFrame = true;
    try {
      const res = await fetch("/api/detect_realsense", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          yolo_conf: state.yoloConf,
          env_filter: state.envFilter,
          is_live: true,
          stream_quality: state.streamQuality,
        }),
      });
      const data = await res.json();
      if (res.ok && !data.error) {
        applyInferenceResult(data);
      } else {
        throw new Error(data.error || "RealSense stream failed");
      }
    } catch (err) {
      console.error("RealSense error:", err);
      if (state.rsTimeout) clearTimeout(state.rsTimeout);
      state.rsTimeout = null;
      state.isRealSenseActive = false;
      if (btn) btn.classList.remove("streaming");
      if (btnTxt) btnTxt.textContent = "RealSense Live";
      alert("RealSense: " + err.message);
      return;
    } finally {
      state.isProcessingCamFrame = false;
      if (state.isRealSenseActive) {
        state.rsTimeout = setTimeout(pollRS, 20);
      }
    }
  };

  pollRS();
}

// ── Feature 2: Tactical False-Color Overlay Selector ─────────────────────────
function setTacticalOverlay(mode) {
  state.currentOverlay = mode;
  document.querySelectorAll(".overlay-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-overlay") === mode);
  });

  if (!state.currentResult) return;
  updateDisplayedOverlayImages(state.currentResult);
}

function updateDisplayedOverlayImages(data) {
  let targetSrc = data.result_png;
  if (state.currentOverlay === "saliency" && data.saliency_png) {
    targetSrc = data.saliency_png;
  } else if (state.currentOverlay === "mask" && data.camo_mask_png) {
    targetSrc = data.camo_mask_png;
  }

  const detImg = getEl("detImg");
  if (detImg && targetSrc && detImg.src !== targetSrc) detImg.src = targetSrc;

  const cmpSnnImg = getEl("cmpSnnImg");
  if (cmpSnnImg && targetSrc && cmpSnnImg.src !== targetSrc) cmpSnnImg.src = targetSrc;
}

// ── Feature 3: PTZ Slew-to-Cue Gimbal Actuator ───────────────────────────────
function executeSlewToCue() {
  const btn = getEl("btnSlewCue");
  const packetStatus = getEl("gcsPacketStatus");
  if (!state.currentResult || !state.currentResult.telemetry) return;

  const tel = state.currentResult.telemetry;
  if (!tel.detected) {
    alert("PTZ Slew-to-Cue Aborted: No confirmed camouflage target locked in sector.");
    return;
  }

  if (btn) {
    btn.textContent = "SLEWING GIMBAL...";
    btn.disabled = true;
  }
  if (packetStatus) {
    packetStatus.textContent = `BUS TRANSMITTING: Slew Azimuth ${tel.bearing_deg > 0 ? "+" : ""}${tel.bearing_deg}° | Elevation ${tel.elevation_deg > 0 ? "+" : ""}${tel.elevation_deg}° | Standoff ${tel.standoff_range_m}m`;
    packetStatus.style.color = "var(--color-cyan)";
  }

  setTimeout(() => {
    if (btn) {
      btn.textContent = "TARGET INTERCEPTED & LOCKED";
      btn.classList.add("active");
    }
    if (packetStatus) {
      packetStatus.textContent = "OPTICAL GCS LOCK CONFIRMED // PTZ TRACKING ACTIVE";
      packetStatus.style.color = "var(--color-emerald)";
    }
    setTimeout(() => {
      if (btn) {
        btn.textContent = "PTZ SLEW-TO-CUE (INTERCEPT)";
        btn.disabled = false;
        btn.classList.remove("active");
      }
    }, 2800);
  }, 900);
}

// ── Feature 4: Environmental Stress Testing ──────────────────────────────────
async function triggerEnvReeval(filter) {
  state.envFilter = filter;
  document.querySelectorAll(".env-chip").forEach((c) => {
    c.classList.toggle("active", c.getAttribute("data-filter") === filter);
  });

  setProcessingState(true);
  try {
    const res = await fetch("/api/env_reeval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ env: filter, filter: filter, yolo_conf: state.yoloConf }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    applyInferenceResult(data);
  } catch (err) {
    console.error("Environment re-evaluation error:", err);
  } finally {
    setProcessingState(false);
  }
}

// ── Feature 5: Temporal Spike Evolution Scrubber ─────────────────────────────
function updateTemporalDisplay(step) {
  state.temporalStep = step;
  const slider = getEl("temporalSlider");
  if (slider && parseInt(slider.value, 10) !== step) {
    slider.value = step;
  }
  setTxt("temporalStepVal", `t = ${step} / 16`);

  if (
    !state.currentResult ||
    !state.currentResult.temporal_evolution ||
    !state.currentResult.temporal_evolution.length
  ) {
    return;
  }
  const evo = state.currentResult.temporal_evolution;
  const entry = evo[step - 1] || evo[evo.length - 1];
  if (!entry) return;

  setTxt("temporalConfVal", `${(entry.conf * 100).toFixed(1)}%`);
  setTxt("temporalSpikesVal", (entry.spikes || 0).toLocaleString());
  setTxt("temporalPeakVal", (entry.peak || 0).toFixed(3));

  let stage = "Poisson Noise Integration";
  let stageSub = "Initial rate-coded Bernoulli spike arrival";
  if (step >= 12) {
    stage = "Dense Mask Convergence";
    stageSub = "Target boundary fully stabilized";
  } else if (step >= 6) {
    stage = "Attentive Saliency Emergence";
    stageSub = "ASAB integration suppressing background texture";
  }
  setTxt("temporalStage", stage);
  setTxt("temporalStageSub", stageSub);
}

function toggleTemporalPlay() {
  const btn = getEl("btnPlayTemporal");
  if (state.isPlayingTemporal) {
    clearInterval(state.temporalTimer);
    state.isPlayingTemporal = false;
    if (btn) btn.textContent = "PLAY TIMESTEPS";
  } else {
    state.isPlayingTemporal = true;
    if (btn) btn.textContent = "PAUSE";
    let step = 1;
    updateTemporalDisplay(step);
    state.temporalTimer = setInterval(() => {
      step++;
      if (step > 16) {
        clearInterval(state.temporalTimer);
        state.isPlayingTemporal = false;
        if (btn) btn.textContent = "PLAY TIMESTEPS";
        return;
      }
      updateTemporalDisplay(step);
    }, 160);
  }
}

// ── Feature 6: Mission Dossier Export ────────────────────────────────────────
function openDossierModal() {
  const modal = getEl("dossierModal");
  if (!modal || !state.currentResult) return;

  const data = state.currentResult;
  const yolo = data.yolo_benchmark || {};
  const bsRaw = data.benchmark_session;
  const bs = bsRaw?.summary || bsRaw || state.sessionMap[state.viewedSessionId]?.summary || {};
  const confPct = ((data.confidence || 0) * 100).toFixed(1);
  const yoloConfPct =
    yolo.detections && yolo.detections.length > 0 ? yolo.detections[0].conf : "0.0";

  const snnPwr =
    bs.phases?.snn_inference?.avg_watts ??
    data.dynamic_energy?.power_snn_w ??
    data.telemetry?.power_snn_w ??
    null;
  const annPwr =
    bs.phases?.yolo_inference?.avg_watts ?? yolo.power_w ?? data.telemetry?.power_ann_w ?? null;
  const snnEnergy =
    bs.phases?.snn_inference?.energy_per_inference_mj ??
    data.dynamic_energy?.measured_snn_energy_mj ??
    data.dynamic_energy?.snn_energy_mj ??
    null;
  const yoloEnergy =
    bs.phases?.yolo_inference?.energy_per_inference_mj ??
    data.dynamic_energy?.measured_yolo_energy_mj ??
    yolo.energy_mj ??
    null;
  const snnLat = Number.isFinite(data.latency_ms) ? data.latency_ms : null;
  const yoloLat = Number.isFinite(yolo.latency_ms) ? yolo.latency_ms : null;
  const sparsityPct = Number.isFinite(data.sparsity) ? (data.sparsity * 100).toFixed(1) : "--";

  const truth = formatSourceTruth(
    bs.source || state.sensorInfo.source,
    bs.is_estimated !== undefined ? bs.is_estimated : state.sensorInfo.is_estimated,
    bs.sensor_detail || state.sensorInfo.sensor_detail
  );

  setTxt("dosTimestamp", bs.completed_at_utc || new Date().toUTCString());
  setTxt(
    "dosMissionCode",
    (bs.session_id || state.viewedSessionId || state.activeSessionId || "NC-LIVE").toUpperCase()
  );
  setTxt("dosHardwareSource", `${truth.label} (${truth.detail})`);

  const dosRaw = getEl("dosRawImg");
  if (dosRaw && data.input_png && dosRaw.src !== data.input_png) dosRaw.src = data.input_png;
  const dosSnn = getEl("dosSnnImg");
  if (dosSnn && data.result_png && dosSnn.src !== data.result_png) dosSnn.src = data.result_png;
  const dosYolo = getEl("dosYoloImg");
  if (dosYolo && yolo.result_png && dosYolo.src !== yolo.result_png) dosYolo.src = yolo.result_png;

  const detected =
    data.detected !== undefined
      ? Boolean(data.detected)
      : data.verdict === "DETECTED" || data.verdict === "POSSIBLE TARGET";
  setTxt("dosSnnVerdict", detected ? `CAMOUFLAGE LOCK (${confPct}%)` : "SECTOR CLEAR");
  setTxt(
    "dosYoloVerdict",
    yolo.boxes_count > 0 ? `${yolo.boxes_count} OBJECTS DETECTED` : "0 OBJECTS DETECTED (MISSED)"
  );

  setTxt("dosSnnConf", `${confPct}%`);
  setTxt("dosYoloConf", `${yoloConfPct}%`);
  const confDiff = (parseFloat(confPct) - parseFloat(yoloConfPct)).toFixed(1);
  setTxt("dosConfAdv", `+${confDiff}% SNN Saliency Advantage`);

  setTxt("dosSnnPower", Number.isFinite(snnPwr) ? `${snnPwr.toFixed(2)} W` : "-- W");
  setTxt("dosYoloPower", Number.isFinite(annPwr) ? `${annPwr.toFixed(2)} W` : "-- W");
  if (Number.isFinite(snnPwr) && Number.isFinite(annPwr) && snnPwr > 0) {
    setTxt("dosPowerAdv", `${(annPwr / snnPwr).toFixed(2)}× Lower Power Consumption`);
  } else {
    setTxt("dosPowerAdv", "--");
  }

  setTxt("dosSnnEnergy", Number.isFinite(snnEnergy) ? `${snnEnergy.toFixed(2)} mJ` : "-- mJ");
  setTxt("dosYoloEnergy", Number.isFinite(yoloEnergy) ? `${yoloEnergy.toFixed(2)} mJ` : "-- mJ");
  if (Number.isFinite(snnEnergy) && Number.isFinite(yoloEnergy) && snnEnergy > 0) {
    setTxt("dosEnergyAdv", `${(yoloEnergy / snnEnergy).toFixed(2)}× Energy Efficiency`);
  } else {
    setTxt("dosEnergyAdv", "--");
  }

  setTxt(
    "dosSnnLat",
    Number.isFinite(snnLat) && snnLat > 0
      ? `${snnLat.toFixed(1)} ms (~${(1000 / snnLat).toFixed(0)} FPS)`
      : "-- ms"
  );
  setTxt(
    "dosYoloLat",
    Number.isFinite(yoloLat) && yoloLat > 0
      ? `${yoloLat.toFixed(1)} ms (~${(1000 / yoloLat).toFixed(0)} FPS)`
      : "-- ms"
  );
  if (Number.isFinite(snnLat) && Number.isFinite(yoloLat) && snnLat > 0) {
    setTxt("dosLatAdv", `${(yoloLat / snnLat).toFixed(2)}× Real-time Edge Speedup`);
  } else {
    setTxt("dosLatAdv", "--");
  }

  setTxt("dosSnnSparsity", `${sparsityPct}% Silent Synapses`);
  modal.style.display = "flex";
}

// ── 8. Apply Detection Results to All Views ──────────────────────────────────
function applyInferenceResult(data, opts = {}) {
  if (!data) return;
  state.currentResult = data;

  try {
    const confPct = ((data.confidence || 0) * 100).toFixed(1);
    const detected =
      data.detected !== undefined
        ? Boolean(data.detected)
        : data.verdict === "DETECTED" || data.verdict === "POSSIBLE TARGET";
    const lat = Number.isFinite(data.latency_ms) ? data.latency_ms.toFixed(1) : "--";
    const fpsEq =
      Number.isFinite(data.latency_ms) && data.latency_ms > 0
        ? (1000.0 / data.latency_ms).toFixed(0)
        : "--";

    // A. Reconnaissance View
    const rawImg = getEl("rawImg");
    const rawPh = getEl("rawPlaceholder");
    if (rawImg && data.input_png) {
      if (rawImg.src !== data.input_png) rawImg.src = data.input_png;
      rawImg.style.display = "block";
      rawImg.classList.add("visible");
      if (rawPh) {
        rawPh.style.display = "none";
        rawPh.classList.add("hidden");
      }
    }
    const szStr = data.image_size ? `${data.image_size[0]}x${data.image_size[1]}` : "--";
    setTxt("rawMeta", `${data.filename || "Optical Frame"} [${szStr}]`);

    const detImg = getEl("detImg");
    const detPh = getEl("detPlaceholder");
    if (detImg && data.result_png) {
      updateDisplayedOverlayImages(data);
      detImg.style.display = "block";
      detImg.classList.add("visible");
      if (detPh) {
        detPh.style.display = "none";
        detPh.classList.add("hidden");
      }
    }
    setTxt("detMeta", "best_model_v4.pth (128x128 LIF Dense Mask)");

    const reconVerdict = getEl("reconVerdict");
    if (reconVerdict) {
      if (detected) {
        reconVerdict.textContent = "TARGET LOCKED // CAMOUFLAGE COMPROMISED";
        reconVerdict.className = "box-value highlight text-emerald";
        setTxt("reconVerdictSub", `Dense spatial mask localized target with ${confPct}% confidence`);
      } else {
        reconVerdict.textContent = "SECTOR CLEAR // NO CAMOUFLAGE";
        reconVerdict.className = "box-value text-muted";
        setTxt(
          "reconVerdictSub",
          data.gate_reason || "Zero significant camouflage activations in sector"
        );
      }
    }
    setTxt("reconConf", `${confPct}%`);
    setStyle("reconConfBar", "width", `${confPct}%`);
    setTxt("reconTex", (data.texture_score || 0).toFixed(1));
    setTxt(
      "reconGateReason",
      data.plain_background ? "Low Complexity: Flat Background" : "High Complexity: Passed Scene Gate"
    );
    setTxt("reconLatency", `${lat} ms`);
    setTxt("reconFps", `Edge Throughput: ~${fpsEq} FPS`);

    // B. Merge Benchmark Session if present (preserving any samples already streamed via WS)
    const bsRaw = data.benchmark_session;
    const bsSummary = bsRaw?.summary || bsRaw || {};
    const bsSamples = Array.isArray(bsRaw?.samples)
      ? bsRaw.samples
      : Array.isArray(bsSummary?.samples)
        ? bsSummary.samples
        : [];
    const anotherSessionLive = Boolean(
      state.activeSessionId &&
        (!bsSummary?.session_id || state.activeSessionId !== bsSummary.session_id)
    );
    if (!opts.skipSessionMerge && bsSummary && bsSummary.session_id) {
      const rec = ensureSessionRecord(bsSummary.session_id, bsSummary);
      if (rec) {
        rec.active = false;
        rec.currentPhase = "complete";
        bsSamples.forEach((s) => ingestSampleIntoSession(rec, s));
        if (state.activeSessionId === bsSummary.session_id) {
          state.activeSessionId = null;
        }
        if (!anotherSessionLive) {
          state.viewedSessionId = bsSummary.session_id;
          if (bsSummary.source) {
            updateSourceBadges(bsSummary.source, bsSummary.is_estimated, bsSummary.sensor_detail);
          }
          updatePhaseBadges("complete");
          updateSessionReadouts(rec);
          renderViewedPowerCharts();
        }
        renderSessionControls();
        rebuildLatencyHistoryFromSessions();
      }
    } else if (
      !anotherSessionLive &&
      (data.power_live?.source || data.telemetry?.source || data.dynamic_energy?.source)
    ) {
      const srcObj = data.power_live?.source
        ? data.power_live
        : data.telemetry?.source
          ? data.telemetry
          : data.dynamic_energy;
      updateSourceBadges(srcObj.source, srcObj.is_estimated, srcObj.sensor_detail);
    }

    // Real SNN & YOLO metrics
    const yolo = data.yolo_benchmark || {};
    const bs = bsSummary;
    const snnPwr =
      bs.phases?.snn_inference?.avg_watts ??
      data.dynamic_energy?.power_snn_w ??
      data.telemetry?.power_snn_w ??
      null;
    const annPwr =
      bs.phases?.yolo_inference?.avg_watts ??
      yolo.power_w ??
      data.telemetry?.power_ann_w ??
      null;

    const snnCmosEnergy = data.dynamic_energy?.snn_energy_mj ?? null;
    const snnMeasEnergy =
      bs.phases?.snn_inference?.energy_per_inference_mj ??
      data.dynamic_energy?.measured_snn_energy_mj ??
      null;
    const yoloMeasEnergy =
      bs.phases?.yolo_inference?.energy_per_inference_mj ??
      data.dynamic_energy?.measured_yolo_energy_mj ??
      yolo.energy_mj ??
      null;

    const sparsityPct = Number.isFinite(data.sparsity) ? (data.sparsity * 100).toFixed(1) : "--";
    const yoloOps = Number.isFinite(yolo.gflops)
      ? `${yolo.gflops.toFixed(2)} GFLOPs (Dense)`
      : "8.70 GFLOPs (Dense)";

    if (!anotherSessionLive) {
      if (Number.isFinite(snnPwr)) {
        state.latestMeasuredSnnW = snnPwr;
        setTxt("reconPower", `${snnPwr.toFixed(2)} W`);
        setTxt("snnCardPower", `${snnPwr.toFixed(2)} W`);
        setStyle("snnCardPowerPill", "display", "inline-flex");
        setTxt("globalSnnPower", `${snnPwr.toFixed(2)} W`);
        setStyle("sysPowerStatus", "display", "inline");
        setTxt("cmpSnnPower", `${snnPwr.toFixed(2)} W`);
        setTxt("hdrSnnPower", `${snnPwr.toFixed(2)}W`);
        setTxt("pwrSnn", `${snnPwr.toFixed(2)} W`);
      }

      if (Number.isFinite(annPwr)) {
        state.latestMeasuredYoloW = annPwr;
        setTxt("cmpYoloPower", `${annPwr.toFixed(2)} W`);
        setTxt("hdrYoloPower", `${annPwr.toFixed(2)}W`);
        setTxt("pwrAnn", `${annPwr.toFixed(2)} W`);
      }

      updateBatteryAutonomy(state.batteryCapacityMah);
    }

    if (Number.isFinite(snnMeasEnergy)) {
      setTxt("reconEnergy", `${formatEnergyValue(snnMeasEnergy)} / inference (measured)`);
      setTxt(
        "cmpSnnEnergy",
        Number.isFinite(snnCmosEnergy)
          ? `${formatEnergyValue(snnMeasEnergy)} hw (${snnCmosEnergy.toFixed(2)} mJ CMOS)`
          : `${formatEnergyValue(snnMeasEnergy)} hw`
      );
    } else if (Number.isFinite(snnCmosEnergy)) {
      setTxt("reconEnergy", `${snnCmosEnergy.toFixed(2)} mJ (CMOS model)`);
      setTxt("cmpSnnEnergy", `${snnCmosEnergy.toFixed(2)} mJ (CMOS)`);
    }

    // C. Head-to-Head Compare View
    const cmpSnnImg = getEl("cmpSnnImg");
    const cmpSnnPh = getEl("cmpSnnPlaceholder");
    if (cmpSnnImg && data.result_png) {
      updateDisplayedOverlayImages(data);
      cmpSnnImg.style.display = "block";
      cmpSnnImg.classList.add("visible");
      if (cmpSnnPh) {
        cmpSnnPh.style.display = "none";
        cmpSnnPh.classList.add("hidden");
      }
    }

    if (detected) {
      setTxt("cmpSnnBadge", `CAMOUFLAGE LOCK (${confPct}%)`);
      setStyle("cmpSnnBadge", "color", "var(--color-emerald)");
      setStyle("cmpSnnBadge", "borderColor", "rgba(0, 229, 160, 0.5)");
    } else {
      setTxt("cmpSnnBadge", "SECTOR CLEAR (NO CAMOUFLAGE)");
      setStyle("cmpSnnBadge", "color", "var(--text-muted)");
      setStyle("cmpSnnBadge", "borderColor", "var(--border-dim)");
    }

    updateYoloDisplay(yolo);
    if (Number.isFinite(yoloMeasEnergy)) {
      setTxt("cmpYoloEnergy", `${formatEnergyValue(yoloMeasEnergy)} hw`);
    }

    setTxt("cmpSnnSparsity", `${sparsityPct}% Silent`);
    setTxt("cmpSnnLatency", `${lat} ms`);
    setTxt("cmpYoloOps", yoloOps);

    // Sync active environmental chip
    const activeEnv = data.env_filter || state.envFilter || "normal";
    document.querySelectorAll(".env-chip").forEach((c) => {
      c.classList.toggle("active", c.getAttribute("data-filter") === activeEnv);
    });

    updateTemporalDisplay(16);

    if (data.sample_info?.id || state.currentSampleId) {
      const activeId = data.sample_info?.id || state.currentSampleId;
      document.querySelectorAll(".cmp-sample-btn").forEach((b) => {
        b.classList.toggle("active", b.getAttribute("data-sample") === activeId);
      });
    }

    const isLiveActive = state.isWebcamActive || state.isRealSenseActive;

    // STANAG 4586 / MAVLink Tactical GCS Console
    const tel = data.telemetry || {};
    const azSign = (tel.bearing_deg || 0) >= 0 ? "+" : "";
    const elSign = (tel.elevation_deg || 0) >= 0 ? "+" : "";
    setTxt("gcsAzimuth", `${azSign}${(tel.bearing_deg || 0).toFixed(1)}°`);
    setTxt("gcsElevation", `${elSign}${(tel.elevation_deg || 0).toFixed(1)}°`);
    setTxt("gcsRange", tel.detected ? `${tel.standoff_range_m} m` : "-- m");
    setTxt("gcsFootprint", tel.detected ? `${tel.target_footprint_m2} m²` : "-- m²");
    if (tel.stanag_packet) {
      setTxt("gcsPacket", tel.stanag_packet);
    }
    const packetStatus = getEl("gcsPacketStatus");
    if (packetStatus) {
      packetStatus.textContent = tel.detected
        ? "RS-422 Serial Link: TARGET LOCK // PTZ READY"
        : "RS-422 Serial Link: Standby / Sector Clear";
      packetStatus.style.color = tel.detected ? "var(--color-emerald)" : "var(--text-muted)";
    }

    // Neuromorphic Core tab
    if (data.spectrogram) {
      if (!isLiveActive || state.activeView === "view-spikes") {
        renderSpectrogram(data.spectrogram);
        if (data.spectrogram.per_step_rate) {
          drawRateCurve(data.spectrogram.per_step_rate);
        }
      }
      setTxt("activeChannelsVal", `${data.spectrogram.active_channels || 0} / 128`);
    }
    setTxt("sparsityVal", `${sparsityPct}%`);
    setTxt("zeroMacVal", `${sparsityPct}%`);

    // Theoretical CMOS energy card
    if (data.dynamic_energy) {
      state.currentEnergy = data.dynamic_energy;
      if (!isLiveActive || state.activeView === "view-power") {
        drawDynamicEnergyChart(state.currentEnergy);
      }
    }
  } catch (err) {
    console.error("Non-fatal error updating UI views:", err);
  }
}

// ── 9. Canvas: Comparative Latency & Throughput Trace (ms) ───────────────────
function drawLatencyOscilloscope(snnHist, yoloHist) {
  const canvas = getEl("latencyCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const padX = 48;
  const padY = 24;
  const drawW = w - padX * 2;
  const drawH = h - padY * 2;

  ctx.strokeStyle = "rgba(0, 210, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, drawW, drawH);

  for (let y = padY + drawH / 4; y < padY + drawH; y += drawH / 4) {
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(padX + drawW, y);
    ctx.stroke();
  }

  const count = Math.max(snnHist?.length || 0, yoloHist?.length || 0);
  if (count === 0) {
    ctx.fillStyle = "#7b889e";
    ctx.font = "12px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Awaiting completed benchmark sessions...", w / 2, h / 2);
    ctx.textAlign = "left";
    setTxt("liveLatencyVal", "-- ms");
    setTxt("liveYoloLatencyVal", "-- ms");
    setTxt("liveLatRatioVal", "--×");
    setTxt("liveLatAdv", "--× Speedup");
    setTxt("liveThroughputAdvVal", "-- FPS");
    setTxt("liveThroughputDeltaSub", "Awaiting benchmark");
    return;
  }

  const allVals = [...(snnHist || []), ...(yoloHist || [])].filter((v) => Number.isFinite(v));
  const maxVal = Math.max(10.0, ...allVals) * 1.18;
  const stepW = count > 1 ? drawW / (count - 1) : 0;

  if (yoloHist && yoloHist.length > 0) {
    ctx.beginPath();
    yoloHist.forEach((val, i) => {
      const x = count > 1 ? padX + i * stepW : padX + drawW / 2;
      const y = padY + drawH - (Math.min(maxVal, val) / maxVal) * drawH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#ff4455";
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  if (snnHist && snnHist.length > 0) {
    ctx.beginPath();
    snnHist.forEach((val, i) => {
      const x = count > 1 ? padX + i * stepW : padX + drawW / 2;
      const y = padY + drawH - (Math.min(maxVal, val) / maxVal) * drawH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#00e5a0";
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  const lastSnn = snnHist[snnHist.length - 1];
  const lastYolo = yoloHist[yoloHist.length - 1];
  const lastX = count > 1 ? padX + (count - 1) * stepW : padX + drawW / 2;

  if (Number.isFinite(lastSnn)) {
    ctx.fillStyle = "#00e5a0";
    ctx.beginPath();
    ctx.arc(lastX, padY + drawH - (lastSnn / maxVal) * drawH, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  if (Number.isFinite(lastYolo)) {
    ctx.fillStyle = "#ff4455";
    ctx.beginPath();
    ctx.arc(lastX, padY + drawH - (lastYolo / maxVal) * drawH, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#7b889e";
  ctx.font = "10px monospace";
  ctx.fillText(`${maxVal.toFixed(0)} ms`, 4, padY + 10);
  ctx.fillText(`${(maxVal / 2).toFixed(0)} ms`, 4, padY + drawH / 2);
  ctx.fillText("0 ms", 14, padY + drawH);
  ctx.fillText(`Sessions (${count})`, padX, h - 6);
  ctx.fillText("LATEST", padX + drawW - 42, h - 6);

  if (Number.isFinite(lastSnn) && Number.isFinite(lastYolo)) {
    const speedup = (lastYolo / Math.max(0.1, lastSnn)).toFixed(2);
    const snnFps = (1000.0 / Math.max(0.1, lastSnn)).toFixed(0);
    const yoloFps = (1000.0 / Math.max(0.1, lastYolo)).toFixed(0);
    setTxt("liveLatencyVal", `${lastSnn.toFixed(1)} ms`);
    setTxt("liveSnnFpsSub", `Throughput: ~${snnFps} FPS`);
    setTxt("liveYoloLatencyVal", `${lastYolo.toFixed(1)} ms`);
    setTxt("liveYoloFpsSub", `Throughput: ~${yoloFps} FPS (dense)`);
    setTxt("liveLatRatioVal", `${speedup}×`);
    setTxt("liveLatAdv", `${speedup}× Speedup`);
    const fpsDelta = Math.abs(parseInt(snnFps, 10) - parseInt(yoloFps, 10));
    setTxt("liveThroughputAdvVal", `${snnFps} FPS`);
    setTxt("liveThroughputDeltaSub", `Δ ${fpsDelta} FPS vs Dense YOLO`);
  }
}

// ── 10. Canvas: Dynamic Synaptic Energy per Frame ────────────────────────────
function drawDynamicEnergyChart(energyData) {
  const canvas = getEl("energyCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  if (
    !energyData ||
    !Number.isFinite(energyData.snn_energy_mj) ||
    !Number.isFinite(energyData.ann_energy_mj)
  ) {
    ctx.fillStyle = "#7b889e";
    ctx.font = "12px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Awaiting inference synaptic energy calculation...", w / 2, h / 2);
    ctx.textAlign = "left";
    setTxt("frameEnergyVal", "-- mJ");
    setTxt("annEnergyVal", "-- mJ");
    setTxt("savingsFactorVal", "--×");
    return;
  }

  const padX = 40;
  const padY = 25;
  const drawW = w - padX * 2;

  const snnVal = energyData.snn_energy_mj;
  const annVal = energyData.ann_energy_mj;
  const maxVal = Math.max(annVal * 1.15, 1.0);

  const barH = 34;
  const y1 = padY + 30;
  const y2 = padY + 95;

  const wSnn = (snnVal / maxVal) * drawW;
  const wAnn = (annVal / maxVal) * drawW;

  ctx.fillStyle = "#7b889e";
  ctx.font = "11px sans-serif";
  ctx.fillText("Proposed SNN (Event-Driven AC Spikes):", padX, y1 - 8);

  ctx.fillStyle = "rgba(0, 229, 160, 0.12)";
  ctx.fillRect(padX, y1, drawW, barH);
  const gradSnn = ctx.createLinearGradient(padX, 0, padX + Math.max(4, wSnn), 0);
  gradSnn.addColorStop(0, "#00b078");
  gradSnn.addColorStop(1, "#00e5a0");
  ctx.fillStyle = gradSnn;
  ctx.fillRect(padX, y1, wSnn, barH);
  ctx.strokeStyle = "#00e5a0";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(padX, y1, wSnn, barH);

  ctx.fillStyle = "#fff";
  ctx.font = "bold 12px monospace";
  ctx.fillText(`${snnVal.toFixed(2)} mJ`, Math.min(padX + drawW - 70, padX + wSnn + 10), y1 + 22);

  ctx.fillStyle = "#7b889e";
  ctx.font = "11px sans-serif";
  ctx.fillText("Conventional Dense ANN Baseline (Dense MACs):", padX, y2 - 8);

  ctx.fillStyle = "rgba(255, 68, 85, 0.12)";
  ctx.fillRect(padX, y2, drawW, barH);
  const gradAnn = ctx.createLinearGradient(padX, 0, padX + Math.max(4, wAnn), 0);
  gradAnn.addColorStop(0, "#b02030");
  gradAnn.addColorStop(1, "#ff4455");
  ctx.fillStyle = gradAnn;
  ctx.fillRect(padX, y2, wAnn, barH);
  ctx.strokeStyle = "#ff4455";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(padX, y2, wAnn, barH);

  ctx.fillStyle = "#fff";
  ctx.font = "bold 12px monospace";
  ctx.fillText(`${annVal.toFixed(2)} mJ`, padX + drawW - 72, y2 + 22);

  setTxt("frameEnergyVal", `${snnVal.toFixed(2)} mJ`);
  setTxt("annEnergyVal", `${annVal.toFixed(2)} mJ`);
  const factor = (annVal / Math.max(0.01, snnVal)).toFixed(1);
  setTxt("savingsFactorVal", `${factor}×`);
}

// ── 11. Canvas: Layer-3 Spike Spectrogram ────────────────────────────────────
function renderSpectrogram(spec) {
  const canvas = getEl("specCanvas");
  if (!canvas || !spec) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const nChannels = spec.n_channels || 128;
  const nSteps = spec.n_steps || 16;
  const data = spec.data;
  if (!data) return;

  const cellW = w / nSteps;
  const cellH = h / nChannels;
  const filter = state.spikeFilter;

  for (let c = 0; c < nChannels; c++) {
    for (let t = 0; t < nSteps; t++) {
      const idx = c * nSteps + t;
      const val = data[idx] / 255.0;

      if (val < 0.02) continue;

      let band = "sparse";
      if (val > 0.66) band = "saturated";
      else if (val > 0.33) band = "active";

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
  const canvas = getEl("specCanvas");
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  const mouseX = (e.clientX - rect.left) * scaleX;
  const mouseY = (e.clientY - rect.top) * scaleY;

  const nChannels = spec.n_channels || 128;
  const nSteps = spec.n_steps || 16;

  const stepIdx = Math.floor(mouseX / (canvas.width / nSteps));
  const chanIdx = Math.floor(mouseY / (canvas.height / nChannels));

  if (stepIdx >= 0 && stepIdx < nSteps && chanIdx >= 0 && chanIdx < nChannels) {
    const rawVal = spec.data[chanIdx * nSteps + stepIdx];
    const rateNorm = (rawVal / 255.0) * (spec.peak_rate || 1.0);

    const txt = `CH ${chanIdx} | t=${stepIdx} | Rate: ${rateNorm.toFixed(3)} (Peak: ${spec.peak_rate.toFixed(3)})`;
    setTxt("specHoverReadout", txt);

    setStyle("specProbe", "display", "block");
    setStyle("specProbe", "left", `${e.clientX - rect.left + 15}px`);
    setStyle("specProbe", "top", `${e.clientY - rect.top + 10}px`);
    setTxt("specProbe", `Ch ${chanIdx} [t${stepIdx}]: ${(rateNorm * 100).toFixed(1)}%`);
  }
}

// ── 12. Canvas: Population Rate Curve (t0 -> t15) ────────────────────────────
function drawRateCurve(rates) {
  const canvas = getEl("rateCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const padX = 35;
  const padY = 25;
  const drawW = w - padX * 2;
  const drawH = h - padY * 2;

  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, drawW, drawH);

  if (!rates || rates.length === 0) return;
  const maxVal = Math.max(0.1, ...rates);

  ctx.beginPath();
  const stepW = drawW / Math.max(1, rates.length - 1);

  rates.forEach((val, i) => {
    const x = padX + i * stepW;
    const y = padY + drawH - (val / maxVal) * drawH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = "#00d2ff";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  ctx.lineTo(padX + drawW, padY + drawH);
  ctx.lineTo(padX, padY + drawH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padY, 0, padY + drawH);
  grad.addColorStop(0, "rgba(0, 210, 255, 0.25)");
  grad.addColorStop(1, "rgba(0, 210, 255, 0.0)");
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.fillStyle = "#7b889e";
  ctx.font = "10px monospace";
  ctx.fillText("t0", padX, h - 8);
  ctx.fillText("t15", padX + drawW - 15, h - 8);
  ctx.fillText(`${maxVal.toFixed(2)}`, 6, padY + 10);
}

// ── 13. Battery Autonomy Calculator (Measured Rail Power) ────────────────────
function updateBatteryAutonomy(mah) {
  const wh = (mah * 12.0) / 1000.0;
  setTxt("battCapText", `${mah} mAh (${wh.toFixed(1)} Wh)`);

  const snnW = state.latestMeasuredSnnW;
  const annW = state.latestMeasuredYoloW;
  if (!Number.isFinite(snnW) || snnW <= 0 || !Number.isFinite(annW) || annW <= 0) {
    setTxt("autoSnn", "-- hrs");
    setTxt("autoAnn", "-- hrs");
    setTxt("autoExtraHours", "Awaiting measured power");
    return;
  }

  const usableWh = wh * 0.85;
  const snnHrs = usableWh / snnW;
  const annHrs = usableWh / annW;
  const delta = snnHrs - annHrs;

  setTxt("autoSnn", `${snnHrs.toFixed(1)} hrs`);
  setTxt("autoAnn", `${annHrs.toFixed(1)} hrs`);
  setTxt(
    "autoExtraHours",
    delta >= 0
      ? `+${delta.toFixed(1)} Extra Patrol Hours`
      : `${delta.toFixed(1)} hrs Delta`
  );
}

// ── Layout: keep canvas bitmaps / ECharts in sync with their CSS boxes ───────
// Canvas bitmaps are resized to the rendered box (no stretched text) and the
// matching chart is redrawn from real data; ECharts instances are resized
// whenever their container changes size (tab switch, window resize, reflow).
(function setupLayoutResizeObservers() {
  if (typeof ResizeObserver === "undefined") return;

  const redrawers = {
    latencyCanvas: () => drawLatencyOscilloscope(state.latencyHistory, state.yoloLatencyHistory),
    energyCanvas: () => drawDynamicEnergyChart(state.currentEnergy),
    sysPowerCardCanvas: () => renderViewedPowerCharts(),
    powerTraceCanvas: () => renderViewedPowerCharts(),
    reconPowerCanvas: () => renderViewedPowerCharts(),
    miniPowerFallbackCanvas: () => renderViewedPowerCharts(),
  };

  function observe() {
    Object.keys(redrawers).forEach((id) => {
      const canvas = document.getElementById(id);
      if (!canvas || canvas.dataset.roBound) return;
      canvas.dataset.roBound = "1";
      let raf = 0;
      new ResizeObserver(() => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const w = Math.round(canvas.clientWidth);
          const h = Math.round(canvas.clientHeight);
          if (w < 8 || h < 8) return;
          if (canvas.width === w && canvas.height === h) return;
          canvas.width = w;
          canvas.height = h;
          try {
            redrawers[id]();
          } catch (err) {
            console.warn("Canvas redraw failed:", id, err);
          }
        });
      }).observe(canvas);
    });

    [["echartsPowerMain", "echartsMain"], ["echartsPowerMini", "echartsMini"]].forEach(([elId, key]) => {
      const el = document.getElementById(elId);
      if (!el || el.dataset.roBound) return;
      el.dataset.roBound = "1";
      new ResizeObserver(() => {
        const inst = state[key];
        if (inst && el.clientWidth > 8 && el.clientHeight > 8) inst.resize();
      }).observe(el);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", observe);
  } else {
    observe();
  }
})();
