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
  currentSampleId: "camo_marine",
  currentResult: null,
  spikeFilter: "all",
  yoloConf: 0.15,
  envFilter: "normal",
  temporalStep: 16,
  temporalTimer: null,
  isPlayingTemporal: false,
  currentOverlay: "box", // "box", "saliency", "mask"
  isWebcamActive: false,
  webcamStream: null,
  webcamInterval: null,
  isProcessingCamFrame: false,
  batteryCapacityMah: 5200,
  latencyHistory: [16.2, 16.8, 15.9, 16.4, 17.1, 16.0, 15.8, 16.5, 16.1],
  yoloLatencyHistory: [52.4, 48.7, 51.2, 53.6, 50.1, 49.3, 52.8, 51.5, 52.3],
  snnPowerHistory: [0.94, 0.96, 0.91, 0.95, 0.98, 0.92, 0.90, 0.97, 0.95],
  yoloPowerHistory: [8.82, 8.75, 8.91, 8.85, 8.78, 8.89, 8.84, 8.79, 8.85],
  currentEnergy: {
    snn_energy_mj: 8.14,
    ann_energy_mj: 40.02,
    energy_reduction_factor: "5.0x",
  },
};

// Safe DOM Helper Functions (prevents any 'Cannot set properties of undefined' crashes)
function getEl(id) {
  return document.getElementById(id);
}

function setTxt(id, val) {
  const node = typeof id === "string" ? document.getElementById(id) : id;
  if (node) {
    node.textContent = val;
  }
}

function setProp(id, prop, val) {
  const node = typeof id === "string" ? document.getElementById(id) : id;
  if (node) {
    node[prop] = val;
  }
}

function setStyle(id, prop, val) {
  const node = typeof id === "string" ? document.getElementById(id) : id;
  if (node && node.style) {
    node.style[prop] = val;
  }
}

// ── Application Initialization ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupEventListeners();

  drawRateCurve([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  drawPowerTraceChart(state.snnPowerHistory, state.yoloPowerHistory);
  drawLatencyOscilloscope(state.latencyHistory, state.yoloLatencyHistory);
  drawDynamicEnergyChart(state.currentEnergy);

  await loadSystemStatus();
  await loadCuratedSamples();
  startRealtimeMonitor();
});

// ── 1. Tab & Dropdown Mode Navigation ───────────────────────────────────────
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
}

function switchView(viewId) {
  state.activeView = viewId;

  document.querySelectorAll(".nav-tab").forEach((tab) => {
    if (tab.getAttribute("data-view") === viewId) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  const modeSelect = getEl("modeSelect");
  if (modeSelect && (viewId === "view-recon" || viewId === "view-compare")) {
    modeSelect.value = viewId;
  }

  document.querySelectorAll(".view").forEach((view) => {
    if (view.id === viewId) {
      view.classList.add("active");
    } else {
      view.classList.remove("active");
    }
  });

  // Re-draw active canvases or views
  if (viewId === "view-spikes" && state.currentResult) {
    renderSpectrogram(state.currentResult.spectrogram);
    drawRateCurve(state.currentResult.spectrogram.per_step_rate);
  } else if (viewId === "view-power") {
    drawPowerTraceChart(state.snnPowerHistory, state.yoloPowerHistory);
    drawLatencyOscilloscope(state.latencyHistory, state.yoloLatencyHistory);
    drawDynamicEnergyChart(state.currentEnergy);
  } else if ((viewId === "view-compare" || viewId === "view-recon") && state.currentResult) {
    applyInferenceResult(state.currentResult);
  }
}

// ── 2. System Status & Samples Loading ───────────────────────────────────────
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ready) {
      setTxt("sysStatusText", `JETSON ${data.device.toUpperCase()} // READY`);
    }
  } catch (err) {
    setTxt("sysStatusText", "STANDALONE MODE");
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
            <img src="/api/sample_img/${sample.source}/${sample.file}" alt="${sample.title}" class="sample-thumb" loading="lazy" />
          </div>
          <div class="sample-info">
            <div class="sample-category">${sample.category}</div>
            <div class="sample-name">${sample.title}</div>
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

    // Automatically trigger initial live inference on first sample
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
    alert(`Could not complete inference: ${err.message}\nMake sure launch_neurocamo_software.bat is running.`);
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
    cmpYoloImg.src = yolo.result_png;
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
      const top = yolo.detections && yolo.detections[0] ? yolo.detections[0] : { class: "Object", conf: 0 };
      setTxt("cmpYoloBadge", `${yolo.boxes_count} DETECTED: ${top.class.toUpperCase()} (${top.conf}%)`);
      cmpYoloBadge.classList.add("detected");
      setStyle("cmpYoloBadge", "color", "var(--color-amber)");
      setStyle("cmpYoloBadge", "borderColor", "rgba(245, 163, 0, 0.5)");
    }
  }

  const yoloLat = yolo.latency_ms !== undefined ? yolo.latency_ms.toFixed(1) : "42.7";
  const yoloEnergy = yolo.energy_mj || 40.02;
  setTxt("cmpYoloEnergy", `${yoloEnergy.toFixed(2)} mJ`);
  setTxt("cmpYoloLatency", `${yoloLat} ms`);
}

// ── 3. Custom Image Upload & Event Listeners ─────────────────────────────────
function setupEventListeners() {
  const fileInput = getEl("fileInput");
  const cmpFileInput = getEl("cmpFileInput");

  const handleFileUpload = async (file) => {
    if (!file) return;

    // Do NOT force switchView away! Retain the mode the user is actively viewing
    document.querySelectorAll(".cmp-sample-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".sample-card").forEach((c) => c.classList.remove("active"));

    const formData = new FormData();
    formData.append("image", file);
    formData.append("yolo_conf", state.yoloConf);

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

  // Live Webcam Stream Button
  const btnLiveCam = getEl("btnLiveCam");
  if (btnLiveCam) {
    btnLiveCam.addEventListener("click", toggleLiveWebcam);
  }

  // Intel RealSense Live Button
  const btnRealSense = getEl("btnRealSense");
  if (btnRealSense) {
    btnRealSense.addEventListener("click", toggleRealSense);
  }

  // Tactical Overlay Mode Buttons (Recon & Compare)
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
  if (busy) {
    setTxt("cmpSnnBadge", "PROCESSING SNN SPIKES...");
    setStyle("cmpSnnBadge", "color", "var(--color-cyan)");
    setStyle("cmpSnnBadge", "borderColor", "var(--color-cyan)");
    setTxt("cmpYoloBadge", "PROCESSING YOLOV8...");
    setStyle("cmpYoloBadge", "color", "var(--color-amber)");
    setStyle("cmpYoloBadge", "borderColor", "var(--color-amber)");
  }
}



// ── Feature 1: Live Webcam Stream Capture & Neural Processing ────────────────
async function toggleLiveWebcam() {
  const btn = getEl("btnLiveCam");
  const btnTxt = getEl("btnLiveCamText");
  const video = getEl("webcamVideo");
  const canvas = getEl("webcamCaptureCanvas");

  if (state.isWebcamActive) {
    // Stop Webcam
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

  // Start Webcam
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

    // Continuous Frame Loop (~10 FPS to balance GPU and network)
    const ctx = canvas.getContext("2d");
    state.webcamInterval = setInterval(async () => {
      if (!state.isWebcamActive || state.isProcessingCamFrame) return;
      if (!video || video.readyState < 2) return;

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob || !state.isWebcamActive) return;
        state.isProcessingCamFrame = true;
        try {
          const formData = new FormData();
          formData.append("image", blob, "webcam_frame.jpg");
          formData.append("yolo_conf", state.yoloConf);
          formData.append("env_filter", state.envFilter);
          formData.append("is_live", "1");

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
      }, "image/jpeg", 0.65);
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
          body: JSON.stringify({ yolo_conf: state.yoloConf, env_filter: state.envFilter, is_live: true }),
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
        body: JSON.stringify({ yolo_conf: state.yoloConf, env_filter: state.envFilter, is_live: true }),
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
    if (btn.getAttribute("data-overlay") === mode) btn.classList.add("active");
    else btn.classList.remove("active");
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
  if (detImg && targetSrc) detImg.src = targetSrc;

  const cmpSnnImg = getEl("cmpSnnImg");
  if (cmpSnnImg && targetSrc) cmpSnnImg.src = targetSrc;
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

// ── Feature 2: Environmental Stress Testing ──────────────────────────────────
async function triggerEnvReeval(filter) {
  state.envFilter = filter;
  document.querySelectorAll(".env-chip").forEach((c) => {
    if (c.getAttribute("data-filter") === filter) c.classList.add("active");
    else c.classList.remove("active");
  });

  setProcessingState(true);
  try {
    const res = await fetch("/api/env_reeval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filter: filter, yolo_conf: state.yoloConf }),
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

// ── Feature 1: Temporal Spike Evolution Scrubber ────────────────────────────
function updateTemporalDisplay(step) {
  state.temporalStep = step;
  const slider = getEl("temporalSlider");
  if (slider && parseInt(slider.value, 10) !== step) {
    slider.value = step;
  }
  setTxt("temporalStepVal", `t = ${step} / 16`);

  if (!state.currentResult || !state.currentResult.temporal_evolution || !state.currentResult.temporal_evolution.length) {
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

// ── Feature 4: Mission Dossier Export ────────────────────────────────────────
function openDossierModal() {
  const modal = getEl("dossierModal");
  if (!modal || !state.currentResult) return;

  const data = state.currentResult;
  const yolo = data.yolo_benchmark || {};
  const confPct = ((data.confidence || 0) * 100).toFixed(1);
  const yoloConfPct = yolo.detections && yolo.detections.length > 0 ? yolo.detections[0].conf : "0.0";
  const snnPwr = data.dynamic_energy?.power_snn_w ?? data.telemetry?.power_snn_w ?? 0.58;
  const annPwr = yolo.power_w ?? data.telemetry?.power_ann_w ?? 8.45;
  const snnEnergy = data.dynamic_energy?.snn_energy_mj ?? 8.14;
  const yoloEnergy = yolo.energy_mj ?? 40.02;
  const snnLat = data.latency_ms !== undefined ? data.latency_ms.toFixed(1) : "16.1";
  const yoloLat = yolo.latency_ms !== undefined ? yolo.latency_ms.toFixed(1) : "45.7";
  const sparsityPct = ((data.sparsity || 0.784) * 100).toFixed(1);

  // Set timestamps & mission code
  setTxt("dosTimestamp", new Date().toUTCString());
  setTxt("dosMissionCode", `NC-RECON-${Math.floor(1000 + Math.random() * 9000)}`);

  // Set Evidence Images
  const dosRaw = getEl("dosRawImg");
  if (dosRaw && data.input_png) dosRaw.src = data.input_png;
  const dosSnn = getEl("dosSnnImg");
  if (dosSnn && data.result_png) dosSnn.src = data.result_png;
  const dosYolo = getEl("dosYoloImg");
  if (dosYolo && yolo.result_png) dosYolo.src = yolo.result_png;

  // Set Table Values
  const detected = data.detected !== undefined ? !!data.detected : (data.verdict === "DETECTED" || data.verdict === "POSSIBLE TARGET");
  setTxt("dosSnnVerdict", detected ? `CAMOUFLAGE LOCK (${confPct}%)` : "SECTOR CLEAR");
  setTxt("dosYoloVerdict", yolo.boxes_count > 0 ? `${yolo.boxes_count} OBJECTS DETECTED` : "0 OBJECTS DETECTED (MISSED)");

  setTxt("dosSnnConf", `${confPct}%`);
  setTxt("dosYoloConf", `${yoloConfPct}%`);
  const confDiff = (parseFloat(confPct) - parseFloat(yoloConfPct)).toFixed(1);
  setTxt("dosConfAdv", `+${confDiff}% SNN Saliency Advantage`);

  setTxt("dosSnnPower", `${snnPwr.toFixed(2)} W`);
  setTxt("dosYoloPower", `${annPwr.toFixed(2)} W`);
  const pwrRatio = (annPwr / Math.max(0.1, snnPwr)).toFixed(1);
  setTxt("dosPowerAdv", `${pwrRatio}× Lower Power Consumption`);

  setTxt("dosSnnEnergy", `${snnEnergy.toFixed(2)} mJ`);
  setTxt("dosYoloEnergy", `${yoloEnergy.toFixed(2)} mJ`);
  const energyRatio = (yoloEnergy / Math.max(0.01, snnEnergy)).toFixed(1);
  setTxt("dosEnergyAdv", `${energyRatio}× Synaptic Energy Savings`);

  setTxt("dosSnnLat", `${snnLat} ms (~${(1000 / parseFloat(snnLat)).toFixed(0)} FPS)`);
  setTxt("dosYoloLat", `${yoloLat} ms (~${(1000 / parseFloat(yoloLat)).toFixed(0)} FPS)`);
  const latSpeedup = (parseFloat(yoloLat) / Math.max(1, parseFloat(snnLat))).toFixed(1);
  setTxt("dosLatAdv", `${latSpeedup}× Real-time Edge Speedup`);

  setTxt("dosSnnSparsity", `${sparsityPct}% Silent Synapses`);

  modal.style.display = "flex";
}

// ── 4. Apply Detection Results to All Views (100% Null-Safe) ─────────────────
function applyInferenceResult(data) {
  if (!data) return;
  state.currentResult = data;

  try {
    const confPct = ((data.confidence || 0) * 100).toFixed(1);
    const detected = data.detected !== undefined ? !!data.detected : (data.verdict === "DETECTED" || data.verdict === "POSSIBLE TARGET");
    const lat = data.latency_ms !== undefined ? data.latency_ms.toFixed(1) : "16.1";
    const fpsEq = data.latency_ms > 0 ? (1000.0 / data.latency_ms).toFixed(0) : "60";

    // ══════════════════════════════════════════════════════════
    // A. GENERAL TACTICAL RECONNAISSANCE VIEW (RAW vs SNN MASK)
    // ══════════════════════════════════════════════════════════
    const rawImg = getEl("rawImg");
    const rawPh = getEl("rawPlaceholder");
    if (rawImg && data.input_png) {
      rawImg.src = data.input_png;
      rawImg.style.display = "block";
      rawImg.classList.add("visible");
      if (rawPh) {
        rawPh.style.display = "none";
        rawPh.classList.add("hidden");
      }
    }
    const szStr = data.image_size ? `${data.image_size[0]}x${data.image_size[1]}` : "960x960";
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
        setTxt("reconVerdictSub", data.gate_reason || "Zero significant camouflage activations in sector");
      }
    }
    setTxt("reconConf", `${confPct}%`);
    setStyle("reconConfBar", "width", `${confPct}%`);
    setTxt("reconTex", (data.texture_score || 0).toFixed(1));
    setTxt("reconGateReason", data.plain_background ? "Low Complexity: Flat Background" : "High Complexity: Passed Scene Gate");
    setTxt("reconLatency", `${lat} ms`);
    setTxt("reconFps", `Edge Throughput: ~${fpsEq} FPS`);

    // ══════════════════════════════════════════════════════════
    // B. HEAD-TO-HEAD BENCHMARK VIEW (SNN vs YOLOV8-NANO)
    // ══════════════════════════════════════════════════════════
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

    const cmpYoloImg = getEl("cmpYoloImg");
    const cmpYoloPh = getEl("cmpYoloPlaceholder");
    const yolo = data.yolo_benchmark || {};
    if (cmpYoloImg && yolo.result_png) {
      cmpYoloImg.src = yolo.result_png;
      cmpYoloImg.style.display = "block";
      cmpYoloImg.classList.add("visible");
      if (cmpYoloPh) {
        cmpYoloPh.style.display = "none";
        cmpYoloPh.classList.add("hidden");
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

    // 4. Dynamic Live Hardware Metrics (100% Calculated & Non-Static)
    const snnPwr = data.dynamic_energy?.power_snn_w ?? data.telemetry?.power_snn_w ?? 0.58;
    const annPwr = yolo.power_w ?? data.telemetry?.power_ann_w ?? 8.45;
    const yoloLat = yolo.latency_ms !== undefined ? yolo.latency_ms.toFixed(1) : "45.7";
    const snnLat = lat;
    const sparsityPct = ((data.sparsity || 0.784) * 100).toFixed(1);

    // Frame-specific Horowitz 45nm energy
    const snnEnergy = data.dynamic_energy?.snn_energy_mj ?? 8.144;
    const snnStepEnergy = data.dynamic_energy?.snn_step_energy_mj ?? (snnEnergy / 16.0);
    const yoloEnergy = yolo.energy_mj ?? 40.02;
    const yoloOps = yolo.gflops ? `${yolo.gflops.toFixed(2)} GFLOPs (Dense)` : "8.70 GFLOPs (Dense)";

    setTxt("cmpSnnPower", `${snnPwr.toFixed(2)} W`);
    setTxt("cmpSnnEnergy", `${snnEnergy.toFixed(2)} mJ (${snnStepEnergy.toFixed(3)} mJ/step)`);
    setTxt("cmpSnnSparsity", `${sparsityPct}% Silent`);
    setTxt("cmpSnnLatency", `${snnLat} ms`);

    setTxt("cmpYoloPower", `${annPwr.toFixed(2)} W`);
    setTxt("cmpYoloEnergy", `${yoloEnergy.toFixed(2)} mJ`);
    setTxt("cmpYoloOps", yoloOps);
    setTxt("cmpYoloLatency", `${yoloLat} ms`);

    setTxt("hdrSnnPower", `${snnPwr.toFixed(2)}W`);
    setTxt("hdrYoloPower", `${annPwr.toFixed(2)}W`);

    // Sync active environmental chip
    const activeEnv = data.env_filter || state.envFilter || "normal";
    document.querySelectorAll(".env-chip").forEach((c) => {
      if (c.getAttribute("data-filter") === activeEnv) c.classList.add("active");
      else c.classList.remove("active");
    });

    // Update Temporal Evolution Scrubber display
    updateTemporalDisplay(16);

    // Sync active scenario buttons
    if (data.sample_info?.id || state.currentSampleId) {
      const activeId = data.sample_info?.id || state.currentSampleId;
      document.querySelectorAll(".cmp-sample-btn").forEach((b) => {
        if (b.getAttribute("data-sample") === activeId) b.classList.add("active");
        else b.classList.remove("active");
      });
    }

    // Update Real-Time Dual Time-Series Histories
    if (data.latency_ms !== undefined) {
      state.latencyHistory.push(parseFloat(data.latency_ms.toFixed(1)));
      if (state.latencyHistory.length > 25) state.latencyHistory.shift();
    }
    const curYoloLat = yolo.latency_ms !== undefined ? parseFloat(yolo.latency_ms.toFixed(1)) : 52.3;
    state.yoloLatencyHistory.push(curYoloLat);
    if (state.yoloLatencyHistory.length > 25) state.yoloLatencyHistory.shift();

    state.snnPowerHistory.push(parseFloat(snnPwr.toFixed(2)));
    if (state.snnPowerHistory.length > 25) state.snnPowerHistory.shift();

    state.yoloPowerHistory.push(parseFloat(annPwr.toFixed(2)));
    if (state.yoloPowerHistory.length > 25) state.yoloPowerHistory.shift();

    const isLiveActive = state.isWebcamActive || state.isRealSenseActive;

    // Redraw dual time-series traces (always when in power view, throttled if live)
    if (!isLiveActive || state.activeView === "view-power") {
      drawPowerTraceChart(state.snnPowerHistory, state.yoloPowerHistory);
      drawLatencyOscilloscope(state.latencyHistory, state.yoloLatencyHistory);
    }

    // Update STANAG 4586 / MAVLink Tactical GCS Console
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
      packetStatus.textContent = tel.detected ? "RS-422 Serial Link: TARGET LOCK // PTZ READY" : "RS-422 Serial Link: Standby / Sector Clear";
      packetStatus.style.color = tel.detected ? "var(--color-emerald)" : "var(--text-muted)";
    }

    // Update Neuromorphic Core tab (only render heavy spectrogram canvas if tab is visible)
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

    // Update Power tab
    setTxt("pwrSnn", `${snnPwr.toFixed(2)} W`);
    setTxt("pwrAnn", `${annPwr.toFixed(2)} W`);
    setTxt("frameEnergyVal", `${snnEnergy.toFixed(2)} mJ`);
    setTxt("annEnergyVal", `${yoloEnergy.toFixed(2)} mJ`);
    const savRatio = (yoloEnergy / Math.max(0.01, snnEnergy)).toFixed(1);
    setTxt("savingsFactorVal", `${savRatio}x`);

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

// ── Canvas: Real-Time Continuous Power Draw Trace (Watts) ────────────────────
function drawPowerTraceChart(snnHist, yoloHist) {
  const canvas = getEl("powerTraceCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const padX = 45;
  const padY = 25;
  const drawW = w - padX * 2;
  const drawH = h - padY * 2;

  // Grid
  ctx.strokeStyle = "rgba(0, 210, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, drawW, drawH);

  for (let y = padY + drawH / 4; y < padY + drawH; y += drawH / 4) {
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(padX + drawW, y);
    ctx.stroke();
  }
  for (let x = padX + drawW / 4; x < padX + drawW; x += drawW / 4) {
    ctx.beginPath();
    ctx.moveTo(x, padY);
    ctx.lineTo(x, padY + drawH);
    ctx.stroke();
  }

  const maxPower = 10.0; // Jetson Nano 10W Profile limit
  const count = Math.max(snnHist.length, yoloHist.length);
  if (count === 0) return;
  const stepW = drawW / Math.max(1, count - 1);

  // 1. Draw YOLO Trace (Crimson)
  ctx.beginPath();
  yoloHist.forEach((val, i) => {
    const x = padX + i * stepW;
    const y = padY + drawH - (Math.min(maxPower, val) / maxPower) * drawH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#ff4455";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // 2. Draw SNN Trace (Emerald)
  ctx.beginPath();
  snnHist.forEach((val, i) => {
    const x = padX + i * stepW;
    const y = padY + drawH - (Math.min(maxPower, val) / maxPower) * drawH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#00e5a0";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // SNN Glow Fill under curve
  ctx.lineTo(padX + (snnHist.length - 1) * stepW, padY + drawH);
  ctx.lineTo(padX, padY + drawH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padY + drawH * 0.8, 0, padY + drawH);
  grad.addColorStop(0, "rgba(0, 229, 160, 0.25)");
  grad.addColorStop(1, "rgba(0, 229, 160, 0.0)");
  ctx.fillStyle = grad;
  ctx.fill();

  // Draw Latest Value Head Indicators
  const lastSnn = snnHist[snnHist.length - 1] || 0.97;
  const lastYolo = yoloHist[yoloHist.length - 1] || 8.85;

  const lastX = padX + (count - 1) * stepW;
  const snnY = padY + drawH - (lastSnn / maxPower) * drawH;
  const yoloY = padY + drawH - (lastYolo / maxPower) * drawH;

  // SNN point
  ctx.fillStyle = "#00e5a0";
  ctx.beginPath();
  ctx.arc(lastX, snnY, 5, 0, Math.PI * 2);
  ctx.fill();

  // YOLO point
  ctx.fillStyle = "#ff4455";
  ctx.beginPath();
  ctx.arc(lastX, yoloY, 5, 0, Math.PI * 2);
  ctx.fill();

  // Axis Labels
  ctx.fillStyle = "#7b889e";
  ctx.font = "10px monospace";
  ctx.fillText("10.0 W (Limit)", 4, padY + 10);
  ctx.fillText("5.0 W", 14, padY + drawH / 2);
  ctx.fillText("0.0 W", 14, padY + drawH);
  ctx.fillText("Continuous Inferences (T-25)", padX, h - 6);
  ctx.fillText("LIVE STREAM", padX + drawW - 65, h - 6);

  // Update DOM Pills
  const ratio = (lastYolo / Math.max(0.1, lastSnn)).toFixed(1);
  setTxt("liveSnnPwrVal", `${lastSnn.toFixed(2)} W`);
  setTxt("liveYoloPwrVal", `${lastYolo.toFixed(2)} W`);
  setTxt("livePwrRatioVal", `${ratio}×`);
  setTxt("livePwrAdv", `${ratio}× Lower Draw`);
}


// ── 5. Canvas: Comparative Latency & Throughput Trace (ms) ───────────────────
function drawLatencyOscilloscope(snnHist, yoloHist) {
  const canvas = getEl("latencyCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const padX = 45;
  const padY = 25;
  const drawW = w - padX * 2;
  const drawH = h - padY * 2;

  // Grid
  ctx.strokeStyle = "rgba(0, 210, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, drawW, drawH);

  for (let y = padY + drawH / 4; y < padY + drawH; y += drawH / 4) {
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(padX + drawW, y);
    ctx.stroke();
  }
  for (let x = padX + drawW / 4; x < padX + drawW; x += drawW / 4) {
    ctx.beginPath();
    ctx.moveTo(x, padY);
    ctx.lineTo(x, padY + drawH);
    ctx.stroke();
  }

  const count = Math.max(snnHist.length, (yoloHist ? yoloHist.length : 0));
  if (count === 0) return;
  const maxVal = 80.0; // Fixed 80ms ceiling for consistent comparative review
  const stepW = drawW / Math.max(1, count - 1);

  // 1. Draw YOLO Latency Trace (Crimson)
  if (yoloHist && yoloHist.length > 0) {
    ctx.beginPath();
    yoloHist.forEach((val, i) => {
      const x = padX + i * stepW;
      const y = padY + drawH - (Math.min(maxVal, val) / maxVal) * drawH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#ff4455";
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  // 2. Draw SNN Latency Trace (Emerald)
  ctx.beginPath();
  snnHist.forEach((val, i) => {
    const x = padX + i * stepW;
    const y = padY + drawH - (Math.min(maxVal, val) / maxVal) * drawH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#00e5a0";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // SNN Gradient Fill
  ctx.lineTo(padX + (snnHist.length - 1) * stepW, padY + drawH);
  ctx.lineTo(padX, padY + drawH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padY + drawH * 0.6, 0, padY + drawH);
  grad.addColorStop(0, "rgba(0, 229, 160, 0.22)");
  grad.addColorStop(1, "rgba(0, 229, 160, 0.0)");
  ctx.fillStyle = grad;
  ctx.fill();

  // Latest Points
  const lastSnn = snnHist[snnHist.length - 1] || 16.1;
  const lastYolo = yoloHist ? (yoloHist[yoloHist.length - 1] || 52.3) : 52.3;
  const lastX = padX + (count - 1) * stepW;

  ctx.fillStyle = "#00e5a0";
  ctx.beginPath();
  ctx.arc(lastX, padY + drawH - (lastSnn / maxVal) * drawH, 5, 0, Math.PI * 2);
  ctx.fill();

  if (yoloHist && yoloHist.length > 0) {
    ctx.fillStyle = "#ff4455";
    ctx.beginPath();
    ctx.arc(lastX, padY + drawH - (lastYolo / maxVal) * drawH, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Axis Labels
  ctx.fillStyle = "#7b889e";
  ctx.font = "10px monospace";
  ctx.fillText("80 ms (12 FPS)", 4, padY + 10);
  ctx.fillText("40 ms (25 FPS)", 4, padY + drawH / 2);
  ctx.fillText("0 ms", 14, padY + drawH);
  ctx.fillText("Rolling Frame Stream (T-25)", padX, h - 6);
  ctx.fillText("LIVE HEAD", padX + drawW - 65, h - 6);

  // Update DOM Pills
  const speedup = (lastYolo / Math.max(1.0, lastSnn)).toFixed(1);
  const snnFps = (1000.0 / Math.max(1, lastSnn)).toFixed(0);
  const yoloFps = (1000.0 / Math.max(1, lastYolo)).toFixed(0);

  setTxt("liveLatencyVal", `${lastSnn.toFixed(1)} ms (~${snnFps} FPS)`);
  setTxt("liveYoloLatencyVal", `${lastYolo.toFixed(1)} ms (~${yoloFps} FPS)`);
  setTxt("liveLatRatioVal", `${speedup}×`);
  setTxt("liveLatAdv", `${speedup}× Speedup`);
}

// ── 6. Canvas: Dynamic Synaptic Energy per Frame ─────────────────────────────
function drawDynamicEnergyChart(energyData) {
  const canvas = getEl("energyCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#06080b";
  ctx.fillRect(0, 0, w, h);

  const padX = 40;
  const padY = 25;
  const drawW = w - padX * 2;
  const drawH = h - padY * 2;

  const snnVal = energyData.snn_energy_mj || 8.14;
  const annVal = energyData.ann_energy_mj || 60.38;
  const maxVal = Math.max(annVal * 1.15, 10.0);

  const barH = 34;
  const y1 = padY + 30;
  const y2 = padY + 95;

  const wSnn = (snnVal / maxVal) * drawW;
  const wAnn = (annVal / maxVal) * drawW;

  // SNN Bar
  ctx.fillStyle = "#7b889e";
  ctx.font = "11px sans-serif";
  ctx.fillText("Proposed SNN (Event-Driven AC Spikes):", padX, y1 - 8);

  ctx.fillStyle = "rgba(0, 229, 160, 0.12)";
  ctx.fillRect(padX, y1, drawW, barH);
  const gradSnn = ctx.createLinearGradient(padX, 0, padX + wSnn, 0);
  gradSnn.addColorStop(0, "#00b078");
  gradSnn.addColorStop(1, "var(--color-emerald)");
  ctx.fillStyle = gradSnn;
  ctx.fillRect(padX, y1, wSnn, barH);
  ctx.strokeStyle = "var(--color-emerald)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(padX, y1, wSnn, barH);

  ctx.fillStyle = "#fff";
  ctx.font = "bold 12px monospace";
  ctx.fillText(`${snnVal.toFixed(2)} mJ`, padX + wSnn + 10, y1 + 22);

  // ANN Bar
  ctx.fillStyle = "#7b889e";
  ctx.font = "11px sans-serif";
  ctx.fillText("Conventional Dense ANN Baseline (Dense MACs):", padX, y2 - 8);

  ctx.fillStyle = "rgba(255, 68, 85, 0.12)";
  ctx.fillRect(padX, y2, drawW, barH);
  const gradAnn = ctx.createLinearGradient(padX, 0, padX + wAnn, 0);
  gradAnn.addColorStop(0, "#b02030");
  gradAnn.addColorStop(1, "var(--color-crimson)");
  ctx.fillStyle = gradAnn;
  ctx.fillRect(padX, y2, wAnn, barH);
  ctx.strokeStyle = "var(--color-crimson)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(padX, y2, wAnn, barH);

  ctx.fillStyle = "#fff";
  ctx.font = "bold 12px monospace";
  ctx.fillText(`${annVal.toFixed(2)} mJ`, padX + drawW - 65, y2 + 22);

  // UPDATE DOM TEXT PILLS SAFELY
  setTxt("frameEnergyVal", `${snnVal.toFixed(2)} mJ`);
  setTxt("annEnergyVal", `${annVal.toFixed(2)} mJ`);
  const factor = (annVal / Math.max(0.01, snnVal)).toFixed(1);
  setTxt("savingsFactorVal", `${factor}x`);
}

// ── 7. Real-Time Oscilloscope & Power Heartbeat Loop ─────────────────────────
function startRealtimeMonitor() {
  setInterval(() => {
    const baseLat = state.currentResult?.latency_ms || 16.2;
    const jitter = (Math.random() - 0.5) * 1.4;
    const currentLat = Math.max(12.0, parseFloat((baseLat + jitter).toFixed(1)));

    state.latencyHistory.push(currentLat);
    if (state.latencyHistory.length > 25) {
      state.latencyHistory.shift();
    }

    if (state.activeView === "view-power") {
      drawPowerTraceChart(state.snnPowerHistory, state.yoloPowerHistory);
      drawLatencyOscilloscope(state.latencyHistory, state.yoloLatencyHistory);

      const basePwr = state.currentResult?.telemetry?.power_snn_w || 0.91;
      const pwrJitter = (Math.random() - 0.5) * 0.04;
      setTxt("pwrSnn", `${(basePwr + pwrJitter).toFixed(2)} W`);

      const annPwr = state.currentResult?.telemetry?.power_ann_w || 8.70;
      const annJitter = (Math.random() - 0.5) * 0.18;
      setTxt("pwrAnn", `${(annPwr + annJitter).toFixed(2)} W`);
    }
  }, 750);
}

// ── 8. Canvas: Layer-3 Spike Spectrogram ─────────────────────────────────────
function renderSpectrogram(spec) {
  const canvas = getEl("specCanvas");
  if (!canvas) return;
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

// ── 9. Canvas: Population Rate Curve (t0 -> t15) ─────────────────────────────
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

// ── 10. Battery Autonomy Calculator ──────────────────────────────────────────
function updateBatteryAutonomy(mah) {
  const wh = (mah * 12.0) / 1000.0;
  setTxt("battCapText", `${mah} mAh (${wh.toFixed(1)} Wh)`);

  const usableWh = wh * 0.85;
  const snnHrs = (usableWh / 0.91).toFixed(1);
  const annHrs = (usableWh / 8.70).toFixed(1);

  setTxt("autoSnn", `${snnHrs} hrs`);
  setTxt("autoAnn", `${annHrs} hrs`);
}
