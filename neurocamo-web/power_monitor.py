"""
NeuroCamo Web — Real Hardware Power & Telemetry Monitor
=======================================================
Provides real-time, non-faked hardware power, utilization, and trapezoidal
energy measurements across Jetson edge platforms and CUDA/CPU workstations.

Hardware telemetry hierarchy (probed once at startup, graceful fallback):
  1. Jetson / Linux INA3221 / INA219 / INA226 sysfs rails (mW / uW / mV*mA)
  2. Jetson `tegrastats` subprocess parser (Linux only)
  3. NVIDIA NVML (`pynvml` via `nvidia-ml-py`) GPU power & utilization
  4. Honest `psutil` system/process telemetry power estimation (`is_estimated=True`)

Zero `random`, zero `sin`/`cos`, zero hardcoded fluctuations.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import psutil

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None  # type: ignore[assignment]


PHASES_ORDER = (
    "idle_baseline",
    "snn_inference",
    "cooldown",
    "yolo_inference",
    "post_idle",
)


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def integrate_energy_j(points: list[tuple[float, float]]) -> float:
    """Pure-Python trapezoidal integration over (elapsed_s, watts).

    Avoids NumPy version differences (np.trapz was removed in NumPy 2.4 while
    Jetson NumPy 1.26 lacks np.trapezoid).
    """
    if len(points) < 2:
        return 0.0
    total = 0.0
    prev_t, prev_w = points[0]
    for cur_t, cur_w in points[1:]:
        dt = float(cur_t) - float(prev_t)
        if dt > 0.0:
            total += 0.5 * (float(prev_w) + float(cur_w)) * dt
        prev_t, prev_w = cur_t, cur_w
    return float(total)


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_float(path: str) -> float | None:
    txt = _read_text(path)
    if txt is None or txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


@dataclass
class HardwareSnapshot:
    watts: float
    cpu_pct: float
    gpu_util_pct: float | None
    mem_pct: float
    source: str
    is_estimated: bool
    sensor_detail: str
    gpu_temp_c: float | None = None
    gpu_mem_pct: float | None = None


class _SysfsInaReader:
    """Tier 1: Linux / Jetson INA3221 / INA219 / INA226 sysfs rail reader."""

    _TOTAL_LABELS = ("VDD_IN", "POM_5V_IN", "VIN_SYS_5V0", "VDD_MUX", "TOTAL", "MAIN")

    def __init__(self) -> None:
        self.source: str = "INA3221"
        self.sensor_detail: str = ""
        self._mode: str = ""  # "iio_mw", "hwmon_uw", "hwmon_mv_ma"
        self._rails: list[tuple[str, str, str | None]] = []
        self._gpu_load_path: str | None = self._find_jetson_gpu_load()
        self._probe()

    @staticmethod
    def _find_jetson_gpu_load() -> str | None:
        candidates = [
            "/sys/devices/gpu.0/load",
            "/sys/devices/platform/gpu.0/load",
            "/sys/devices/17000000.gv11b/load",
            "/sys/devices/17000000.ga10b/load",
        ]
        for p in candidates:
            if os.path.exists(p) and _read_float(p) is not None:
                return p
        return None

    @staticmethod
    def _chip_source_name(name: str) -> str:
        low = name.lower()
        if "ina219" in low:
            return "INA219"
        if "ina226" in low:
            return "INA226"
        return "INA3221"

    def _is_total_label(self, label: str) -> bool:
        up = (label or "").upper()
        return any(tok in up for tok in self._TOTAL_LABELS)

    def _probe(self) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("sysfs INA rails only supported on Linux")

        # 1a. Jetson IIO INA3221x interface (JetPack 4 / Xavier / Nano)
        iio_dirs = sorted(
            set(
                glob.glob("/sys/bus/i2c/drivers/ina3221x/*/iio:device*")
                + glob.glob("/sys/bus/i2c/drivers/ina3221*/*/iio:device*")
                + glob.glob("/sys/bus/i2c/devices/*/iio:device*")
            )
        )
        for d in iio_dirs:
            name_txt = _read_text(os.path.join(d, "name")) or os.path.basename(d)
            if "ina" not in name_txt.lower() and "ina3221" not in d.lower():
                continue
            power_files = sorted(glob.glob(os.path.join(d, "in_power*_input")))
            if not power_files:
                continue
            rails: list[tuple[str, str, str | None]] = []
            for pf in power_files:
                if _read_float(pf) is None:
                    continue
                m = re.search(r"in_power(\d+)_input$", pf)
                idx = m.group(1) if m else "0"
                label = (
                    _read_text(os.path.join(d, f"rail_name_{idx}"))
                    or _read_text(os.path.join(d, f"in_power{idx}_label"))
                    or f"rail_{idx}"
                )
                rails.append((label, pf, None))
            if rails:
                total_rails = [r for r in rails if self._is_total_label(r[0])]
                self._rails = total_rails[:1] if total_rails else rails
                self._mode = "iio_mw"
                self.source = self._chip_source_name(name_txt)
                self.sensor_detail = f"{d} ({', '.join(r[0] for r in self._rails)})"
                return

        # 1b. Linux hwmon INA3221 / INA219 / INA226 (JetPack 5/6 & standard Linux)
        hwmon_dirs = sorted(
            set(
                glob.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*")
                + glob.glob("/sys/class/hwmon/hwmon*")
            )
        )
        for d in hwmon_dirs:
            chip_name = _read_text(os.path.join(d, "name")) or ""
            if not any(k in chip_name.lower() for k in ("ina3221", "ina219", "ina226")):
                continue

            # Try direct power<n>_input (microwatts)
            pw_files = sorted(glob.glob(os.path.join(d, "power*_input")))
            pw_rails: list[tuple[str, str, str | None]] = []
            for pf in pw_files:
                if _read_float(pf) is None:
                    continue
                m = re.search(r"power(\d+)_input$", pf)
                idx = m.group(1) if m else "1"
                label = (
                    _read_text(os.path.join(d, f"power{idx}_label"))
                    or _read_text(os.path.join(d, f"in{idx}_label"))
                    or f"ch{idx}"
                )
                pw_rails.append((label, pf, None))

            if pw_rails:
                total_rails = [r for r in pw_rails if self._is_total_label(r[0])]
                self._rails = total_rails[:1] if total_rails else pw_rails
                self._mode = "hwmon_uw"
                self.source = self._chip_source_name(chip_name)
                self.sensor_detail = f"{d} ({', '.join(r[0] for r in self._rails)})"
                return

            # Fallback to in<n>_input (mV) * curr<n>_input (mA)
            in_files = sorted(glob.glob(os.path.join(d, "in*_input")))
            vi_rails: list[tuple[str, str, str | None]] = []
            for vf in in_files:
                m = re.search(r"in(\d+)_input$", vf)
                if not m:
                    continue
                idx = m.group(1)
                cf = os.path.join(d, f"curr{idx}_input")
                if not os.path.exists(cf):
                    continue
                if _read_float(vf) is None or _read_float(cf) is None:
                    continue
                label = _read_text(os.path.join(d, f"in{idx}_label")) or f"ch{idx}"
                vi_rails.append((label, vf, cf))

            if vi_rails:
                total_rails = [r for r in vi_rails if self._is_total_label(r[0])]
                self._rails = total_rails[:1] if total_rails else vi_rails
                self._mode = "hwmon_mv_ma"
                self.source = self._chip_source_name(chip_name)
                self.sensor_detail = f"{d} ({', '.join(r[0] for r in self._rails)})"
                return

        raise RuntimeError("No readable INA3221/INA219/INA226 sysfs rail found")

    def read(self) -> HardwareSnapshot:
        watts = 0.0
        valid = 0
        for _, p1, p2 in self._rails:
            if self._mode == "iio_mw":
                v = _read_float(p1)
                if v is not None:
                    watts += v / 1000.0
                    valid += 1
            elif self._mode == "hwmon_uw":
                v = _read_float(p1)
                if v is not None:
                    watts += v / 1_000_000.0
                    valid += 1
            elif self._mode == "hwmon_mv_ma" and p2 is not None:
                mv = _read_float(p1)
                ma = _read_float(p2)
                if mv is not None and ma is not None:
                    watts += (mv * ma) / 1_000_000.0
                    valid += 1

        if valid == 0:
            raise OSError("INA sysfs sensor returned no readable values")

        gpu_pct: float | None = None
        if self._gpu_load_path:
            g_raw = _read_float(self._gpu_load_path)
            if g_raw is not None:
                gpu_pct = round(min(100.0, max(0.0, g_raw / 10.0)), 1)

        cpu_pct = float(psutil.cpu_percent(interval=None))
        mem_pct = float(psutil.virtual_memory().percent)
        return HardwareSnapshot(
            watts=round(watts, 3),
            cpu_pct=round(cpu_pct, 1),
            gpu_util_pct=gpu_pct,
            mem_pct=round(mem_pct, 1),
            source=self.source,
            is_estimated=False,
            sensor_detail=self.sensor_detail,
        )

    def close(self) -> None:
        pass


class _TegrastatsReader:
    """Tier 2: Jetson `tegrastats --interval 150` subprocess reader (Linux only)."""

    _RE_VDD_IN = re.compile(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)(?:mW)?/")
    _RE_RAIL_MW = re.compile(r"(VDD_GPU_SOC|VDD_CPU_CV|VIN_SYS_5V0|VDD_SYS_GPU|VDD_SYS_CPU|VDD_SYS_SOC)\s+(\d+)(?:mW)?/")
    _RE_GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")
    _RE_CPU = re.compile(r"CPU\s+\[([^\]]+)\]")
    _RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")

    def __init__(self) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("tegrastats only available on Linux")
        exe = shutil.which("tegrastats")
        if not exe:
            raise RuntimeError("tegrastats binary not found")

        self._lock = threading.Lock()
        self._latest: HardwareSnapshot | None = None
        self._stop = threading.Event()
        self._proc = subprocess.Popen(
            [exe, "--interval", "150"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="tegrastats-reader")
        self._thread.start()

        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest is not None:
                    return
            if self._proc.poll() is not None:
                break
            time.sleep(0.05)

        self.close()
        raise RuntimeError("tegrastats produced no valid power reading within 1.0s")

    def _parse_line(self, line: str) -> HardwareSnapshot | None:
        watts: float | None = None
        m_total = self._RE_VDD_IN.search(line)
        if m_total:
            watts = float(m_total.group(1)) / 1000.0
        else:
            rails = self._RE_RAIL_MW.findall(line)
            if rails:
                watts = sum(float(mw) for _, mw in rails) / 1000.0
        if watts is None:
            return None

        gpu_pct: float | None = None
        m_gpu = self._RE_GR3D.search(line)
        if m_gpu:
            gpu_pct = float(m_gpu.group(1))

        cpu_pct = float(psutil.cpu_percent(interval=None))
        m_cpu = self._RE_CPU.search(line)
        if m_cpu:
            vals = [float(x) for x in re.findall(r"(\d+)%@", m_cpu.group(1))]
            if vals:
                cpu_pct = sum(vals) / len(vals)

        mem_pct = float(psutil.virtual_memory().percent)
        m_ram = self._RE_RAM.search(line)
        if m_ram:
            used_mb, total_mb = float(m_ram.group(1)), float(m_ram.group(2))
            if total_mb > 0:
                mem_pct = (used_mb / total_mb) * 100.0

        return HardwareSnapshot(
            watts=round(watts, 3),
            cpu_pct=round(cpu_pct, 1),
            gpu_util_pct=round(gpu_pct, 1) if gpu_pct is not None else None,
            mem_pct=round(mem_pct, 1),
            source="tegrastats",
            is_estimated=False,
            sensor_detail="tegrastats --interval 150",
        )

    def _reader_loop(self) -> None:
        stdout = self._proc.stdout
        if stdout is None:
            return
        try:
            for line in stdout:
                if self._stop.is_set():
                    break
                snap = self._parse_line(line)
                if snap is not None:
                    with self._lock:
                        self._latest = snap
        except Exception:
            pass

    def read(self) -> HardwareSnapshot:
        with self._lock:
            if self._latest is None or self._proc.poll() is not None:
                raise RuntimeError("tegrastats stream unavailable")
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass


class _NvmlReader:
    """Tier 3: NVIDIA NVML (`pynvml`) hardware GPU board power reader."""

    def __init__(self) -> None:
        import pynvml  # noqa: PLC0415

        self._nvml = pynvml
        self._initialized = False
        self._nvml.nvmlInit()
        self._initialized = True

        count = self._nvml.nvmlDeviceGetCount()
        if count <= 0:
            self.close()
            raise RuntimeError("No NVIDIA NVML GPU detected")

        dev_idx = 0
        if torch is not None and torch.cuda.is_available():
            try:
                dev_idx = min(count - 1, max(0, int(torch.cuda.current_device())))
            except Exception:
                dev_idx = 0

        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(dev_idx)
        # Validate that power reading itself is supported before accepting Tier 3.
        try:
            mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
            if mw is None or mw < 0:
                raise RuntimeError("Invalid NVML power reading")
        except Exception as exc:
            self.close()
            raise RuntimeError(f"NVML power draw unsupported: {exc}") from exc

        raw_name = self._nvml.nvmlDeviceGetName(self._handle)
        if isinstance(raw_name, bytes):
            self.sensor_detail = raw_name.decode("utf-8", errors="replace")
        else:
            self.sensor_detail = str(raw_name)

        # Prime psutil CPU percent baseline
        psutil.cpu_percent(interval=None)

    def read(self) -> HardwareSnapshot:
        power_mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
        watts = float(power_mw) / 1000.0

        gpu_util: float | None = None
        gpu_mem: float | None = None
        gpu_temp: float | None = None

        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            gpu_util = float(util.gpu)
            gpu_mem = float(util.memory)
        except Exception:
            pass

        try:
            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            if mem_info.total > 0:
                gpu_mem = (float(mem_info.used) / float(mem_info.total)) * 100.0
        except Exception:
            pass

        try:
            gpu_temp = float(
                self._nvml.nvmlDeviceGetTemperature(self._handle, self._nvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass

        cpu_pct = float(psutil.cpu_percent(interval=None))
        mem_pct = float(psutil.virtual_memory().percent)

        return HardwareSnapshot(
            watts=round(watts, 3),
            cpu_pct=round(cpu_pct, 1),
            gpu_util_pct=round(gpu_util, 1) if gpu_util is not None else None,
            mem_pct=round(mem_pct, 1),
            source="NVML",
            is_estimated=False,
            sensor_detail=self.sensor_detail,
            gpu_temp_c=round(gpu_temp, 1) if gpu_temp is not None else None,
            gpu_mem_pct=round(gpu_mem, 1) if gpu_mem is not None else None,
        )

    def close(self) -> None:
        if self._initialized:
            self._initialized = False
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class _PsutilEstimatorReader:
    """Tier 4: Honest deterministic OS telemetry power estimator (`is_estimated=True`)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc = psutil.Process()
        self._n_cpus = max(1, int(psutil.cpu_count(logical=True) or 1))
        self._idle_w = float(os.environ.get("NEUROCAMO_EST_IDLE_W", "2.5"))
        self._tdp_w = float(os.environ.get("NEUROCAMO_EST_TDP_W", "15.0"))
        self._prev_times = self._proc.cpu_times()
        self._prev_t = time.perf_counter()
        psutil.cpu_percent(interval=None)

    def read(self) -> HardwareSnapshot:
        with self._lock:
            now = time.perf_counter()
            dt = max(1e-3, now - self._prev_t)
            cur_times = self._proc.cpu_times()
            proc_cpu_dt = max(
                0.0,
                (float(cur_times.user) - float(self._prev_times.user))
                + (float(cur_times.system) - float(self._prev_times.system)),
            )
            self._prev_times = cur_times
            self._prev_t = now

        sys_cpu_pct = min(100.0, max(0.0, float(psutil.cpu_percent(interval=None))))
        mem_pct = float(psutil.virtual_memory().percent)
        proc_cpu_norm = min(1.0, max(0.0, proc_cpu_dt / (dt * self._n_cpus)))

        freq_ratio = 1.0
        try:
            freq = psutil.cpu_freq()
            if freq is not None and freq.max and freq.max > 0:
                freq_ratio = min(1.2, max(0.3, float(freq.current) / float(freq.max)))
        except Exception:
            freq_ratio = 1.0

        load_factor = 0.65 * (sys_cpu_pct / 100.0) * freq_ratio + 0.35 * proc_cpu_norm
        est_w = self._idle_w + (self._tdp_w - self._idle_w) * max(0.0, min(1.25, load_factor))

        return HardwareSnapshot(
            watts=round(est_w, 3),
            cpu_pct=round(sys_cpu_pct, 1),
            gpu_util_pct=None,
            mem_pct=round(mem_pct, 1),
            source="psutil-estimated",
            is_estimated=True,
            sensor_detail=f"psutil CPU/process telemetry estimate ({self._n_cpus} logical cores)",
        )

    def close(self) -> None:
        pass


def _probe_hardware_reader() -> tuple[Any, list[str]]:
    """Select the highest-fidelity hardware telemetry reader available on this host."""
    probe_log: list[str] = []
    tiers: list[tuple[str, Callable[[], Any]]] = [
        ("INA_SYSFS", _SysfsInaReader),
        ("TEGRASTATS", _TegrastatsReader),
        ("NVML", _NvmlReader),
        ("PSUTIL_ESTIMATE", _PsutilEstimatorReader),
    ]
    for name, factory in tiers:
        try:
            reader = factory()
            # Verify one real read before locking in the tier
            snap = reader.read()
            probe_log.append(f"{name}: OK ({snap.source}, {snap.watts:.2f}W, est={snap.is_estimated})")
            return reader, probe_log
        except Exception as exc:
            probe_log.append(f"{name}: unavailable ({exc})")
    # Ultimate safety fallback (should never reach since _PsutilEstimatorReader always succeeds)
    est = _PsutilEstimatorReader()
    return est, probe_log


class PowerMonitor:
    """Thread-safe hardware telemetry monitor with 150ms session streaming & trapezoidal energy."""

    def __init__(
        self,
        interval_s: float = 0.15,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.interval_s = max(0.10, min(0.25, float(interval_s)))
        self._on_event = on_event

        self._reader, self.probe_log = _probe_hardware_reader()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._shutdown = False

        self._active_session_id: str | None = None
        self._active_image_name: str = ""
        self._active_started_utc: str = ""
        self._session_t0: float = 0.0
        self._current_phase: str = "idle"
        self._seq: int = 0
        self._samples: list[dict[str, Any]] = []
        self._phase_bounds: dict[str, list[float]] = {}
        self._next_sample_at: float = 0.0
        self._last_snapshot: HardwareSnapshot = self._reader.read()

        self._thread = threading.Thread(
            target=self._sampler_loop,
            daemon=True,
            name="neurocamo-power-sampler",
        )
        self._thread.start()
        atexit.register(self.close)

    def set_event_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        with self._lock:
            self._on_event = callback

    def get_capabilities(self) -> dict[str, Any]:
        snap = self._last_snapshot
        return {
            "source": snap.source,
            "is_estimated": snap.is_estimated,
            "sensor_detail": snap.sensor_detail,
            "interval_ms": round(self.interval_s * 1000.0, 1),
            "probe_log": list(self.probe_log),
        }

    def _safe_read_snapshot(self) -> HardwareSnapshot:
        try:
            snap = self._reader.read()
        except Exception:
            # If the primary hardware reader transiently/permanently fails mid-run,
            # fall back cleanly to psutil-estimated rather than crashing inference.
            if not isinstance(self._reader, _PsutilEstimatorReader):
                try:
                    self._reader.close()
                except Exception:
                    pass
                self._reader = _PsutilEstimatorReader()
                self.probe_log.append("RUNTIME_FALLBACK: switched to psutil-estimated after reader error")
            snap = self._reader.read()
        self._last_snapshot = snap
        return snap

    def read_live(self) -> dict[str, Any]:
        snap = self._safe_read_snapshot()
        with self._lock:
            sid = self._active_session_id
            phase = self._current_phase if sid else "idle"
        return {
            "type": "power_live",
            "session_id": sid,
            "phase": phase,
            "watts": snap.watts,
            "cpu_pct": snap.cpu_pct,
            "gpu_util_pct": snap.gpu_util_pct,
            "mem_pct": snap.mem_pct,
            "gpu_temp_c": snap.gpu_temp_c,
            "gpu_mem_pct": snap.gpu_mem_pct,
            "source": snap.source,
            "is_estimated": snap.is_estimated,
            "sensor_detail": snap.sensor_detail,
            "timestamp_utc": utc_iso_now(),
        }

    def get_active_session_info(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active_session_id is None:
                return None
            snap = self._last_snapshot
            return {
                "session_id": self._active_session_id,
                "image_name": self._active_image_name,
                "started_at_utc": self._active_started_utc,
                "phase": self._current_phase,
                "elapsed_s": round(time.perf_counter() - self._session_t0, 3),
                "source": snap.source,
                "is_estimated": snap.is_estimated,
                "sensor_detail": snap.sensor_detail,
                "sample_count": len(self._samples),
            }

    def _build_sample_locked(self, snap: HardwareSnapshot, now_perf: float) -> dict[str, Any]:
        self._seq += 1
        elapsed_s = round(max(0.0, now_perf - self._session_t0), 3)
        phase = self._current_phase
        return {
            "type": "power_sample",
            "session_id": self._active_session_id,
            "seq": self._seq,
            "timestamp_utc": utc_iso_now(),
            "elapsed_s": elapsed_s,
            "phase": phase,
            "watts": snap.watts,
            "power_snn_w": snap.watts if phase == "snn_inference" else None,
            "power_yolo_w": snap.watts if phase == "yolo_inference" else None,
            "cpu_pct": snap.cpu_pct,
            "gpu_util_pct": snap.gpu_util_pct,
            "mem_pct": snap.mem_pct,
            "source": snap.source,
            "is_estimated": snap.is_estimated,
            "sensor_detail": snap.sensor_detail,
        }

    def _emit_events(self, events: list[dict[str, Any]]) -> None:
        cb = self._on_event
        if cb is None:
            return
        for evt in events:
            try:
                cb(evt)
            except Exception:
                pass

    def start_session(
        self,
        session_id: str,
        image_name: str,
        initial_phase: str = "idle_baseline",
    ) -> dict[str, Any]:
        """Begin a benchmark power capture session and emit session_start + phase_change + first sample."""
        snap = self._safe_read_snapshot()
        now_perf = time.perf_counter()
        started_utc = utc_iso_now()

        with self._lock:
            self._active_session_id = session_id
            self._active_image_name = image_name
            self._active_started_utc = started_utc
            self._session_t0 = now_perf
            self._current_phase = initial_phase
            self._seq = 0
            self._samples = []
            self._phase_bounds = {initial_phase: [0.0, 0.0]}

            start_evt = {
                "type": "session_start",
                "session_id": session_id,
                "image_name": image_name,
                "started_at_utc": started_utc,
                "source": snap.source,
                "is_estimated": snap.is_estimated,
                "sensor_detail": snap.sensor_detail,
            }
            phase_evt = {
                "type": "phase_change",
                "session_id": session_id,
                "phase": initial_phase,
                "elapsed_s": 0.0,
            }
            first_sample = self._build_sample_locked(snap, now_perf)
            self._samples.append(first_sample)
            self._next_sample_at = now_perf + self.interval_s
            self._cond.notify_all()

        self._emit_events([start_evt, phase_evt, first_sample])
        return first_sample

    def set_phase(self, phase: str) -> dict[str, Any] | None:
        """Transition active session phase, emit phase_change, and capture boundary sample."""
        snap = self._safe_read_snapshot()
        now_perf = time.perf_counter()

        with self._lock:
            if self._active_session_id is None:
                return None
            elapsed_s = round(max(0.0, now_perf - self._session_t0), 3)
            if self._current_phase in self._phase_bounds:
                self._phase_bounds[self._current_phase][1] = elapsed_s
            self._current_phase = phase
            self._phase_bounds[phase] = [elapsed_s, elapsed_s]

            phase_evt = {
                "type": "phase_change",
                "session_id": self._active_session_id,
                "phase": phase,
                "elapsed_s": elapsed_s,
            }
            boundary_sample = self._build_sample_locked(snap, now_perf)
            self._samples.append(boundary_sample)
            self._next_sample_at = now_perf + self.interval_s
            self._cond.notify_all()

        self._emit_events([phase_evt, boundary_sample])
        return boundary_sample

    def stop_session(self, phase_run_stats: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        """Finish active benchmark session, integrate per-phase energy, and return summary + samples."""
        snap = self._safe_read_snapshot()
        now_perf = time.perf_counter()
        completed_utc = utc_iso_now()

        with self._lock:
            sid = self._active_session_id
            if sid is None:
                raise RuntimeError("No active power monitor session to stop")

            elapsed_end = round(max(0.0, now_perf - self._session_t0), 3)
            if self._current_phase in self._phase_bounds:
                self._phase_bounds[self._current_phase][1] = elapsed_end

            # Capture closing sample if at least 40ms elapsed since previous sample
            last_elapsed = self._samples[-1]["elapsed_s"] if self._samples else -1.0
            final_sample = None
            if (elapsed_end - last_elapsed) >= 0.04:
                final_sample = self._build_sample_locked(snap, now_perf)
                self._samples.append(final_sample)

            samples = list(self._samples)
            image_name = self._active_image_name
            started_utc = self._active_started_utc
            phase_bounds = {k: list(v) for k, v in self._phase_bounds.items()}

            self._active_session_id = None
            self._current_phase = "idle"
            self._samples = []
            self._phase_bounds = {}
            self._cond.notify_all()

        if final_sample is not None:
            self._emit_events([final_sample])

        summary = self._build_summary(
            session_id=sid,
            image_name=image_name,
            started_utc=started_utc,
            completed_utc=completed_utc,
            duration_s=elapsed_end,
            samples=samples,
            phase_bounds=phase_bounds,
            phase_run_stats=phase_run_stats or {},
        )
        return {"summary": summary, "samples": samples}

    def abort_session(self) -> None:
        with self._lock:
            self._active_session_id = None
            self._current_phase = "idle"
            self._samples = []
            self._phase_bounds = {}
            self._cond.notify_all()

    @staticmethod
    def _build_summary(
        session_id: str,
        image_name: str,
        started_utc: str,
        completed_utc: str,
        duration_s: float,
        samples: list[dict[str, Any]],
        phase_bounds: dict[str, list[float]],
        phase_run_stats: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        last = samples[-1] if samples else {}
        source = str(last.get("source", "psutil-estimated"))
        is_estimated = bool(last.get("is_estimated", True))
        sensor_detail = str(last.get("sensor_detail", ""))

        phases: dict[str, dict[str, Any]] = {}
        for phase_name in PHASES_ORDER:
            idxs = [i for i, s in enumerate(samples) if s["phase"] == phase_name]
            if not idxs:
                entry: dict[str, Any] = {
                    "avg_watts": 0.0,
                    "peak_watts": 0.0,
                    "min_watts": 0.0,
                    "duration_s": 0.0,
                    "total_energy_j": 0.0,
                    "total_energy_mj": 0.0,
                    "sample_count": 0,
                }
                if phase_name in ("snn_inference", "yolo_inference"):
                    entry.update({
                        "net_energy_mj": 0.0,
                        "runs": 0,
                        "mean_latency_ms": 0.0,
                        "energy_per_inference_mj": 0.0,
                        "net_energy_per_inference_mj": 0.0,
                    })
                phases[phase_name] = entry
                continue

            pts = [(float(samples[i]["elapsed_s"]), float(samples[i]["watts"])) for i in idxs]
            last_idx = idxs[-1]
            # Bridge integration seamlessly to the immediate boundary of the next phase
            if last_idx + 1 < len(samples):
                nxt = samples[last_idx + 1]
                pts.append((float(nxt["elapsed_s"]), float(nxt["watts"])))

            bounds = phase_bounds.get(phase_name)
            bound_dur = max(0.0, float(bounds[1]) - float(bounds[0])) if bounds else 0.0
            pts_dur = max(0.0, pts[-1][0] - pts[0][0])
            phase_dur = round(max(bound_dur, pts_dur), 3)

            total_j = integrate_energy_j(pts)
            total_mj = total_j * 1000.0
            watts_vals = [float(samples[i]["watts"]) for i in idxs]
            avg_w = (total_j / pts_dur) if pts_dur > 1e-6 else (sum(watts_vals) / len(watts_vals))

            entry = {
                "avg_watts": round(avg_w, 3),
                "peak_watts": round(max(watts_vals), 3),
                "min_watts": round(min(watts_vals), 3),
                "duration_s": phase_dur,
                "total_energy_j": round(total_j, 4),
                "total_energy_mj": round(total_mj, 2),
                "sample_count": len(idxs),
            }
            phases[phase_name] = entry

        idle_avg = float(phases["idle_baseline"]["avg_watts"])
        for inf_phase in ("snn_inference", "yolo_inference"):
            p = phases[inf_phase]
            stats = phase_run_stats.get(inf_phase, {})
            runs = int(stats.get("runs", 0))
            mean_lat = float(stats.get("mean_latency_ms", 0.0))
            dur = float(p["duration_s"])
            tot_mj = float(p["total_energy_mj"])
            net_mj = max(0.0, tot_mj - (idle_avg * dur * 1000.0))
            p["net_energy_mj"] = round(net_mj, 2)
            p["runs"] = runs
            p["mean_latency_ms"] = round(mean_lat, 2)
            p["energy_per_inference_mj"] = round(tot_mj / runs, 3) if runs > 0 else 0.0
            p["net_energy_per_inference_mj"] = round(net_mj / runs, 3) if runs > 0 else 0.0

        snn_p = phases["snn_inference"]
        yolo_p = phases["yolo_inference"]
        snn_w = float(snn_p["avg_watts"])
        yolo_w = float(yolo_p["avg_watts"])
        snn_epi = float(snn_p["energy_per_inference_mj"])
        yolo_epi = float(yolo_p["energy_per_inference_mj"])
        snn_lat = float(snn_p["mean_latency_ms"])
        yolo_lat = float(yolo_p["mean_latency_ms"])

        power_ratio = round(yolo_w / snn_w, 2) if snn_w > 1e-6 and yolo_w > 0 else 0.0
        power_savings_pct = round((1.0 - (snn_w / yolo_w)) * 100.0, 1) if yolo_w > 1e-6 and snn_w > 0 else 0.0
        energy_ratio = round(yolo_epi / snn_epi, 2) if snn_epi > 1e-6 and yolo_epi > 0 else 0.0
        energy_savings_pct = round((1.0 - (snn_epi / yolo_epi)) * 100.0, 1) if yolo_epi > 1e-6 and snn_epi > 0 else 0.0
        lat_ratio = round(yolo_lat / snn_lat, 2) if snn_lat > 1e-6 and yolo_lat > 0 else 0.0

        return {
            "session_id": session_id,
            "image_name": image_name,
            "started_at_utc": started_utc,
            "completed_at_utc": completed_utc,
            "duration_s": round(duration_s, 3),
            "source": source,
            "is_estimated": is_estimated,
            "sensor_detail": sensor_detail,
            "sample_count": len(samples),
            "phases": phases,
            "comparison": {
                "power_ratio_yolo_over_snn": power_ratio,
                "power_savings_pct": power_savings_pct,
                "energy_ratio_yolo_over_snn": energy_ratio,
                "energy_savings_pct": energy_savings_pct,
                "yolo_over_snn_latency_ratio": lat_ratio,
            },
        }

    def _sampler_loop(self) -> None:
        while True:
            with self._lock:
                while not self._shutdown and self._active_session_id is None:
                    self._cond.wait(timeout=0.5)
                if self._shutdown:
                    return
                wait_s = self._next_sample_at - time.perf_counter()
                if wait_s > 0.005:
                    self._cond.wait(timeout=wait_s)
                    continue
                if self._shutdown or self._active_session_id is None:
                    continue

            # Read hardware telemetry outside self._lock
            snap = self._safe_read_snapshot()
            now_perf = time.perf_counter()

            sample: dict[str, Any] | None = None
            with self._lock:
                if self._shutdown or self._active_session_id is None:
                    continue
                # Guard against firing right after set_phase() already took a boundary sample
                if now_perf < self._next_sample_at - 0.01:
                    continue
                sample = self._build_sample_locked(snap, now_perf)
                self._samples.append(sample)
                self._next_sample_at = now_perf + self.interval_s

            if sample is not None:
                self._emit_events([sample])

    def close(self) -> None:
        with self._lock:
            self._shutdown = True
            self._active_session_id = None
            self._cond.notify_all()
        try:
            self._reader.close()
        except Exception:
            pass
