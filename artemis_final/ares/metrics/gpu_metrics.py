"""
System Metrics Server for vLLM Host

Run this on the SAME machine as your vLLM server. It exposes a FastAPI app
that returns detailed system + GPU metrics as JSON.

Usage (from shell on vLLM host):

    pip install fastapi uvicorn psutil nvidia-ml-py3

    python -m uvicorn gpu_metrics:app --host 0.0.0.0 --port 9000

Then from your client machine, you can call:

    GET http://VLLM_SERVER_IP:9000/metrics

and feed the JSON into your router / data-collection pipeline.
"""

import os
import socket
import time
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Optional dependencies
try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except Exception:
    psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False

try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False


app = FastAPI(
    title="System Metrics API",
    description="Expose detailed system and GPU metrics for vLLM router logging.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def get_system_info() -> Dict[str, Any]:
    """Basic system/OS info."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def get_uptime_info() -> Dict[str, Any]:
    """System uptime and boot time."""
    boot_ts = None
    uptime_sec = None

    if _PSUTIL_AVAILABLE:
        try:
            boot_ts = psutil.boot_time()
            uptime_sec = time.time() - boot_ts
        except Exception:
            pass

    return {
        "boot_time_utc": (
            datetime.fromtimestamp(boot_ts, tz=timezone.utc).isoformat()
            if boot_ts is not None
            else None
        ),
        "uptime_seconds": uptime_sec,
    }


def get_cpu_info() -> Dict[str, Any]:
    """CPU counts and utilization."""
    info: Dict[str, Any] = {}

    if not _PSUTIL_AVAILABLE:
        return info

    try:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)

        # Instantaneous percentage usage
        # (interval=0.0 -> non-blocking, last known values)
        info["cpu_usage_percent"] = psutil.cpu_percent(interval=0.0)
        info["per_core_usage_percent"] = psutil.cpu_percent(interval=0.0, percpu=True)

        # Load averages (Unix only)
        try:
            load1, load5, load15 = os.getloadavg()
            info["load_avg_1m"] = load1
            info["load_avg_5m"] = load5
            info["load_avg_15m"] = load15
        except (AttributeError, OSError):
            # Not available on some platforms (e.g. Windows)
            pass
    except Exception:
        pass

    return info


def get_memory_info() -> Dict[str, Any]:
    """Main memory and swap usage."""
    mem_info: Dict[str, Any] = {}
    swap_info: Dict[str, Any] = {}

    if _PSUTIL_AVAILABLE:
        try:
            vmem = psutil.virtual_memory()
            mem_info = {
                "total_bytes": vmem.total,
                "available_bytes": vmem.available,
                "used_bytes": vmem.used,
                "free_bytes": getattr(vmem, "free", None),
                "percent": vmem.percent,
                "active_bytes": getattr(vmem, "active", None),
                "inactive_bytes": getattr(vmem, "inactive", None),
                "buffers_bytes": getattr(vmem, "buffers", None),
                "cached_bytes": getattr(vmem, "cached", None),
                "shared_bytes": getattr(vmem, "shared", None),
                "slab_bytes": getattr(vmem, "slab", None),
            }

            smem = psutil.swap_memory()
            swap_info = {
                "total_bytes": smem.total,
                "used_bytes": smem.used,
                "free_bytes": smem.free,
                "percent": smem.percent,
                "sin_bytes": smem.sin,
                "sout_bytes": smem.sout,
            }
        except Exception:
            pass

    return {"virtual_memory": mem_info, "swap_memory": swap_info}


def get_disk_info() -> Dict[str, Any]:
    """Disk partitions, usage, and I/O stats."""
    disks: Dict[str, Any] = {"partitions": [], "io_counters": {}}

    if not _PSUTIL_AVAILABLE:
        return disks

    try:
        partitions = psutil.disk_partitions(all=False)
        for p in partitions:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                disks["partitions"].append(
                    {
                        "device": p.device,
                        "mountpoint": p.mountpoint,
                        "fstype": p.fstype,
                        "opts": p.opts,
                        "total_bytes": usage.total,
                        "used_bytes": usage.used,
                        "free_bytes": usage.free,
                        "percent": usage.percent,
                    }
                )
            except Exception:
                # Partition might not be accessible
                continue

        # Aggregate I/O stats
        try:
            io_counters = psutil.disk_io_counters(perdisk=False)
            if io_counters is not None:
                disks["io_counters"] = {
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count,
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes,
                    "read_time_ms": io_counters.read_time,
                    "write_time_ms": io_counters.write_time,
                    "busy_time_ms": getattr(io_counters, "busy_time", None),
                }
        except Exception:
            pass

    except Exception:
        pass

    return disks


def get_network_info() -> Dict[str, Any]:
    """Network I/O stats."""
    info: Dict[str, Any] = {"io_counters": {}, "per_nic": {}}

    if not _PSUTIL_AVAILABLE:
        return info

    try:
        # Aggregate I/O
        io = psutil.net_io_counters(pernic=False)
        if io is not None:
            info["io_counters"] = {
                "bytes_sent": io.bytes_sent,
                "bytes_recv": io.bytes_recv,
                "packets_sent": io.packets_sent,
                "packets_recv": io.packets_recv,
                "errin": io.errin,
                "errout": io.errout,
                "dropin": io.dropin,
                "dropout": io.dropout,
            }

        # Per-interface I/O
        pernic = psutil.net_io_counters(pernic=True)
        per_nic_info: Dict[str, Any] = {}
        for nic, stats in pernic.items():
            per_nic_info[nic] = {
                "bytes_sent": stats.bytes_sent,
                "bytes_recv": stats.bytes_recv,
                "packets_sent": stats.packets_sent,
                "packets_recv": stats.packets_recv,
                "errin": stats.errin,
                "errout": stats.errout,
                "dropin": stats.dropin,
                "dropout": stats.dropout,
            }
        info["per_nic"] = per_nic_info

    except Exception:
        pass

    return info


def get_process_info() -> Dict[str, Any]:
    """Info about THIS process (the FastAPI server)."""
    info: Dict[str, Any] = {}

    if not _PSUTIL_AVAILABLE:
        return info

    try:
        proc = psutil.Process(os.getpid())
        with proc.oneshot():
            mem = proc.memory_info()
            cpu_times = proc.cpu_times()

            info = {
                "pid": proc.pid,
                "name": proc.name(),
                "exe": proc.exe(),
                "cmdline": proc.cmdline(),
                "create_time_utc": datetime.fromtimestamp(
                    proc.create_time(), tz=timezone.utc
                ).isoformat(),
                "status": proc.status(),
                "num_threads": proc.num_threads(),
                "cpu_percent": proc.cpu_percent(interval=0.0),
                "cpu_times": {
                    "user": cpu_times.user,
                    "system": cpu_times.system,
                    "children_user": getattr(cpu_times, "children_user", None),
                    "children_system": getattr(cpu_times, "children_system", None),
                    "iowait": getattr(cpu_times, "iowait", None),
                },
                "memory": {
                    "rss_bytes": mem.rss,
                    "vms_bytes": mem.vms,
                    "shared_bytes": getattr(mem, "shared", None),
                    "text_bytes": getattr(mem, "text", None),
                    "lib_bytes": getattr(mem, "lib", None),
                    "data_bytes": getattr(mem, "data", None),
                    "dirty_bytes": getattr(mem, "dirty", None),
                },
                "open_files": [
                    {"path": f.path, "fd": f.fd} for f in proc.open_files()
                ],
                "num_fds": getattr(proc, "num_fds", lambda: None)(),
            }
    except Exception:
        pass

    return info


def get_gpu_info() -> Dict[str, Any]:
    """Detailed GPU info using NVML (pynvml)."""
    info: Dict[str, Any] = {
        "nvml_available": _NVML_AVAILABLE,
        "gpus": [],
    }

    if not _NVML_AVAILABLE:
        return info

    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return info

    for idx in range(count):
        gpu_data: Dict[str, Any] = {"index": idx}
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            name = pynvml.nvmlDeviceGetName(handle)
            try:
                name = name.decode("utf-8")
            except Exception:
                name = str(name)

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            temp = None
            power_draw = None
            power_limit = None
            fan_speed = None

            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                pass

            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                pass

            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                pass

            gpu_data.update(
                {
                    "name": name,
                    "uuid": pynvml.nvmlDeviceGetUUID(handle).decode("utf-8"),
                    "memory": {
                        "total_bytes": mem.total,
                        "used_bytes": mem.used,
                        "free_bytes": mem.free,
                        "used_mb": mem.used / (1024 ** 2),
                        "total_mb": mem.total / (1024 ** 2),
                        "free_mb": mem.free / (1024 ** 2),
                    },
                    "utilization": {
                        "gpu_percent": util.gpu,
                        "memory_percent": util.memory,
                    },
                    "temperature_celsius": temp,
                    "power_watts": power_draw,
                    "power_limit_watts": power_limit,
                    "fan_speed_percent": fan_speed,
                }
            )

            # Optional: running processes on this GPU
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                proc_list: List[Dict[str, Any]] = []
                for p in procs:
                    proc_list.append(
                        {
                            "pid": p.pid,
                            "used_gpu_memory_bytes": p.usedGpuMemory,
                        }
                    )
                gpu_data["compute_processes"] = proc_list
            except Exception:
                pass

        except Exception:
            # If any failure for this GPU, still push partial data
            pass

        info["gpus"].append(gpu_data)

    return info


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """
    Simple health endpoint for liveness checks.
    """
    return {"status": "ok"}


@app.get("/metrics", summary="Full system metrics")
async def metrics() -> JSONResponse:
    """
    Return a full snapshot of system, CPU, memory, disk, network,
    process, and GPU metrics.

    This endpoint is designed to be called from your router client to
    attach rich system metrics to each vLLM request.
    """
    now = datetime.now(tz=timezone.utc).isoformat()

    payload: Dict[str, Any] = {
        "timestamp_utc": now,
        "system": get_system_info(),
        "uptime": get_uptime_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disks": get_disk_info(),
        "network": get_network_info(),
        "process": get_process_info(),
        "gpu": get_gpu_info(),
    }

    return JSONResponse(content=payload)


# Optional: a minimal/flat numeric view if you want a super simple client
@app.get("/metrics/flat", summary="Flattened numeric metrics (optional)")
async def metrics_flat() -> JSONResponse:
    """
    Flattened numeric snapshot, useful if your client just wants a
    simple key->float mapping.

    Keys are namespaced like:
      - cpu.cpu_usage_percent
      - memory.virtual_memory.total_bytes
      - gpu.gpus.0.memory.used_mb
    """
    full = await metrics()
    data = full.body  # bytes
    # FastAPI's JSONResponse.body is bytes; decode+load
    import json

    parsed = json.loads(data)

    flat: Dict[str, float] = {}

    def _walk(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                _walk(f"{prefix}.{idx}", v)
        else:
            val = _safe_float(obj)
            if val is not None:
                flat[prefix] = val

    _walk("", parsed)

    return JSONResponse(content=flat)
