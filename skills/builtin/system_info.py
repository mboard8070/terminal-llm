"""System information skill."""

import os
import platform
import subprocess
from skills import skill


@skill(
    name="system_info",
    description="Get system information (CPU, memory, disk, GPU, network)",
    version="1.0.0",
    author="MAUDE",
    triggers=["system", "sysinfo", "hardware", "cpu", "memory", "disk", "gpu"],
    parameters={
        "type": "object",
        "properties": {
            "component": {
                "type": "string",
                "enum": ["all", "cpu", "memory", "disk", "gpu", "network", "os"],
                "description": "Which component to query (default: all)",
                "default": "all"
            }
        }
    }
)
def system_info(component: str = "all") -> str:
    """Get system information."""
    sections = []

    if component in ("all", "os"):
        sections.append(_get_os_info())

    if component in ("all", "cpu"):
        sections.append(_get_cpu_info())

    if component in ("all", "memory"):
        sections.append(_get_memory_info())

    if component in ("all", "disk"):
        sections.append(_get_disk_info())

    if component in ("all", "gpu"):
        sections.append(_get_gpu_info())

    if component in ("all", "network"):
        sections.append(_get_network_info())

    return "\n\n".join(filter(None, sections))


def _get_os_info() -> str:
    """Get OS information."""
    lines = ["[OS Information]"]
    lines.append(f"  System: {platform.system()}")
    lines.append(f"  Release: {platform.release()}")
    lines.append(f"  Version: {platform.version()}")
    lines.append(f"  Machine: {platform.machine()}")
    lines.append(f"  Hostname: {platform.node()}")

    # Uptime
    try:
        with open("/proc/uptime", "r") as f:
            uptime_seconds = float(f.read().split()[0])
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            lines.append(f"  Uptime: {days}d {hours}h {minutes}m")
    except:
        pass

    return "\n".join(lines)


def _get_cpu_info() -> str:
    """Get CPU information."""
    lines = ["[CPU Information]"]

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        # Get model name
        for line in cpuinfo.split("\n"):
            if "model name" in line:
                lines.append(f"  Model: {line.split(':')[1].strip()}")
                break

        # Count cores
        cores = cpuinfo.count("processor")
        lines.append(f"  Cores: {cores}")

        # Get load average
        with open("/proc/loadavg", "r") as f:
            load = f.read().split()[:3]
            lines.append(f"  Load: {', '.join(load)} (1m, 5m, 15m)")

    except Exception as e:
        lines.append(f"  Error reading CPU info: {e}")

    return "\n".join(lines)


def _get_memory_info() -> str:
    """Get memory information."""
    lines = ["[Memory Information]"]

    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().replace(" kB", "")
                    try:
                        meminfo[key] = int(value)
                    except:
                        pass

        total = meminfo.get("MemTotal", 0) / 1024 / 1024
        free = meminfo.get("MemAvailable", 0) / 1024 / 1024
        used = total - free
        usage_pct = (used / total * 100) if total > 0 else 0

        lines.append(f"  Total: {total:.1f} GB")
        lines.append(f"  Used: {used:.1f} GB ({usage_pct:.1f}%)")
        lines.append(f"  Available: {free:.1f} GB")

        # Swap
        swap_total = meminfo.get("SwapTotal", 0) / 1024 / 1024
        swap_free = meminfo.get("SwapFree", 0) / 1024 / 1024
        if swap_total > 0:
            lines.append(f"  Swap: {swap_total - swap_free:.1f} / {swap_total:.1f} GB")

    except Exception as e:
        lines.append(f"  Error reading memory info: {e}")

    return "\n".join(lines)


def _get_disk_info() -> str:
    """Get disk information."""
    lines = ["[Disk Information]"]

    try:
        result = subprocess.run(
            ["df", "-h", "--output=target,size,used,avail,pcent", "-x", "tmpfs", "-x", "devtmpfs"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 5:
                    mount, size, used, avail, pct = parts[:5]
                    if mount.startswith("/"):  # Only real mounts
                        lines.append(f"  {mount}: {used}/{size} ({pct})")
    except Exception as e:
        lines.append(f"  Error reading disk info: {e}")

    return "\n".join(lines)


def _get_gpu_info() -> str:
    """Get GPU information."""
    lines = ["[GPU Information]"]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    name, mem_total, mem_used, mem_free, temp, util = parts[:6]
                    lines.append(f"  GPU {i}: {name}")
                    lines.append(f"    Memory: {mem_used}/{mem_total} MB")
                    lines.append(f"    Temp: {temp}°C, Utilization: {util}%")
        else:
            lines.append("  No NVIDIA GPU detected")
    except FileNotFoundError:
        lines.append("  nvidia-smi not found")
    except Exception as e:
        lines.append(f"  Error reading GPU info: {e}")

    return "\n".join(lines)


def _get_network_info() -> str:
    """Get network information."""
    lines = ["[Network Information]"]

    try:
        # Get IP addresses
        result = subprocess.run(
            ["ip", "-4", "addr", "show"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "inet " in line and "127.0.0.1" not in line:
                    parts = line.strip().split()
                    ip = parts[1].split("/")[0]
                    iface = parts[-1] if len(parts) > 1 else "unknown"
                    lines.append(f"  {iface}: {ip}")

        # Check Tailscale
        try:
            ts_result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if ts_result.returncode == 0:
                import json
                ts_data = json.loads(ts_result.stdout)
                if ts_data.get("Self", {}).get("TailscaleIPs"):
                    ts_ip = ts_data["Self"]["TailscaleIPs"][0]
                    lines.append(f"  tailscale0: {ts_ip}")
        except:
            pass

    except Exception as e:
        lines.append(f"  Error reading network info: {e}")

    return "\n".join(lines)
