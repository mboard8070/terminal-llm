"""System information skill - cross-platform (Linux/macOS)."""

import os
import platform
import subprocess
import re
from skills import skill


def _run_cmd(cmd: list, timeout: int = 5) -> str:
    """Run a command and return output, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else ""
    except:
        return ""


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


@skill(
    name="system_info",
    description="Get system information (CPU, memory, disk, GPU, network)",
    version="1.1.0",
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
    if _is_linux():
        try:
            with open("/proc/uptime", "r") as f:
                uptime_seconds = float(f.read().split()[0])
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                lines.append(f"  Uptime: {days}d {hours}h {minutes}m")
        except:
            pass
    elif _is_macos():
        # macOS uptime via sysctl
        output = _run_cmd(["sysctl", "-n", "kern.boottime"])
        if output:
            # Format: { sec = 1234567890, usec = 0 }
            match = re.search(r'sec = (\d+)', output)
            if match:
                import time
                boot_time = int(match.group(1))
                uptime_seconds = time.time() - boot_time
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                lines.append(f"  Uptime: {days}d {hours}h {minutes}m")

    return "\n".join(lines)


def _get_cpu_info() -> str:
    """Get CPU information."""
    lines = ["[CPU Information]"]

    if _is_linux():
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    lines.append(f"  Model: {line.split(':')[1].strip()}")
                    break

            cores = cpuinfo.count("processor")
            lines.append(f"  Cores: {cores}")

            with open("/proc/loadavg", "r") as f:
                load = f.read().split()[:3]
                lines.append(f"  Load: {', '.join(load)} (1m, 5m, 15m)")

        except Exception as e:
            lines.append(f"  Error: {e}")

    elif _is_macos():
        # CPU model
        model = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            lines.append(f"  Model: {model}")

        # Core count
        cores = _run_cmd(["sysctl", "-n", "hw.ncpu"])
        if cores:
            lines.append(f"  Cores: {cores}")

        # Performance cores (Apple Silicon)
        perf_cores = _run_cmd(["sysctl", "-n", "hw.perflevel0.logicalcpu"])
        eff_cores = _run_cmd(["sysctl", "-n", "hw.perflevel1.logicalcpu"])
        if perf_cores and eff_cores:
            lines.append(f"  Performance cores: {perf_cores}, Efficiency: {eff_cores}")

        # Load average
        load = _run_cmd(["sysctl", "-n", "vm.loadavg"])
        if load:
            # Format: { 1.23 1.45 1.67 }
            match = re.search(r'\{\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', load)
            if match:
                lines.append(f"  Load: {match.group(1)}, {match.group(2)}, {match.group(3)} (1m, 5m, 15m)")

    return "\n".join(lines)


def _get_memory_info() -> str:
    """Get memory information."""
    lines = ["[Memory Information]"]

    if _is_linux():
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

            swap_total = meminfo.get("SwapTotal", 0) / 1024 / 1024
            swap_free = meminfo.get("SwapFree", 0) / 1024 / 1024
            if swap_total > 0:
                lines.append(f"  Swap: {swap_total - swap_free:.1f} / {swap_total:.1f} GB")

        except Exception as e:
            lines.append(f"  Error: {e}")

    elif _is_macos():
        # Total memory
        mem_bytes = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem_bytes:
            total_gb = int(mem_bytes) / (1024 ** 3)
            lines.append(f"  Total: {total_gb:.1f} GB")

        # Memory usage via vm_stat
        vm_stat = _run_cmd(["vm_stat"])
        if vm_stat:
            page_size = 16384  # Default for Apple Silicon, 4096 for Intel
            ps_output = _run_cmd(["sysctl", "-n", "hw.pagesize"])
            if ps_output:
                page_size = int(ps_output)

            stats = {}
            for line in vm_stat.split("\n"):
                if ":" in line:
                    key, val = line.split(":")
                    val = val.strip().rstrip(".")
                    try:
                        stats[key.strip()] = int(val)
                    except:
                        pass

            # Calculate used memory
            wired = stats.get("Pages wired down", 0) * page_size
            active = stats.get("Pages active", 0) * page_size
            compressed = stats.get("Pages occupied by compressor", 0) * page_size
            used_gb = (wired + active + compressed) / (1024 ** 3)

            if mem_bytes:
                total_gb = int(mem_bytes) / (1024 ** 3)
                free_gb = total_gb - used_gb
                usage_pct = (used_gb / total_gb * 100)
                lines.append(f"  Used: {used_gb:.1f} GB ({usage_pct:.1f}%)")
                lines.append(f"  Available: {free_gb:.1f} GB")

    return "\n".join(lines)


def _get_disk_info() -> str:
    """Get disk information."""
    lines = ["[Disk Information]"]

    if _is_linux():
        try:
            result = subprocess.run(
                ["df", "-h", "--output=target,size,used,avail,pcent", "-x", "tmpfs", "-x", "devtmpfs"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 5:
                        mount, size, used, avail, pct = parts[:5]
                        if mount.startswith("/"):
                            lines.append(f"  {mount}: {used}/{size} ({pct})")
        except Exception as e:
            lines.append(f"  Error: {e}")

    elif _is_macos():
        # macOS df doesn't support --output, use different format
        result = _run_cmd(["df", "-h"])
        if result:
            for line in result.split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 5 and parts[0].startswith("/dev/"):
                    # Filesystem Size Used Avail Capacity Mounted
                    size, used, avail, pct = parts[1:5]
                    mount = parts[-1] if len(parts) > 5 else parts[5]
                    if mount in ("/", "/System/Volumes/Data") or mount.startswith("/Volumes"):
                        lines.append(f"  {mount}: {used}/{size} ({pct})")

    return "\n".join(lines)


def _get_gpu_info() -> str:
    """Get GPU information."""
    lines = ["[GPU Information]"]

    if _is_linux():
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
                        lines.append(f"    Temp: {temp}C, Utilization: {util}%")
            else:
                lines.append("  No NVIDIA GPU detected")
        except FileNotFoundError:
            lines.append("  nvidia-smi not found")
        except Exception as e:
            lines.append(f"  Error: {e}")

    elif _is_macos():
        # macOS - check for Apple Silicon GPU or discrete GPU
        result = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        if result:
            # Parse the output
            current_gpu = None
            for line in result.split("\n"):
                line = line.strip()
                if line.endswith(":") and not line.startswith("Displays:"):
                    # GPU name line
                    if "Chipset Model:" in line:
                        current_gpu = line.replace("Chipset Model:", "").strip()
                    elif not any(x in line.lower() for x in ["display", "resolution", "vendor"]):
                        current_gpu = line.rstrip(":")
                        if current_gpu and current_gpu not in ["Graphics/Displays", "Displays"]:
                            lines.append(f"  {current_gpu}")
                elif "Chipset Model:" in line:
                    gpu_name = line.split(":")[1].strip()
                    lines.append(f"  {gpu_name}")
                elif "Total Number of Cores:" in line:
                    cores = line.split(":")[1].strip()
                    lines.append(f"    GPU Cores: {cores}")
                elif "VRAM" in line or "Memory" in line:
                    if ":" in line:
                        mem = line.split(":")[1].strip()
                        lines.append(f"    Memory: {mem}")
                elif "Metal Support:" in line or "Metal Family:" in line:
                    metal = line.split(":")[1].strip()
                    lines.append(f"    Metal: {metal}")

        if len(lines) == 1:
            lines.append("  No GPU info available")

    return "\n".join(lines)


def _get_network_info() -> str:
    """Get network information."""
    lines = ["[Network Information]"]

    if _is_linux():
        try:
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
        except:
            pass

    elif _is_macos():
        # Use ifconfig on macOS
        result = _run_cmd(["ifconfig"])
        if result:
            current_iface = None
            for line in result.split("\n"):
                if not line.startswith("\t") and ":" in line:
                    current_iface = line.split(":")[0]
                elif "inet " in line and "127.0.0.1" not in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        if current_iface:
                            lines.append(f"  {current_iface}: {ip}")

    # Check Tailscale (works on both platforms)
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

    return "\n".join(lines)
