from maude_bootstrap import ensure_local_maude

ensure_local_maude()

"""
MAUDE Mesh Network - Discovery and routing for remote subagents.

Supports:
1. Tailscale auto-discovery (discovers peers via tailscale status)
2. Manual node configuration (MAUDE_NODES env var)
3. Health checks and capability detection
4. Multi-modal capability mesh (LLM, vision, video, 3D, etc.)
"""

import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from rich.console import Console

from maude.config import runtime_paths
from capabilities import (
    Capability,
    CapabilityType,
    EndpointType,
    capability_for_comfyui,
    capability_from_ollama_model,
)

console = Console()


@dataclass
class MaudeNode:
    """Represents a MAUDE instance on the network."""

    hostname: str
    ip: str
    tailscale_ip: str | None = None
    api_port: int = 30080
    ollama_port: int = 11434
    comfyui_port: int = 8188
    capabilities: list[str] = field(default_factory=list)  # Legacy: model names
    models: dict[str, str] = field(default_factory=dict)  # model_name -> model_id
    node_capabilities: list[Capability] = field(default_factory=list)  # New: typed capabilities
    last_seen: float = 0.0
    last_check: float = 0.0
    healthy: bool = False
    load: float = 0.0  # 0.0-1.0 current utilization

    def get_ollama_url(self) -> str:
        """Get the Ollama API URL for this node."""
        host = self.tailscale_ip or self.ip or self.hostname
        return f"http://{host}:{self.ollama_port}/v1"

    def get_comfyui_url(self) -> str:
        """Get the ComfyUI API URL for this node."""
        host = self.tailscale_ip or self.ip or self.hostname
        return f"http://{host}:{self.comfyui_port}"


@dataclass
class MeshConfig:
    """Configuration for the MAUDE mesh."""

    enabled: bool = True
    health_check_interval: int = 60  # seconds
    node_timeout: int = 120  # seconds before marking node as dead
    discovery_port: int = 31337
    auto_discover_tailscale: bool = True
    manual_nodes: list[str] = field(default_factory=list)  # ["host:port", ...]


class MaudeMesh:
    """Mesh network of MAUDE instances for remote subagent routing."""

    CONFIG_FILE = runtime_paths().config_dir / "mesh.json"

    def __init__(self, my_capabilities: list[str] = None):
        self.my_hostname = socket.gethostname()
        self.my_capabilities = my_capabilities or []
        self.nodes: dict[str, MaudeNode] = {}
        self.config = self._load_config()
        self.running = False
        self._lock = threading.Lock()

    def _load_config(self) -> MeshConfig:
        """Load mesh configuration."""
        config = MeshConfig()

        # Environment overrides
        if os.environ.get("MAUDE_MESH_ENABLED", "").lower() == "false":
            config.enabled = False

        if os.environ.get("MAUDE_MESH_PORT"):
            try:
                config.discovery_port = int(os.environ["MAUDE_MESH_PORT"])
            except:
                pass

        # Manual nodes from env
        nodes_str = os.environ.get("MAUDE_NODES", "")
        if nodes_str:
            config.manual_nodes = [n.strip() for n in nodes_str.split(",") if n.strip()]

        # Load from config file
        if self.CONFIG_FILE.exists():
            try:
                data = json.loads(self.CONFIG_FILE.read_text())
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            except:
                pass

        return config

    def _save_config(self):
        """Save mesh configuration."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.CONFIG_FILE.write_text(json.dumps(asdict(self.config), indent=2))

    def start(self):
        """Start mesh discovery and health checking."""
        if not self.config.enabled:
            console.print("[dim]Mesh networking disabled[/dim]")
            return

        self.running = True
        console.print("[dim]Starting MAUDE mesh network...[/dim]")

        # Initial discovery
        self._discover_nodes()

        # Start background threads
        threading.Thread(target=self._health_check_loop, daemon=True).start()

    def stop(self):
        """Stop mesh services."""
        self.running = False

    def _discover_nodes(self):
        """Discover nodes from all sources."""
        # Discover from Tailscale
        if self.config.auto_discover_tailscale:
            self._discover_tailscale()

        # Add manual nodes
        for node_spec in self.config.manual_nodes:
            self._add_manual_node(node_spec)

    def _discover_tailscale(self):
        """Discover MAUDE instances on the Tailscale network."""
        try:
            result = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return

            data = json.loads(result.stdout)
            peers = data.get("Peer", {})

            for _peer_id, peer in peers.items():
                hostname = peer.get("HostName", "").lower()
                if not hostname or hostname == self.my_hostname.lower():
                    continue

                # Get Tailscale IPs
                ts_ips = peer.get("TailscaleIPs", [])
                ts_ip = ts_ips[0] if ts_ips else None

                # Check if node is online
                if not peer.get("Online", False):
                    continue

                with self._lock:
                    if hostname not in self.nodes:
                        self.nodes[hostname] = MaudeNode(
                            hostname=hostname,
                            ip=hostname,  # Use hostname as IP for DNS resolution
                            tailscale_ip=ts_ip,
                        )
                    else:
                        self.nodes[hostname].tailscale_ip = ts_ip

        except FileNotFoundError:
            pass  # Tailscale not installed
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            console.print(f"[yellow]Tailscale discovery error: {e}[/yellow]")

    def _add_manual_node(self, node_spec: str):
        """Add a manually configured node."""
        # Parse "hostname:port" or just "hostname"
        parts = node_spec.split(":")
        hostname = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 30080

        if hostname.lower() == self.my_hostname.lower():
            return

        with self._lock:
            if hostname not in self.nodes:
                self.nodes[hostname] = MaudeNode(hostname=hostname, ip=hostname, api_port=port)

    def _health_check_loop(self):
        """Periodically check health of all nodes."""
        while self.running:
            self._check_all_nodes()
            time.sleep(self.config.health_check_interval)

    def _check_all_nodes(self):
        """Check health and capabilities of all nodes."""
        with self._lock:
            nodes_to_check = list(self.nodes.values())

        for node in nodes_to_check:
            self._check_node(node)

    def _check_node(self, node: MaudeNode):
        """Check a single node's health and capabilities."""
        now = time.time()

        # Check Ollama
        ollama_models = self._check_ollama(node)
        if ollama_models:
            node.healthy = True
            node.last_seen = now
            node.models = ollama_models
            node.capabilities = list(ollama_models.keys())
        else:
            # Check if main API is available
            api_healthy = self._check_api(node)
            if api_healthy:
                node.healthy = True
                node.last_seen = now
            else:
                # Mark as unhealthy if too long since last seen
                if now - node.last_seen > self.config.node_timeout:
                    node.healthy = False

        # Build typed capabilities (includes ComfyUI check)
        self._build_node_capabilities(node, node.models if ollama_models else None)

        # Exchange gossip for collaboration state
        if node.healthy:
            gossip = self._fetch_gossip(node)
            if gossip:
                try:
                    from collab import get_hub

                    get_hub().merge_gossip(node.hostname, gossip)
                except Exception:
                    pass

        node.last_check = now

    def _check_ollama(self, node: MaudeNode) -> dict[str, str] | None:
        """Check if node has Ollama running and get available models."""
        try:
            host = node.tailscale_ip or node.ip or node.hostname
            url = f"http://{host}:{node.ollama_port}/api/tags"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                models = {}
                for model in data.get("models", []):
                    name = model.get("name", "")
                    # Extract base model name
                    base_name = name.split(":")[0] if ":" in name else name
                    models[base_name] = name
                return models
        except:
            pass
        return None

    def _check_api(self, node: MaudeNode) -> bool:
        """Check if node's main API is responding."""
        try:
            host = node.tailscale_ip or node.ip or node.hostname
            url = f"http://{host}:{node.api_port}/v1/models"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def _check_comfyui(self, node: MaudeNode) -> dict | None:
        """Check if ComfyUI is running on this node."""
        try:
            host = node.tailscale_ip or node.ip or node.hostname
            url = f"http://{host}:{node.comfyui_port}/system_stats"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def _fetch_capability_manifest(self, node: MaudeNode) -> list[dict] | None:
        """Fetch capability manifest from a remote node's /v1/capabilities endpoint."""
        try:
            host = node.tailscale_ip or node.ip or node.hostname
            url = f"http://{host}:{node.api_port}/v1/capabilities"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("capabilities", [])
        except:
            pass
        return None

    def _fetch_gossip(self, node: MaudeNode) -> dict | None:
        """Fetch collaboration gossip bundle from a remote node."""
        try:
            host = node.tailscale_ip or node.ip or node.hostname
            url = f"http://{host}:{node.api_port}/api/collab/gossip"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def _build_node_capabilities(self, node: MaudeNode, ollama_models: dict[str, str] = None):
        """Build the capabilities list for a node."""
        host = node.tailscale_ip or node.ip or node.hostname
        caps = []

        # First try to fetch capability manifest from remote node
        manifest = self._fetch_capability_manifest(node)
        if manifest:
            for cap_data in manifest:
                try:
                    cap = Capability.from_dict(cap_data)
                    caps.append(cap)
                except Exception:
                    pass

        # If no manifest, build from detected services
        if not caps:
            # Add capabilities from Ollama models
            if ollama_models:
                for _base_name, full_name in ollama_models.items():
                    cap = capability_from_ollama_model(full_name, host, node.ollama_port)
                    caps.append(cap)

            # Check for ComfyUI
            comfyui_stats = self._check_comfyui(node)
            if comfyui_stats:
                comfyui_cap = capability_for_comfyui(host, node.comfyui_port)
                # Also add as VIDEO_GEN capability
                video_cap = Capability(
                    type=CapabilityType.VIDEO_GEN,
                    name="mochi",
                    endpoint_type=EndpointType.COMFYUI,
                    endpoint_url=f"http://{host}:{node.comfyui_port}",
                    port=node.comfyui_port,
                    models=["mochi"],
                    metadata={"via_comfyui": True},
                )
                caps.append(comfyui_cap)
                caps.append(video_cap)

        # Load manual capabilities from config
        manual_caps = self._load_manual_capabilities(node.hostname)
        caps.extend(manual_caps)

        node.node_capabilities = caps

    def _load_manual_capabilities(self, hostname: str) -> list[Capability]:
        """Load manually registered capabilities for a node from config."""
        caps = []
        config_file = runtime_paths().config_dir / "capabilities.json"

        if not config_file.exists():
            return caps

        try:
            data = json.loads(config_file.read_text())
            node_caps = data.get(hostname.lower(), [])
            for cap_data in node_caps:
                try:
                    cap = Capability.from_dict(cap_data)
                    caps.append(cap)
                except Exception:
                    pass
        except Exception:
            pass

        return caps

    def register_capability(self, hostname: str, capability: Capability) -> bool:
        """Register a capability manually for a node."""
        config_file = runtime_paths().config_dir / "capabilities.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text())
            except:
                pass

        hostname_key = hostname.lower()
        if hostname_key not in data:
            data[hostname_key] = []

        data[hostname_key].append(capability.to_dict())
        config_file.write_text(json.dumps(data, indent=2))

        # Refresh node capabilities
        with self._lock:
            if hostname_key in self.nodes:
                node = self.nodes[hostname_key]
                self._build_node_capabilities(node, node.models)

        return True

    def find_node_for_model(self, model_name: str) -> MaudeNode | None:
        """Find a healthy node that has the specified model."""
        with self._lock:
            candidates = [n for n in self.nodes.values() if n.healthy and model_name in n.capabilities]

        if not candidates:
            return None

        # Return node with lowest load
        return min(candidates, key=lambda n: n.load)

    def get_remote_ollama_url(self, model_name: str) -> str | None:
        """Get Ollama URL for a model, checking remote nodes if not local."""
        node = self.find_node_for_model(model_name)
        if node:
            return node.get_ollama_url()
        return None

    def list_nodes(self) -> list[MaudeNode]:
        """List all known nodes."""
        with self._lock:
            return list(self.nodes.values())

    def list_available_models(self) -> dict[str, list[str]]:
        """List all available models across the mesh."""
        models = {}
        with self._lock:
            for node in self.nodes.values():
                if node.healthy:
                    for model in node.capabilities:
                        if model not in models:
                            models[model] = []
                        models[model].append(node.hostname)
        return models

    def refresh(self):
        """Force refresh of node discovery and health checks."""
        self._discover_nodes()
        self._check_all_nodes()


# Global mesh instance
_mesh: MaudeMesh | None = None


def get_mesh(capabilities: list[str] = None) -> MaudeMesh:
    """Get or create the global mesh instance."""
    global _mesh
    if _mesh is None:
        _mesh = MaudeMesh(capabilities)
    return _mesh
