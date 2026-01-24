"""
MAUDE Mesh Network - Discovery and routing for remote subagents.

Supports:
1. Tailscale auto-discovery (discovers peers via tailscale status)
2. Manual node configuration (MAUDE_NODES env var)
3. Health checks and capability detection
"""

import os
import json
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from rich.console import Console

console = Console()


@dataclass
class MaudeNode:
    """Represents a MAUDE instance on the network."""
    hostname: str
    ip: str
    tailscale_ip: Optional[str] = None
    api_port: int = 30000
    ollama_port: int = 11434
    capabilities: List[str] = field(default_factory=list)
    models: Dict[str, str] = field(default_factory=dict)  # model_name -> model_id
    last_seen: float = 0.0
    last_check: float = 0.0
    healthy: bool = False
    load: float = 0.0  # 0.0-1.0 current utilization

    def get_ollama_url(self) -> str:
        """Get the Ollama API URL for this node."""
        host = self.tailscale_ip or self.ip or self.hostname
        return f"http://{host}:{self.ollama_port}/v1"

    def get_api_url(self) -> str:
        """Get the main API URL for this node."""
        host = self.tailscale_ip or self.ip or self.hostname
        return f"http://{host}:{self.api_port}/v1"


@dataclass
class MeshConfig:
    """Configuration for the MAUDE mesh."""
    enabled: bool = True
    health_check_interval: int = 60  # seconds
    node_timeout: int = 120  # seconds before marking node as dead
    discovery_port: int = 31337
    auto_discover_tailscale: bool = True
    manual_nodes: List[str] = field(default_factory=list)  # ["host:port", ...]


class MaudeMesh:
    """Mesh network of MAUDE instances for remote subagent routing."""

    CONFIG_FILE = Path.home() / ".config" / "maude" / "mesh.json"

    def __init__(self, my_capabilities: List[str] = None):
        self.my_hostname = socket.gethostname()
        self.my_capabilities = my_capabilities or []
        self.nodes: Dict[str, MaudeNode] = {}
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
        console.print(f"[dim]Starting MAUDE mesh network...[/dim]")

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
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return

            data = json.loads(result.stdout)
            peers = data.get("Peer", {})

            for peer_id, peer in peers.items():
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
        port = int(parts[1]) if len(parts) > 1 else 30000

        if hostname.lower() == self.my_hostname.lower():
            return

        with self._lock:
            if hostname not in self.nodes:
                self.nodes[hostname] = MaudeNode(
                    hostname=hostname,
                    ip=hostname,
                    api_port=port
                )

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

        node.last_check = now

    def _check_ollama(self, node: MaudeNode) -> Optional[Dict[str, str]]:
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

    def find_node_for_model(self, model_name: str) -> Optional[MaudeNode]:
        """Find a healthy node that has the specified model."""
        with self._lock:
            candidates = [
                n for n in self.nodes.values()
                if n.healthy and model_name in n.capabilities
            ]

        if not candidates:
            return None

        # Return node with lowest load
        return min(candidates, key=lambda n: n.load)

    def find_node_for_capability(self, capability: str) -> Optional[MaudeNode]:
        """Find a healthy node with the specified capability."""
        return self.find_node_for_model(capability)

    def get_remote_ollama_url(self, model_name: str) -> Optional[str]:
        """Get Ollama URL for a model, checking remote nodes if not local."""
        node = self.find_node_for_model(model_name)
        if node:
            return node.get_ollama_url()
        return None

    def list_nodes(self) -> List[MaudeNode]:
        """List all known nodes."""
        with self._lock:
            return list(self.nodes.values())

    def list_available_models(self) -> Dict[str, List[str]]:
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
_mesh: Optional[MaudeMesh] = None


def get_mesh(capabilities: List[str] = None) -> MaudeMesh:
    """Get or create the global mesh instance."""
    global _mesh
    if _mesh is None:
        _mesh = MaudeMesh(capabilities)
    return _mesh


def handle_mesh_command(args: list) -> str:
    """Handle /mesh command."""
    mesh = get_mesh()

    if not args:
        # Show mesh status
        nodes = mesh.list_nodes()
        if not nodes:
            return (
                "No mesh nodes discovered.\n\n"
                "To add nodes:\n"
                "  - Ensure Tailscale is connected (for auto-discovery)\n"
                "  - Or set MAUDE_NODES=host1:port,host2:port\n"
                "  - Or use: /mesh add <hostname>"
            )

        lines = ["MAUDE Mesh Network:\n"]
        for node in nodes:
            status = "[green]✓[/green]" if node.healthy else "[red]✗[/red]"
            ts_ip = f" ({node.tailscale_ip})" if node.tailscale_ip else ""
            lines.append(f"  {status} {node.hostname}{ts_ip}")
            if node.capabilities:
                lines.append(f"      Models: {', '.join(node.capabilities)}")
            if node.last_seen:
                ago = int(time.time() - node.last_seen)
                lines.append(f"      Last seen: {ago}s ago")

        # Show available models across mesh
        models = mesh.list_available_models()
        if models:
            lines.append("\nAvailable models across mesh:")
            for model, hosts in sorted(models.items()):
                lines.append(f"  {model}: {', '.join(hosts)}")

        return "\n".join(lines)

    action = args[0].lower()

    if action == "refresh":
        mesh.refresh()
        return "Mesh refreshed. Use /mesh to see status."

    elif action == "add" and len(args) > 1:
        node_spec = args[1]
        mesh._add_manual_node(node_spec)
        mesh.config.manual_nodes.append(node_spec)
        mesh._save_config()
        mesh._check_all_nodes()
        return f"Added node: {node_spec}"

    elif action == "remove" and len(args) > 1:
        hostname = args[1].lower()
        with mesh._lock:
            if hostname in mesh.nodes:
                del mesh.nodes[hostname]
                mesh.config.manual_nodes = [
                    n for n in mesh.config.manual_nodes
                    if not n.lower().startswith(hostname)
                ]
                mesh._save_config()
                return f"Removed node: {hostname}"
        return f"Node not found: {hostname}"

    elif action == "enable":
        mesh.config.enabled = True
        mesh._save_config()
        mesh.start()
        return "Mesh networking enabled"

    elif action == "disable":
        mesh.config.enabled = False
        mesh._save_config()
        mesh.stop()
        return "Mesh networking disabled"

    return f"Unknown mesh command: {action}\n\nUsage: /mesh [refresh|add <host>|remove <host>|enable|disable]"
