"""
MAUDE Capability Router.

Routes tasks to the best available node based on capability type, load, and health.
"""

import time
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
from rich.console import Console

from capabilities import Capability, CapabilityType, EndpointType

if TYPE_CHECKING:
    from mesh import MaudeNode, MaudeMesh

console = Console()


class CapabilityRouter:
    """Routes capability requests to the best available mesh node."""

    # Routing score weights
    WEIGHT_LOAD = 100        # Higher load = higher score (worse)
    WEIGHT_REMOTE = 50       # Penalty for remote nodes when prefer_local
    WEIGHT_ERROR = 10        # Penalty per recent error
    WEIGHT_UNHEALTHY = 1000  # Large penalty for unhealthy nodes

    def __init__(self, mesh: "MaudeMesh" = None):
        """
        Initialize the router.

        Args:
            mesh: MaudeMesh instance. If None, will get global mesh on first use.
        """
        self._mesh = mesh

    @property
    def mesh(self) -> "MaudeMesh":
        """Get the mesh instance, lazily loading if needed."""
        if self._mesh is None:
            from mesh import get_mesh
            self._mesh = get_mesh()
        return self._mesh

    def _calculate_score(
        self,
        node: "MaudeNode",
        cap: Capability,
        prefer_local: bool = True
    ) -> float:
        """
        Calculate a routing score for a node/capability pair.

        Lower score = better candidate.
        """
        score = 0.0

        # Load factor (0-100)
        score += cap.load * self.WEIGHT_LOAD

        # Remote penalty
        if prefer_local and node.hostname != self.mesh.my_hostname:
            score += self.WEIGHT_REMOTE

        # Error penalty
        score += cap.error_count * self.WEIGHT_ERROR

        # Health penalty
        if not cap.healthy or not node.healthy:
            score += self.WEIGHT_UNHEALTHY

        return score

    def find_capability(
        self,
        cap_type: CapabilityType,
        name: str = None,
        prefer_local: bool = True
    ) -> Optional[Tuple["MaudeNode", Capability]]:
        """
        Find the best node with a specific capability.

        Args:
            cap_type: The capability type to find
            name: Optional specific capability name (e.g., "mochi", "codestral")
            prefer_local: Prefer local capabilities over remote

        Returns:
            Tuple of (MaudeNode, Capability) or None if not found
        """
        candidates = []

        for node in self.mesh.list_nodes():
            if not node.healthy:
                continue

            for cap in node.node_capabilities:
                if cap.matches(cap_type, name) and cap.healthy:
                    score = self._calculate_score(node, cap, prefer_local)
                    candidates.append((score, node, cap))

        if not candidates:
            return None

        # Sort by score (lowest = best)
        candidates.sort(key=lambda x: x[0])
        _, best_node, best_cap = candidates[0]

        return (best_node, best_cap)

    def find_llm(
        self,
        model_name: str = None,
        prefer_local: bool = True
    ) -> Optional[Tuple["MaudeNode", Capability]]:
        """
        Find a node with LLM capability.

        Args:
            model_name: Optional specific model name to find
            prefer_local: Prefer local models over remote

        Returns:
            Tuple of (MaudeNode, Capability) or None if not found
        """
        # Try to find specific model first
        if model_name:
            result = self.find_capability(CapabilityType.LLM, model_name, prefer_local)
            if result:
                return result

            # Check if the model is in any capability's models list
            for node in self.mesh.list_nodes():
                if not node.healthy:
                    continue
                for cap in node.node_capabilities:
                    if cap.type == CapabilityType.LLM and model_name in cap.models:
                        return (node, cap)

        # Fall back to any LLM
        return self.find_capability(CapabilityType.LLM, prefer_local=prefer_local)

    def find_vision(
        self,
        model_name: str = None,
        prefer_local: bool = True
    ) -> Optional[Tuple["MaudeNode", Capability]]:
        """Find a node with vision capability."""
        if model_name:
            result = self.find_capability(CapabilityType.VISION, model_name, prefer_local)
            if result:
                return result

        return self.find_capability(CapabilityType.VISION, prefer_local=prefer_local)

    def find_video_gen(
        self,
        name: str = None,
        prefer_local: bool = True
    ) -> Optional[Tuple["MaudeNode", Capability]]:
        """
        Find a node with video generation capability.

        Args:
            name: Optional specific generator name (e.g., "mochi")
            prefer_local: Prefer local over remote

        Returns:
            Tuple of (MaudeNode, Capability) or None
        """
        # First try VIDEO_GEN type
        result = self.find_capability(CapabilityType.VIDEO_GEN, name, prefer_local)
        if result:
            return result

        # Fall back to ComfyUI that supports video workflows
        if name:
            for node in self.mesh.list_nodes():
                if not node.healthy:
                    continue
                for cap in node.node_capabilities:
                    if cap.type == CapabilityType.COMFYUI:
                        if name in cap.models or name in cap.metadata.get("workflows", []):
                            return (node, cap)

        return self.find_capability(CapabilityType.COMFYUI, prefer_local=prefer_local)

    def find_3d_gen(
        self,
        name: str = None,
        prefer_local: bool = True
    ) -> Optional[Tuple["MaudeNode", Capability]]:
        """
        Find a node with 3D model generation capability.

        Args:
            name: Optional specific generator name
            prefer_local: Prefer local over remote

        Returns:
            Tuple of (MaudeNode, Capability) or None
        """
        return self.find_capability(CapabilityType.MODEL_3D, name, prefer_local)

    def find_image_gen(
        self,
        name: str = None,
        prefer_local: bool = True
    ) -> Optional[Tuple["MaudeNode", Capability]]:
        """
        Find a node with image generation capability.

        Args:
            name: Optional specific generator name (e.g., "sd-xl")
            prefer_local: Prefer local over remote

        Returns:
            Tuple of (MaudeNode, Capability) or None
        """
        # First try IMAGE_GEN type
        result = self.find_capability(CapabilityType.IMAGE_GEN, name, prefer_local)
        if result:
            return result

        # Fall back to ComfyUI that supports image workflows
        return self.find_capability(CapabilityType.COMFYUI, prefer_local=prefer_local)

    def find_comfyui(
        self,
        workflow: str = None,
        prefer_local: bool = True
    ) -> Optional[Tuple["MaudeNode", Capability]]:
        """
        Find a ComfyUI node, optionally with specific workflow support.

        Args:
            workflow: Optional specific workflow name
            prefer_local: Prefer local over remote

        Returns:
            Tuple of (MaudeNode, Capability) or None
        """
        if workflow:
            for node in self.mesh.list_nodes():
                if not node.healthy:
                    continue
                for cap in node.node_capabilities:
                    if cap.type == CapabilityType.COMFYUI:
                        workflows = cap.metadata.get("workflows", [])
                        if workflow in cap.models or workflow in workflows:
                            return (node, cap)

        return self.find_capability(CapabilityType.COMFYUI, prefer_local=prefer_local)

    def list_capabilities(
        self,
        cap_type: CapabilityType = None
    ) -> Dict[str, List[str]]:
        """
        List all capabilities across the mesh, grouped by type.

        Args:
            cap_type: Optional filter by capability type

        Returns:
            Dict mapping capability type names to list of "name @ hostname"
        """
        result: Dict[str, List[str]] = {}

        for node in self.mesh.list_nodes():
            for cap in node.node_capabilities:
                if cap_type and cap.type != cap_type:
                    continue

                type_name = cap.type.value.upper()
                if type_name not in result:
                    result[type_name] = []

                # Format: "name @ hostname (endpoint_type)"
                endpoint_info = f"({cap.endpoint_type.value})" if cap.endpoint_type != EndpointType.OLLAMA else ""
                status = "" if cap.healthy else " [unhealthy]"
                entry = f"{cap.name} @ {node.hostname}{endpoint_info}{status}"

                if entry not in result[type_name]:
                    result[type_name].append(entry)

        return result

    def get_ollama_url_for_model(self, model_name: str) -> Optional[str]:
        """
        Get Ollama URL for a specific model (backward compatible with mesh.get_remote_ollama_url).

        Args:
            model_name: Model name to find

        Returns:
            Ollama API URL or None
        """
        # Check both LLM and VISION capability types
        for cap_type in [CapabilityType.LLM, CapabilityType.VISION]:
            result = self.find_capability(cap_type, model_name, prefer_local=False)
            if result:
                node, cap = result
                if cap.endpoint_type == EndpointType.OLLAMA:
                    host = node.tailscale_ip or node.ip or node.hostname
                    return f"http://{host}:{cap.port}/v1"

        # Also check models lists
        for node in self.mesh.list_nodes():
            if not node.healthy:
                continue
            # Check legacy models dict
            if model_name in node.models:
                host = node.tailscale_ip or node.ip or node.hostname
                return f"http://{host}:{node.ollama_port}/v1"
            # Check node_capabilities
            for cap in node.node_capabilities:
                if cap.endpoint_type == EndpointType.OLLAMA and model_name in cap.models:
                    host = node.tailscale_ip or node.ip or node.hostname
                    return f"http://{host}:{cap.port}/v1"

        return None


# Global router instance
_router: Optional[CapabilityRouter] = None


def get_router(mesh: "MaudeMesh" = None) -> CapabilityRouter:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = CapabilityRouter(mesh)
    return _router
