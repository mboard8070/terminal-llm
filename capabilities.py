"""
MAUDE Capability Types and Data Structures.

Defines the capability types and endpoint configurations for the multi-modal mesh network.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityType(Enum):
    """Types of capabilities that can be provided by mesh nodes."""

    LLM = "llm"  # Text generation (chat, code, etc.)
    VISION = "vision"  # Image understanding/analysis
    IMAGE_GEN = "image_gen"  # Image generation (Stable Diffusion, DALL-E, etc.)
    VIDEO_GEN = "video_gen"  # Video generation (Mochi, etc.)
    MODEL_3D = "model_3d"  # 3D model generation (Meshy, etc.)
    GAUSSIAN_SPLAT = "gaussian_splat"  # Gaussian splatting for 3D reconstruction
    COMFYUI = "comfyui"  # ComfyUI workflow server


class EndpointType(Enum):
    """Types of API endpoints."""

    OLLAMA = "ollama"  # Ollama API
    OPENAI_COMPAT = "openai_compat"  # OpenAI-compatible API (llama.cpp, vLLM, etc.)
    COMFYUI = "comfyui"  # ComfyUI workflow API
    CUSTOM = "custom"  # Custom REST API


@dataclass
class Capability:
    """A capability offered by a mesh node."""

    type: CapabilityType
    name: str  # "mochi", "codestral", "sd-xl", etc.
    endpoint_type: EndpointType
    endpoint_url: str  # Base URL for the endpoint
    port: int
    models: list[str] = field(default_factory=list)  # Available models for this capability
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional config
    healthy: bool = True
    load: float = 0.0  # Current load 0.0-1.0
    max_concurrent: int = 1  # Max concurrent requests
    error_count: int = 0  # Recent errors for routing decisions

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "name": self.name,
            "endpoint_type": self.endpoint_type.value,
            "endpoint_url": self.endpoint_url,
            "port": self.port,
            "models": self.models,
            "metadata": self.metadata,
            "healthy": self.healthy,
            "load": self.load,
            "max_concurrent": self.max_concurrent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Capability":
        """Create from dictionary."""
        return cls(
            type=CapabilityType(data["type"]),
            name=data["name"],
            endpoint_type=EndpointType(data["endpoint_type"]),
            endpoint_url=data["endpoint_url"],
            port=data["port"],
            models=data.get("models", []),
            metadata=data.get("metadata", {}),
            healthy=data.get("healthy", True),
            load=data.get("load", 0.0),
            max_concurrent=data.get("max_concurrent", 1),
        )

    def matches(self, cap_type: CapabilityType = None, name: str = None) -> bool:
        """Check if this capability matches the given criteria."""
        if cap_type and self.type != cap_type:
            return False
        return not (name and self.name.lower() != name.lower())


# Well-known capability names for classification
VISION_MODELS = {"llava", "llava-llama3", "llava-phi3", "moondream", "bakllava", "minicpm-v"}
CODE_MODELS = {"codestral", "codellama", "qwen2.5-coder", "deepseek-coder", "starcoder"}
EMBEDDING_MODELS = {"nomic-embed-text", "mxbai-embed-large", "all-minilm", "bge"}


def classify_ollama_model(model_name: str) -> CapabilityType:
    """Classify an Ollama model into a capability type."""
    base_name = model_name.split(":")[0].lower()

    # Check for vision models
    for vision in VISION_MODELS:
        if vision in base_name:
            return CapabilityType.VISION

    # All other text models are LLMs
    return CapabilityType.LLM


def capability_from_ollama_model(model_name: str, host: str, port: int = 11434) -> Capability:
    """Create a Capability from an Ollama model."""
    cap_type = classify_ollama_model(model_name)
    base_name = model_name.split(":")[0]

    return Capability(
        type=cap_type,
        name=base_name,
        endpoint_type=EndpointType.OLLAMA,
        endpoint_url=f"http://{host}:{port}/v1",
        port=port,
        models=[model_name],
        healthy=True,
    )


def capability_for_comfyui(host: str, port: int = 8188, workflows: list[str] = None) -> Capability:
    """Create a ComfyUI capability."""
    return Capability(
        type=CapabilityType.COMFYUI,
        name="comfyui",
        endpoint_type=EndpointType.COMFYUI,
        endpoint_url=f"http://{host}:{port}",
        port=port,
        models=workflows or ["mochi", "stable-diffusion"],
        metadata={"workflows": workflows or []},
        healthy=True,
    )
