"""
3D generation tool — TRELLIS 2 (local) + Tripo (cloud) via Forge3D.
"""

import sys
from pathlib import Path

from tool_registry import register_tool

from .log import log

# Add forge3d to path
FORGE3D_DIR = Path(__file__).parent.parent.parent / "forge3d"
if FORGE3D_DIR.exists():
    sys.path.insert(0, str(FORGE3D_DIR))


def _get_clients():
    from trellis_client import TrellisClient
    from tripo_client import TripoClient

    return TrellisClient(), TripoClient()


@register_tool("generate_3d")
def _dispatch_generate_3d(args):
    import time
    from concurrent.futures import ThreadPoolExecutor

    image_path = args.get("image_path", "")
    engine = args.get("engine", "trellis")
    resolution = args.get("resolution", "1024")
    seed = args.get("seed", 0)
    quad = args.get("quad", False)
    face_limit = args.get("face_limit")

    if not image_path or not Path(image_path).exists():
        return f"Error: Image not found at {image_path}"

    trellis, tripo = _get_clients()
    results = []

    if engine in ("trellis", "both"):
        if not trellis.available:
            results.append("TRELLIS: Offline (Docker container not running on port 7860)")
        else:
            try:
                t0 = time.time()
                glb = trellis.generate(image_path, seed=seed, resolution=resolution)
                elapsed = time.time() - t0
                results.append(f"TRELLIS: {glb} ({elapsed:.0f}s, free)")
            except Exception as e:
                results.append(f"TRELLIS: Error - {e}")

    if engine in ("tripo", "both"):
        if not tripo.available:
            results.append("Tripo: No API key configured")
        else:
            try:
                t0 = time.time()
                glb, task_id = tripo.image_to_3d(
                    image_path,
                    quad=quad,
                    face_limit=face_limit,
                )
                elapsed = time.time() - t0
                results.append(f"Tripo: {glb} ({elapsed:.0f}s, ~30 credits) | task_id: {task_id}")
            except Exception as e:
                results.append(f"Tripo: Error - {e}")

    return "\n".join(results) if results else "Error: No engine selected"
