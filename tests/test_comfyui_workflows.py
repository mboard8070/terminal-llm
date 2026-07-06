import json

from maude.tools.domains import media as media_domain
from maude.tools.handlers import media


def test_flux2_klein_workflow_loader_and_patcher(tmp_path):
    workflow_path = tmp_path / "flux2_klein_4b.json"
    workflow_path.write_text(
        json.dumps(
            {
                "prompt": {
                    "1": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {"text": "{{prompt}}"},
                    },
                    "2": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {"width": 512, "height": 512},
                    },
                    "3": {
                        "class_type": "KSampler",
                        "inputs": {"seed": 1, "steps": 4},
                    },
                    "4": {
                        "class_type": "SaveImage",
                        "inputs": {"filename_prefix": "old"},
                    },
                }
            }
        )
    )

    workflow = media._load_local_comfyui_workflow("flux2-klein-4b", str(workflow_path))
    media._patch_comfyui_workflow(
        workflow,
        "a neon glass fox",
        width=1344,
        height=768,
        seed=123,
        steps=16,
        filename_prefix="maude/gen_flux2_klein_123",
    )

    assert workflow["1"]["inputs"]["text"] == "a neon glass fox"
    assert workflow["2"]["inputs"]["width"] == 1344
    assert workflow["2"]["inputs"]["height"] == 768
    assert workflow["3"]["inputs"]["seed"] == 123
    assert workflow["3"]["inputs"]["steps"] == 16
    assert workflow["4"]["inputs"]["filename_prefix"] == "maude/gen_flux2_klein_123"


def test_generate_image_schema_exposes_local_flux2_klein_model():
    generate_image = next(
        schema for schema in media_domain.SCHEMAS if schema["function"]["name"] == "generate_image"
    )
    props = generate_image["function"]["parameters"]["properties"]

    assert "model" in props
    assert "flux2-klein-4b" in props["model"]["enum"]
    assert "workflow_path" in props


def test_flux2_klein_has_dedicated_local_tool_schema():
    tool = next(
        schema for schema in media_domain.SCHEMAS if schema["function"]["name"] == "generate_image_flux2_klein"
    )
    props = tool["function"]["parameters"]["properties"]

    assert "workflow_path" in props
    assert "Flux2 Klein 4B" in tool["function"]["description"]


def test_flux2_klein_message_selects_local_not_cloud_tool():
    from maude_core.tool_groups import get_tools_for_message

    names = {tool["function"]["name"] for tool in get_tools_for_message("generate with flux2 klein 4b")}

    assert "generate_image_flux2_klein" in names
    assert "generate_image" in names
    assert "generate_image_flux2" not in names
