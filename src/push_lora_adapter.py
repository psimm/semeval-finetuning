# Push a LoRA adapter to the Hugging Face Hub.
# This is a workaround to a bug in axolotl that stalls the training process
# at the end when hub_model_id is set in the config.yml.

import os
from pathlib import Path

import modal
import yaml

from .common import VOLUME_CONFIG, app

GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100")

hf_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install("peft", "transformers", "huggingface_hub", "hf-transfer")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)


def get_base_model_from_run(path: Path) -> Path:
    with (path / "config.yml").open() as f:
        return yaml.safe_load(f.read())["base_model"]


def get_lora_path_from_run(path: Path) -> Path:
    with (path / "config.yml").open() as f:
        return path / yaml.safe_load(f.read())["output_dir"]


@app.function(
    image=hf_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
)
def push_lora_adapter(run_name: str, repo_id: str):
    """
    Load a local LoRA adapter and push it to the Hugging Face Hub.
    """
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM
    from huggingface_hub import snapshot_download

    path = Path("/runs/" + run_name)
    base_model = get_base_model_from_run(path)
    print("Base model:", base_model)

    lora_path = get_lora_path_from_run(path)
    print("LoRA path:", lora_path)

    # Check that the base model is downloaded to the volume
    snapshot_download(base_model, local_files_only=True)
    print(f"Volume contains {base_model}.")

    # Check that the LoRA adapter can be loaded
    # and is compatible with the base model
    peft_config = PeftConfig.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model),
        lora_path,
    )
    print("Model and LoRA adapter loaded.")

    # Push the LoRA adapter to the Hugging Face Hub
    model.push_to_hub(repo_id, use_auth_token=True)
    peft_config.push_to_hub(repo_id, use_auth_token=True)
    print(f"Pushed LoRA adapter to {repo_id}.")


@app.local_entrypoint()
def push_lora_adapter_main(run_name: str, repo_id: str):
    push_lora_adapter.remote(run_name, repo_id)
