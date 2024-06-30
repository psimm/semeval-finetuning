from .common import app, vllm_image, VOLUME_CONFIG, HOURS
from .train import SINGLE_GPU_CONFIG

from pathlib import Path


@app.function(
    image=vllm_image,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def quantize(run: str):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = str(Path("/runs") / run / "lora-out" / "merged")

    # Change the path to save the quantized model
    quant_path = str(Path(model_path).parent / "quantized")

    print(f"Quantizing model from {model_path} to {quant_path}")

    # Load model
    # https://docs.vllm.ai/en/stable/quantization/auto_awq.html
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    print(f"Quantized model saved to {quant_path}")

    tokenizer.save_pretrained(quant_path)
    print(f"Tokenizer saved to {quant_path}")


@app.local_entrypoint()
def run_quantize(run: str):
    quantize.remote(run=run)
