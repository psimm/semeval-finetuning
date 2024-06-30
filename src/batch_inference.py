from typing import List
from pydantic import BaseModel

import json
import time
from pathlib import Path

import modal
import yaml

from .common import VOLUME_CONFIG, app, vllm_image

N_INFERENCE_GPU = 1

with vllm_image.imports():
    from outlines import generate, models
    from vllm import LLM
    from vllm.sampling_params import SamplingParams


def get_model_path_from_run(path: Path, quantized: bool = False) -> Path:
    ending = "merged" if not quantized else "quantized"

    with (path / "config.yml").open() as f:
        return path / yaml.safe_load(f.read())["output_dir"] / ending


class Aspect(BaseModel):
    term: str
    polarity: str


class AbsaAnswer(BaseModel):
    aspects: List[Aspect]


@app.function(
    gpu=modal.gpu.H100(count=N_INFERENCE_GPU),
    image=vllm_image,
    volumes=VOLUME_CONFIG,
    timeout=3600,
)
def batch_completion(
    run_name: str,
    prompts: list[str],
    quantized: bool = False,
    use_outlines: bool = False,
) -> list[str]:

    path = Path("/runs/" + run_name)
    model_path = get_model_path_from_run(path, quantized=quantized)

    print(f"Loading model: {model_path}")

    # Load the (quantized) model in vLLM
    llm_args = {
        "model": str(model_path),
        "tensor_parallel_size": N_INFERENCE_GPU,
        "max_model_len": 2048,
    }

    if quantized:
        llm_args["quantization"] = "AWQ"

    llm = LLM(**llm_args)

    # Set sampling parameters, see https://docs.vllm.ai/en/latest/offline_inference/sampling_params.html
    sampling_params = SamplingParams(
        repetition_penalty=1.1,
        temperature=0,
        top_p=0.95,
        top_k=50,
        max_tokens=128,
    )

    if not use_outlines:

        print("Running inference on vLLM")

        start_time = time.time()

        gen_future = llm.generate(prompts, sampling_params)

        outputs = []

        for output in gen_future:
            generated_text = output.outputs[0].text
            outputs.append(generated_text)

        elapsed_time = time.time() - start_time

        print(f"Generated {len(outputs)} completions in {elapsed_time:.2f} seconds.")

        return outputs

    # Wrap the model with outlines
    # https://outlines-dev.github.io/outlines/reference/models/vllm/
    model = models.VLLM(llm)

    print("Running inference on vLLM wrapped with outlines")

    generator = generate.json(model, AbsaAnswer, whitespace_pattern="")

    start_time = time.time()

    outputs = generator(prompts, sampling_params=sampling_params)

    elapsed_time = time.time() - start_time

    print(f"Generated {len(outputs)} completions in {elapsed_time:.2f} seconds.")

    return outputs


@app.local_entrypoint()
def batch_inference_main(
    run_name: str,
    prompt_file: str,
    quantized: bool = False,
    use_outlines: bool = False,
    add_tags: bool = True,
) -> None:
    """
    Run inference on a batch of prompts using a trained model using vLLM.
    Automatically adds the [INST] and [/INST] tags to the prompts.

    Args:
        run_name: The name of the run directory containing the trained model.
        prompt_file: Path to a JSONL file containing the prompts.
        quantized: Whether to use the model quantized with AWQ.
        use_outlines: Use the outlines library for structured output.
        add_tags: Add [INST] and [/INST] tags to the prompts.
    """

    prompt_file = Path(prompt_file)
    assert prompt_file.exists(), f"Prompt file {prompt_file} does not exist."

    with open(prompt_file) as f:
        prompts = [json.loads(line)["input"] for line in f]

    # Add [INST] and [/INST] to the prompts
    if add_tags:
        prompts = [f"[INST]{prompt}[/INST]" for prompt in prompts]

    print(f"Processing {len(prompts)} prompts")

    outputs = batch_completion.remote(
        run_name, prompts=prompts, quantized=quantized, use_outlines=use_outlines
    )

    # Add a suffix to the prompt file name
    output_file = prompt_file.with_name(prompt_file.stem + "_predictions.jsonl")

    print(f"Writing outputs to {output_file}")

    with open(output_file, "w") as f:
        for prompt, output in zip(prompts, outputs):

            if use_outlines:
                completion = json.dumps(output.dict())
            else:
                completion = output

            json.dump(
                {"instruction": "", "input": prompt, "output": completion},
                f,
                ensure_ascii=False,
            )
            f.write("\n")
