# ABSA with open source LLMs

## Overview

This repository contains code to train and evaluate a large language model (LLM) on the SemEval 2014 dataset for Aspect-Based Sentiment Analysis (ABSA). The LLM is fine-tuned on the SemEval 2014 dataset and evaluated on the test set.

See the [article](https://simmering.dev/open-absa) for more information and evaluation results.

This repository is a fork of [modal-labs/llm-finetuning](https://github.com/modal-labs/llm-finetuning). See the original repository for a general overview.

## Requirements

This project uses Modal for training and inference. You'll need a Modal account to run the training and inference scripts. You can sign up for a free account at [https://modal.com](https://modal.com). The free tier is enough to train and evaluate multiple models.

Weights & Biases can be used for tracking experiments. A free account can be created at [https://wandb.ai](https://wandb.ai). If you use W&B, set the `ALLOW_WANDB` environment variable to `true` and edit the wandb configuration in the axolotl config files. Also set the `wandb` secret in Modal.

To download the [Llama-3-8B model](https://huggingface.co/meta-llama/Meta-Llama-3-8B), a HuggingFace account is required. Further, access has to be requested for the model.

Steps that don't require a GPU run locally. Steps that require a GPU run on Modal. Install the local requirements in a virtual environment and activate it.

```bash
python -m venv venv
pip install -r requirements.txt
source venv/bin/activate
```

## 1: Prepare the data (local)

```bash
python src/prep.py 
```

## 2: Finetune a model using LoRA (Modal)

Prepare an axolotl [config](https://openaccess-ai-collective.github.io/axolotl/docs/config.html) for training. Set a sensible name in the `wandb_name` field.

Using GPUs with less VRAM than the H100 GPU causes OOM errors with some configs. Using multiple GPUs doesn't speed up training much. Use two H100 GPUs for 70B models.

```bash
export ALLOW_WANDB=true  # if you're using Weights & Biases
export GPU_CONFIG=h100  # h100:2 for 70B models
modal run --detach src.train --config config/llama-3-8b-semeval2014.yml --data data/semeval2014/semeval2014_train.jsonl
```

Note the run-name that Modal returns. This will be used for inference.

Optional: quantize the merged model.

```bash
modal run --detach src.quantize --run <run-name>
```

## 3: Run inference on the test set (Modal)

```bash
modal run src.batch_inference --run-name <run-name> --prompt-file data/semeval2014/semeval2014_test.jsonl
```

Where run-name is the name of the run with the trained model from the previous step.

## 4: Evaluate the model (local)

```bash
python src/eval.py --wandb-name <wandb-id>
```

Where the wandb-id is the id of the run in Weights & Biases. Leave out the `--wandb-name` argument if you're not using Weights & Biases. Then the metrics will be printed to the console.

## 5: Optional: Push the LoRA adapter to HuggingFace (Modal)

```bash
modal run src.push_lora_adapter --run-name <run-name> --repo-id <huggingface-repo-id>
```

Where run-name is the name of the run with the trained model from step 2 and repo-id is the id of the HuggingFace repository you want to push the adapter to. This is done as a separate step because axolotl can get stuck at the end of a training run if hub_model_id is set in the config. This is a workaround to avoid that. The code in push_lora_adapter.py can also help you load the adapter for inference because that requires a step of adding new tokens to the base model's embedding layer.
