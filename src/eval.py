import json
from pathlib import Path
from typing import List, Optional

import click
import wandb

from absa import AbsaAnswer
from pydantic import BaseModel


def alpaca_to_absa(filepath: Path) -> List[AbsaAnswer | None]:
    """
    Convert an Alpaca-formatted JSONL file to a list of AbsaAnswer objects.
    """
    with open(filepath, "r") as f:
        answers = []

        for line in f:
            output = json.loads(line)["output"]
            try:
                output_dict = json.loads(output)
                answers.append(AbsaAnswer(**output_dict))
            except json.JSONDecodeError:
                print(f"Error parsing output as JSON: {output}")
                answers.append(None)

    return answers


# Create a class to represent the evaluation metrics
class AbsaRowMetrics(BaseModel):
    malformed: bool
    tp: int
    fp: int
    fn: int


def evaluate(predicted: AbsaAnswer | None, truth: AbsaAnswer) -> AbsaRowMetrics:
    """
    Evaluate the model's output against the expected output.
    """
    metrics = {
        "malformed": False,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    if predicted is None:
        metrics["malformed"] = True
        return AbsaRowMetrics(**metrics)

    # Compare the predicted and expected aspects
    pred_set = set((a.term, a.polarity) for a in predicted.aspects)
    exp_set = set((a.term, a.polarity) for a in truth.aspects)

    metrics["tp"] = len(pred_set & exp_set)
    metrics["fp"] = len(pred_set - exp_set)
    metrics["fn"] = len(exp_set - pred_set)

    # If both sets are empty, the prediction is correct
    if len(pred_set) == 0 and len(exp_set) == 0:
        metrics["tp"] = 1

    return AbsaRowMetrics(**metrics)


class AbsaMetrics(BaseModel):
    malformed: int
    malformed_pct: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    examples: int


def aggregate_metrics(row_metrics=List[AbsaRowMetrics]) -> AbsaMetrics:
    """
    Aggregate the row-level metrics to compute the overall metrics.
    """
    agg_metrics = {
        "malformed": sum(r.malformed for r in row_metrics),
        "malformed_pct": sum(r.malformed for r in row_metrics) / len(row_metrics),
        "tp": sum(r.tp for r in row_metrics),
        "fp": sum(r.fp for r in row_metrics),
        "fn": sum(r.fn for r in row_metrics),
        "examples": len(row_metrics),
    }

    agg_metrics["precision"] = agg_metrics["tp"] / (
        agg_metrics["tp"] + agg_metrics["fp"]
    )
    agg_metrics["recall"] = agg_metrics["tp"] / (agg_metrics["tp"] + agg_metrics["fn"])
    agg_metrics["f1"] = (
        2
        * agg_metrics["precision"]
        * agg_metrics["recall"]
        / (agg_metrics["precision"] + agg_metrics["recall"])
    )

    return AbsaMetrics(**agg_metrics)


def upload_metrics(
    metrics: AbsaMetrics,
    predictions_file: Path,
    wandb_id: str,
    wandb_project: str = "absa-semeval2014",
) -> None:
    """
    Upload metrics and predictions to W&B.
    """
    wandb.init(id=wandb_id, project=wandb_project)
    wandb.log(metrics.dict())

    # Upload the predictions file
    wandb.save(predictions_file)

    wandb.finish()


@click.command()
@click.option(
    "--wandb-id",
    help="The id of the W&B run to which to write the metrics.",
)
def main(wandb_id: Optional[str] = None) -> None:
    """
    Evaluate the model's predictions against the gold standard. The metrics are added to the W&B run.
    """

    data_dir = Path("data/semeval2014")

    truth_path = data_dir / "semeval2014_test.jsonl"
    predictions_path = data_dir / "semeval2014_test_predictions.jsonl"

    predictions = alpaca_to_absa(predictions_path)
    truth = alpaca_to_absa(truth_path)

    row_metrics = [evaluate(pred, gold) for pred, gold in zip(predictions, truth)]

    absa_metrics = aggregate_metrics(row_metrics)

    if wandb_id:
        upload_metrics(
            metrics=absa_metrics, predictions_file=predictions_path, wandb_id=wandb_id
        )
    else:
        print(absa_metrics)


if __name__ == "__main__":
    main()
