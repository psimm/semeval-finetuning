import json
from pathlib import Path
from typing import Literal

import click
import polars as pl
import xmltodict
from absa import AbsaAnswer


def read_semeval_xml(filepath: Path) -> pl.DataFrame:
    """
    Read a SemEval 2014 XML file and return a DataFrame.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    with filepath.open("rb") as f:
        parsed_dict = xmltodict.parse(f.read())

    sentences_list = parsed_dict["sentences"]["sentence"]

    for sentence in sentences_list:
        sentence["sentence_id"] = sentence["@id"]
        sentence["aspect_terms"] = []

        if "aspectTerms" in sentence:
            aspect_terms = sentence["aspectTerms"]["aspectTerm"]
            if isinstance(aspect_terms, dict):
                aspect_terms = [aspect_terms]
            for aspect_term in aspect_terms:
                aterm = {
                    "to": aspect_term["@to"],
                    "from": aspect_term["@from"],
                    "term": aspect_term["@term"],
                    "polarity": aspect_term["@polarity"],
                }
                sentence["aspect_terms"].append(aterm)

    df = pl.DataFrame(sentences_list).select("sentence_id", "text", "aspect_terms")
    return df


def remove_conflict_examples(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove examples with conflict polarity from the DataFrame.
    """

    def has_conflict(aspect_terms: list[dict]) -> bool:
        return any(term["polarity"] == "conflict" for term in aspect_terms)

    n_before = df.shape[0]
    filtered = df.filter(
        ~pl.col("aspect_terms").map_elements(has_conflict, return_dtype=pl.Boolean)
    )
    print(f"Removed {n_before - len(filtered)} examples with conflict polarity")

    return filtered


def write_alpaca(df: pl.DataFrame, out_path: Path) -> None:
    """
    Write the DataFrame to a JSONL file in Alpaca format.
    Example dataset: https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test
    """
    examples = [
        json.dumps(
            {
                "instruction": "",
                "input": input_,
                "output": json.dumps(AbsaAnswer(aspects=output).dict()),
            }
        )
        for input_, output in zip(df["text"].to_list(), df["aspect_terms"].to_list())
    ]
    out_path.write_text("\n".join(examples) + "\n", encoding="utf-8")


def write_input_output(df: pl.DataFrame, out_path: Path) -> None:
    """
    Write the DataFrame to a JSONL file with a template-free input_output format
    https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html
    """
    examples = [
        json.dumps(
            {
                "segments": [
                    {"label": False, "text": "<s>" + input_ + "\n"},
                    {
                        "label": True,
                        "text": json.dumps(AbsaAnswer(aspects=output).dict()) + "<\s>",
                    },
                ]
            }
        )
        for input_, output in zip(df["text"].to_list(), df["aspect_terms"].to_list())
    ]
    out_path.write_text("\n".join(examples) + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--format",
    type=click.Choice(["alpaca", "input_output"]),
    default="alpaca",
    help="Output format",
)
def main(
    format: Literal["alpaca", "input_output"] = "alpaca",
):
    print("Preparing SemEval 2014 data")
    data_dir = Path("data/semeval2014")

    # Read all files and merge them into one DataFrame
    files = [
        {"domain": "laptops", "split": "train", "path": data_dir / "laptops_train.xml"},
        {"domain": "laptops", "split": "test", "path": data_dir / "laptops_test.xml"},
        {
            "domain": "restaurants",
            "split": "train",
            "path": data_dir / "restaurants_train.xml",
        },
        {
            "domain": "restaurants",
            "split": "test",
            "path": data_dir / "restaurants_test.xml",
        },
    ]

    dfs = []
    for file in files:
        print(f"Reading {file['path']}")
        df = read_semeval_xml(file["path"])
        df = df.with_columns(domain=pl.lit(file["domain"]), split=pl.lit(file["split"]))
        dfs.append(df)

    df_complete = pl.concat(dfs)

    df_clean = remove_conflict_examples(df_complete)

    # Write cleaned data to disk for EDA
    df_path = data_dir / "cleaned.parquet"
    df_clean.write_parquet(df_path)
    print(f"Cleaned data written to {df_path}")

    output_fun = write_alpaca if format == "alpaca" else write_input_output

    df_train = df_clean.filter(pl.col("split") == "train")
    jsonl_train_path = data_dir / "semeval2014_train.jsonl"
    output_fun(df_train, jsonl_train_path)
    print(f"Training data written to {jsonl_train_path} in {format} format")

    df_test = df_clean.filter(pl.col("split") == "test")
    jsonl_test_path = data_dir / "semeval2014_test.jsonl"
    output_fun(df_test, jsonl_test_path)
    print(f"Test data written to {jsonl_test_path} in {format} format")


if __name__ == "__main__":
    main()
