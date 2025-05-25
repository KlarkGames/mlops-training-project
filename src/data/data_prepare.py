import json
import os

import click
import pandas as pd
from num2words import num2words


@click.command()
@click.option("--numbers-data-path", default="data/numbers", help="Path to numbers data")
@click.option("--words-data-path", default="data/words", help="Path to words data")
@click.option("--tokens-data-path", default="data/tokens", help="Path to tokens data")
def main(
    numbers_data_path: str,
    words_data_path: str,
    tokens_data_path: str,
):
    numbers_vocabulary = "1234567890"
    numbers_vocabulary_json = os.path.join(numbers_data_path, "vocabulary.json")
    with open(numbers_vocabulary_json, "w", encoding="utf-8") as f:
        json.dump(numbers_vocabulary, f)

    dev_df = pd.read_csv(os.path.join(numbers_data_path, "dev.csv"))
    train_df = pd.read_csv(os.path.join(numbers_data_path, "train.csv"))

    # Creating Words representation of dataset
    words_vocabulary = "абвгдежзийклмнопрстуфхцчшщъыьэюя -"
    os.makedirs(words_data_path, exist_ok=True)
    words_vocabulary_json = os.path.join(words_data_path, "vocabulary.json")
    with open(words_vocabulary_json, "w", encoding="utf-8") as f:
        json.dump(words_vocabulary, f)

    words_dev_df = dev_df.copy()
    words_dev_df["transcription"] = words_dev_df["transcription"].apply(lambda x: numbers_to_words(str(x)))
    words_train_df = train_df.copy()
    words_train_df["transcription"] = words_train_df["transcription"].apply(lambda x: numbers_to_words(str(x)))
    words_dev_df.to_csv(os.path.join(words_data_path, "dev.csv"), index=False)
    words_train_df.to_csv(os.path.join(words_data_path, "train.csv"), index=False)

    os.symlink(
        os.path.abspath(os.path.join(numbers_data_path, "train")),
        os.path.abspath(os.path.join(words_data_path, "train")),
    )
    os.symlink(
        os.path.abspath(os.path.join(numbers_data_path, "dev")),
        os.path.abspath(os.path.join(words_data_path, "dev")),
    )

    # Creating Token representation of dataset
    tokens_vocabulary = [
        "<1>",
        "<2>",
        "<3>",
        "<4>",
        "<5>",
        "<6>",
        "<7>",
        "<8>",
        "<9>",
        "<10>",
        "<20>",
        "<30>",
        "<40>",
        "<50>",
        "<60>",
        "<70>",
        "<80>",
        "<90>",
        "<100>",
        "<200>",
        "<300>",
        "<400>",
        "<500>",
        "<600>",
        "<700>",
        "<800>",
        "<900>",
        "|",
    ]
    os.makedirs(tokens_data_path, exist_ok=True)
    tokens_vocabulary_json = os.path.join(tokens_data_path, "vocabulary.json")
    with open(tokens_vocabulary_json, "w", encoding="utf-8") as f:
        json.dump(tokens_vocabulary, f)

    tokens_dev_df = dev_df.copy()
    tokens_dev_df["transcription"] = tokens_dev_df["transcription"].apply(lambda x: numbers_to_tokens(str(x)))
    tokens_train_df = train_df.copy()
    tokens_train_df["transcription"] = tokens_train_df["transcription"].apply(lambda x: numbers_to_tokens(str(x)))
    tokens_dev_df.to_csv(os.path.join(tokens_data_path, "dev.csv"), index=False)
    tokens_train_df.to_csv(os.path.join(tokens_data_path, "train.csv"), index=False)

    os.symlink(
        os.path.abspath(os.path.join(numbers_data_path, "train")),
        os.path.abspath(os.path.join(tokens_data_path, "train")),
    )
    os.symlink(
        os.path.abspath(os.path.join(numbers_data_path, "dev")),
        os.path.abspath(os.path.join(tokens_data_path, "dev")),
    )


def numbers_to_words(text: str) -> str:
    return num2words(text, lang="ru")


def numbers_to_tokens(text: str) -> list:
    text = text.strip()
    thousands = text[:-3]
    remainder = text[-3:]

    tokens = []
    for place, digit in enumerate(thousands):
        if digit != "0":
            value = int(digit) * (10 ** (len(thousands) - 1 - place))
            tokens.append(f"<{value}>")

    tokens.append("|")

    for place, digit in enumerate(remainder):
        if digit != "0":
            value = int(digit) * (10 ** (2 - place))
            tokens.append(f"<{value}>")
    return tokens


if __name__ == "__main__":
    main()
