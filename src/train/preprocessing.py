import json
from pathlib import Path
import logging
import pandas as pd
import re
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def _build_path(data_dir: Path, file_name: str) -> Path:
    return data_dir / Path(file_name)


def load_json(data_dir: str, file_name: str) -> pd.DataFrame:
    """Load JSON lines from data/* dir"""

    reviews = []
    path = _build_path(Path(data_dir), file_name)
    logger.info(f"Loading data from {path}")
    with open(path, "r") as file:
        for line in file:
            review = json.loads(line)
            reviews.append(review)
    logger.info(f"Loaded {len(reviews)} reviews")
    return pd.DataFrame(reviews)


def split_reviews_and_assign_rating(data: pd.DataFrame) -> pd.DataFrame:
    """Explode each review into one row per sentence, keeping the review's rating.

    Per the task spec, a multi-sentence review yields one training example per
    sentence, each inheriting the review's star rating.
    """
    data = data.copy()
    data["rating"] = data["rating"].astype(float).round().astype(int)

    # split on sentence-ending punctuation, then explode to one row per sentence
    data["review_text"] = data["text"].str.split(r"(?<=[.!?])\s+")
    data = data.explode("review_text", ignore_index=True)

    # tidy up and drop empty sentences
    data["review_text"] = data["review_text"].str.strip()
    data = data[data["review_text"].str.len() > 0].reset_index(drop=True)
    return data


def encode_rating_into_sentiment_labels(data):
    """Encode 1-2 stars rating into negaitve, 3 starts into neutral, 4-5 positive."""

    conditions = [
        (data["rating"] >= 1) & (data["rating"] <= 2),
        (data["rating"] == 3),
        (data["rating"] >= 4) & (data["rating"] <= 5),
    ]
    choices = ["negative", "neutral", "positive"]
    data["sentiment"] = pd.Series(
        np.select(conditions, choices, default="unknown"), index=data.index
    )
    return data


def _clean_text(text: str) -> str:
    """Clean the review text by removing unwanted characters."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_text(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the review text."""
    data["review_text"] = data["review_text"].apply(_clean_text)
    return data


def tokenize(data):
    # TODO: if any other model than huggingface is selected, a custom tokenizer must be implemented
    pass
