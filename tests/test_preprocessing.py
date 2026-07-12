import json
from pathlib import Path
import pandas as pd

from src.train.preprocessing import (
    load_json,
    split_reviews_and_assign_rating,
    encode_rating_into_sentiment_labels,
    preprocess_text,
)


def test_split_and_encode_and_preprocess():
    raw = [
        {"rating": 5.0, "text": "Great product!"},
        {"rating": 2.0, "text": "Not good"},
        {"rating": 3.0, "text": "Okay"},
    ]
    df = pd.DataFrame(raw)
    df = split_reviews_and_assign_rating(df)
    assert "review_text" in df.columns and "rating" in df.columns

    df = preprocess_text(df)
    assert df.loc[0, "review_text"] == "great product!"

    df = encode_rating_into_sentiment_labels(df)
    assert df.loc[0, "sentiment"] == "positive"
    assert df.loc[1, "sentiment"] == "negative"
    assert df.loc[2, "sentiment"] == "neutral"


def test_load_json_reads_lines(tmp_path: Path):
    data = [{"rating": 1.0, "text": "A"}, {"rating": 2.0, "text": "B"}]
    p = tmp_path / "sample.jsonl"
    with open(p, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    df = load_json(str(tmp_path), p.name)
    assert len(df) == 2
    assert df.loc[0, "text"] == "A"
