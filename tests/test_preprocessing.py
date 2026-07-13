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
        {"rating": 5.0, "text": "Great product! I loved it."},
        {"rating": 2.0, "text": "Not good"},
        {"rating": 3.0, "text": "Okay"},
    ]
    df = pd.DataFrame(raw)
    df = split_reviews_and_assign_rating(df)
    assert "review_text" in df.columns and "rating" in df.columns

    # the first review has two sentences -> two rows, both rated 5
    assert len(df) == 4
    five_star = df[df["rating"] == 5]
    assert set(five_star["review_text"]) == {"Great product!", "I loved it."}

    df = preprocess_text(df)
    df = encode_rating_into_sentiment_labels(df)
    sentiments = dict(zip(df["review_text"], df["sentiment"]))
    assert sentiments["great product!"] == "positive"
    assert sentiments["not good"] == "negative"
    assert sentiments["okay"] == "neutral"


def test_load_json_reads_lines(tmp_path: Path):
    data = [{"rating": 1.0, "text": "A"}, {"rating": 2.0, "text": "B"}]
    p = tmp_path / "sample.jsonl"
    with open(p, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    df = load_json(str(tmp_path), p.name)
    assert len(df) == 2
    assert df.loc[0, "text"] == "A"
