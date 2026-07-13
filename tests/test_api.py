import asyncio

import pytest
from fastapi import HTTPException

from src.api import main


@pytest.fixture
def classifier():
    original = main.classifier
    yield
    main.classifier = original


def _request(text: str) -> main.PredictRequest:
    return main.PredictRequest(text=text)


def test_predict_raises_when_model_not_loaded(classifier):
    main.classifier = None
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.predict(_request("some text")))
    assert exc.value.status_code == 503
    assert exc.value.detail == "Model is not loaded"


def test_predict_returns_sentiment(classifier):
    received = {}

    def capture(text):
        received["value"] = text
        return [{"label": "positive", "score": 0.99}]

    main.classifier = capture

    result = asyncio.run(main.predict(_request("great book")))

    assert isinstance(received["value"], str)
    assert received["value"] == "great book"
    assert result.text == "great book"
    assert result.sentiment == "positive"


def test_predict_raises_after_exhausting_retries(classifier):
    calls = {"n": 0}

    def always_fail(text):
        assert isinstance(text, str)
        calls["n"] += 1
        raise RuntimeError("boom")

    main.classifier = always_fail

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.predict(_request("anything")))
    assert exc.value.status_code == 503
    assert "Prediction failed after" in str(exc.value.detail)
    assert calls["n"] == main.MAX_RETRIES
