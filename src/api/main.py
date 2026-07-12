from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import logging
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any, Optional
import yaml

from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# fetch model path from the output dir specified in the training config
with open("conf/base/parameters.yml", "r") as _config_file:
    _params = yaml.safe_load(_config_file)
MODEL_PATH: str = _params["training"]["output_dir"]

MAX_RETRIES: int = 2
RETRY_DELAY_SECONDS: float = 2.0


class ModelResponse(BaseModel):
    text: str
    sentiment: str


classifier: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # load the model at start up so the first burst of requests does not trigger
    # cold-load spikes
    global classifier
    logger.info("Loading model at startup from %s", MODEL_PATH)
    try:
        # TODO: check that this is fine and below 300ms latency
        # TODO: batch processing
        classifier = pipeline(
            "text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH
        )
    except Exception as e:  # pragma: no cover - startup diagnostics
        logger.error("Could not load model at startup: %s", e)
        raise HTTPException(status_code=404, detail="Could not load model at startup")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> RedirectResponse:
    logger.info("Root endpoint called")
    return RedirectResponse(url="/docs")


@app.get("/predict", response_model=ModelResponse)
async def predict(text: str) -> ModelResponse:
    logger.info("Predict endpoint called with text: %s", text)

    last_error: Optional[Exception] = None
    # attempt the prediction, retrying a specified number of times on transient errors
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if classifier is None:
                raise HTTPException(status_code=503, detail="Model is not loaded")
            result = classifier(text)
            label = result[0]["label"]
            return ModelResponse(text=text, sentiment=str(label))
        except Exception as e:
            last_error = e
            logger.warning(
                "Prediction failed on attempt %d/%d: %s", attempt, MAX_RETRIES, e
            )
            if attempt < MAX_RETRIES:
                # brief backoff before retrying; kept short to limit request latency
                await asyncio.sleep(RETRY_DELAY_SECONDS)

    raise HTTPException(
        status_code=503,
        detail=f"Prediction failed after {MAX_RETRIES} attempts: {last_error}",
    )
