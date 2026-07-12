from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import logging
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
import yaml
import pickle

from src.exceptions import ModelError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# fetch model path from output dir specified in config
MODEL_PATH = yaml.load(open("conf/base/parameters.yml", "r"))["output_dir"]


class ModelResponse(BaseModel):
    text: str
    sentiment: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # load the model at start up so the first burst of requests does not trigger cold-load spikes
    logger.info("Loading model at startup:")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> RedirectResponse:
    logger.info("Root endpoint called")
    return RedirectResponse(url="/docs")


@app.get("/predict", response_model=ModelResponse)
async def predict(text: str) -> ModelResponse:
    try:
        logger.info(f"Predict endpoint called with text: {text}")
        # Here you would add your prediction logic
        return ModelResponse(text=text, sentiment="positive")
    except Exception as e:
        logger.warning(f"Error occurred: {e}, Trying again.")
        # retry after some time
        await asyncio.sleep(2)  # do not wait too long to retry to reduce latency
        return await model.predict(text=text)
        # what if the retry fails
        raise ModelError(f"Prediction failed after retry: {e}")
