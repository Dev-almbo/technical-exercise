"""Project pipelines registry."""

from typing import Dict

from kedro.pipeline import Pipeline

from pipelines.training.pipeline import create_pipeline as create_training_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects. The
        ``__default__`` pipeline runs when no ``--pipeline`` flag is provided.
    """
    training_pipeline = create_training_pipeline()

    return {
        "training": training_pipeline,
        "__default__": training_pipeline,
    }
