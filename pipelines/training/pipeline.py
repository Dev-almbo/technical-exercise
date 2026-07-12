from kedro.pipeline import Pipeline, node
import logging

from src.train.trainer import HuggingFaceTrainer
from src.train.preprocessing import (
    load_json,
    split_reviews_and_assign_rating,
    preprocess_text,
    encode_rating_into_sentiment_labels,
)

from src.exceptions import ConfigurationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_data_node(input_data: dict):
    """Node wrapper on top of data loading module.

    `input_data` is the Kedro `params:input_data` mapping containing
    `data_dir` and `file_name`.
    """
    return load_json(data_dir=input_data["data_dir"], file_name=input_data["file_name"])


def _split_reviews_and_assign_rating_node(input_df):
    """Node wrapper on top of preprocessing steps:
    Reviews need to be separated into individual rows and assigned a rating
    """
    return split_reviews_and_assign_rating(input_df)


def _clean_text_node(input_df):
    """Node wrapper on top of preprocessing steps:
    Clean text string from special characters, punctuation
    """
    return preprocess_text(input_df)


def _encode_rating_into_sentiment_labels_node(input_df):
    """Node wrapper on top of preprocessing steps:
    Rating is being encoded into sentiment labels
    """
    return encode_rating_into_sentiment_labels(input_df)


def _train_node(labeled_reviews, training: dict):
    """Node wrapper that delegates to the project Trainer wrapper.

    This node expects:
    - `labeled_reviews`: the preprocessed DataFrame produced by the upstream
      pipeline nodes (model-agnostic preprocessing).
    - `training`: a Kedro `params:training` config
    """
    if not isinstance(training, dict):
        raise ConfigurationError(
            "Expected training parameter to be a dict; configure via conf/base/parameters.yml"
        )

    trainer = HuggingFaceTrainer()
    trainer.fit(data=labeled_reviews, config=training)


def create_pipeline(**kwargs) -> Pipeline:
    """Kedro Pipeline Wrapper for Training"""
    return Pipeline(
        [
            # Load raw data from params:input_data (expects dict with data_dir & file_name)
            node(
                func=_load_data_node,
                inputs="params:input_data",
                outputs="raw_reviews",
                name="Load_data_node",
            ),
            node(
                func=_split_reviews_and_assign_rating_node,
                inputs="raw_reviews",
                outputs="split_reviews",
                name="Split_reviews_and_assign_rating_node",
            ),
            node(
                func=_clean_text_node,
                inputs="split_reviews",
                outputs="clean_reviews",
                name="Clean_text_node",
            ),
            node(
                func=_encode_rating_into_sentiment_labels_node,
                inputs="clean_reviews",
                outputs="labeled_reviews",
                name="Encode_rating_into_sentiment_labels_node",
            ),
            node(
                func=_train_node,
                inputs=["labeled_reviews", "params:training"],
                outputs=None,
                name="Training_node",
            ),
        ]
    )
