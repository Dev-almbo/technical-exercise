import logging
from typing import Dict, Any, Optional, Tuple
from abc import ABC
import pandas as pd
import evaluate
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer as HfTrainer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer(ABC):
    """Abstract Trainer base class for the project."""

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError()


class HuggingFaceTrainer(Trainer):
    """Trainer implementation that wraps Hugging Face Transformers Trainer."""

    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self._hf_trainer: Optional[HfTrainer] = None

    @staticmethod
    def _compute_metrics(pred) -> Dict[str, Optional[float]]:
        """Compute metrics for model evaluation.

        Passed directly to the HF Trainer. `pred` is an EvalPrediction-like
        tuple of (predictions, label_ids) where predictions are logits.
        """
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy.compute(predictions=preds, references=labels) or {}  # ty: ignore[missing-argument]
        f1m = f1.compute(predictions=preds, references=labels, average="macro") or {}  # ty: ignore[missing-argument]
        return {"accuracy": acc.get("accuracy"), "f1_macro": f1m.get("f1")}

    def preprocess(
        self,
        data: pd.DataFrame,
        config: Optional[dict] = None,
        model_name: str = "distilbert-base-uncased",
    ) -> Tuple[
        Dataset, Optional[Dataset], Dataset, Any, Dict[str, int], Dict[int, str]
    ]:
        """Model-specific data preparation for Hugging Face training.

        This method onlyperforms the HF-specific steps: label-map construction,
        train/val/test splitting and tokenization.

        Keeping this on the trainer makes the training node switchable: swap the
        trainer implementation and its model-specific prep travels with it.

        Args:
            data: Preprocessed DataFrame containing the text and label columns.
            config: Optional config dict (Kedro `params:training`) controlling
                column names and split sizes.
            model_name: Tokenizer/model checkpoint name.

        Returns:
            train_ds, val_ds, test_ds, tokenizer, label2id, id2label
        """

        cfg = config or {}
        text_col = cfg.get("text_column", "review_text")
        label_col = cfg.get("label_column", "sentiment")
        test_size = cfg.get("test_size", 0.1)
        val_size = cfg.get("val_size", 0.0)

        df = data

        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"Expected columns '{text_col}' and '{label_col}' in preprocessed data"
            )

        def _build_label_maps(labels: pd.Series) -> Dict[str, int]:
            unique = sorted(labels.dropna().unique())
            return {label: i for i, label in enumerate(unique)}

        label2id = _build_label_maps(df[label_col])
        id2label = {v: k for k, v in label2id.items()}

        df = df[[text_col, label_col]].reset_index(drop=True)
        df["label"] = df[label_col].map(label2id)

        ds = Dataset.from_pandas(df[[text_col, "label"]])

        # split into train / val / test
        if val_size and val_size > 0:
            split_total = test_size + val_size
            split = ds.train_test_split(test_size=split_total, seed=42)
            train_ds = split["train"]
            rest = split["test"]
            rest_split = rest.train_test_split(
                test_size=val_size / split_total, seed=42
            )
            val_ds = rest_split["train"]
            test_ds = rest_split["test"]
        else:
            split = ds.train_test_split(test_size=test_size, seed=42)
            train_ds = split["train"]
            val_ds = None
            test_ds = split["test"]

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def _tokenize(x):
            return tokenizer(x[text_col], truncation=True)  # ty: ignore[call-non-callable]

        train_ds = train_ds.map(_tokenize, batched=False)
        test_ds = test_ds.map(_tokenize, batched=False)
        if val_ds is not None:
            val_ds = val_ds.map(_tokenize, batched=False)

        keep_cols = ["input_ids", "attention_mask", "label"]
        train_ds = train_ds.remove_columns(
            [c for c in train_ds.column_names if c not in keep_cols]
        )
        test_ds = test_ds.remove_columns(
            [c for c in test_ds.column_names if c not in keep_cols]
        )
        if val_ds is not None:
            val_ds = val_ds.remove_columns(
                [c for c in val_ds.column_names if c not in keep_cols]
            )

        # keep the tokenizer available on the instance for downstream use
        self.tokenizer = tokenizer

        return train_ds, val_ds, test_ds, tokenizer, label2id, id2label

    def fit(
        self,
        data: pd.DataFrame,
        config: Optional[dict] = None,
    ) -> None:
        """Run model-specific preprocessing then train.

        This is the single entry point the Kedro training node calls. It keeps
        the model-specific prep and training coupled to the trainer so the whole
        step can be switched by swapping the trainer implementation.
        """
        cfg = config or {}
        model_name = cfg.get("model_name_or_path") or "distilbert-base-uncased"
        output_dir = cfg.get("output_dir") or "outputs/model"
        training_args_cfg = cfg.get("training_args", {})

        train_ds, val_ds, test_ds, tokenizer, label2id, id2label = self.preprocess(
            data=data, config=cfg, model_name=model_name
        )

        self.train(
            train_dataset=train_ds,
            eval_dataset=val_ds if val_ds is not None else test_ds,
            tokenizer=tokenizer,
            model_name=model_name,
            label2id=label2id,
            id2label=id2label,
            output_dir=output_dir,
            training_args_cfg=training_args_cfg,
        )

    def train(
        self,
        train_dataset,
        eval_dataset,
        tokenizer,
        model_name: str,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        output_dir: str,
        training_args_cfg: Dict[str, Any],
    ) -> None:
        """Train given already-prepared HF `datasets.Dataset` objects."""

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=training_args_cfg.get("eval_strategy", "epoch"),
            per_device_train_batch_size=training_args_cfg.get(
                "per_device_train_batch_size", 8
            ),
            per_device_eval_batch_size=training_args_cfg.get(
                "per_device_eval_batch_size", 8
            ),
            num_train_epochs=training_args_cfg.get("num_train_epochs", 3),
            save_strategy=training_args_cfg.get("save_strategy", "epoch"),
            logging_steps=training_args_cfg.get("logging_steps", 50),
            load_best_model_at_end=training_args_cfg.get(
                "load_best_model_at_end", True
            ),
            metric_for_best_model=training_args_cfg.get(
                "metric_for_best_model", "f1_macro"
            ),
            fp16=training_args_cfg.get("fp16", False),
        )

        # instantiate HF Trainer and keep a reference on self for evaluate/predict
        self._hf_trainer = HfTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting training")
        self._hf_trainer.train()
        logger.info(f"Saving model to {output_dir}")
        self._hf_trainer.save_model(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    def evaluate(self, *args, **kwargs):
        if self._hf_trainer is None:
            raise RuntimeError("Trainer has not been initialized; call train() first")
        return self._hf_trainer.evaluate(*args, **kwargs)

    def predict(self, dataset, *args, **kwargs):
        if self._hf_trainer is None:
            raise RuntimeError("Trainer has not been initialized; call train() first")
        return self._hf_trainer.predict(dataset, *args, **kwargs)
