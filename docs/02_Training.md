# 02_Training

Training and Preprocessing is wrapped into a Kedro pipeline.

## Why Kedro?

- It provides a lightweight training pipeline framework that is cloud-agnostic or runs locally for development.
- It enforces best practices on training configurations, and structures to make code more testable.
- Kedro Pipleines can be integrated with Prometheus using kedro hooks for more advanced training pipeline monitoring to set up alerting for training failures or data quality issues during training.

## Model Choice

- Pre-trained models and their tokenizers are a good choice for when not enough data is available, and language models would need to learn language from scratch for this task of text classification.
- This pre-trained model from huggingface can easily be finetuned with the Amazon Reviews, and hugging face provides custom tokenization and utiliti.

## Trainer Code Design

- The Kedro pipeline training pipeline should use code that is model agnostic, so the pipeline is only needed to be defined once.

- There is an abstract class called "Trainer" that implements which training functionalities are used in the Kedro pipeline. It should be extenable to use other types of models or model providers to facilitate experimenting with other model architectures.

- It takes all needed parameters from a centralized config that can be adpated for experiments.

- The HuggingFace Trainer is an example implementation of how training could look like with a pretrained model.

- There is model-agnostic data preprocessing steps that are contained in the kedro pipeline framework, since they should be shared. Otherwise, there is specific preprocessing for the hugging face trainer.

- The model is saved locally for local deployment. For use in a cloud environment, the trainer must be extended to store all model artifacts in a model registry, e.g. MlFlow or Vertex AI model registry.

## Future Considerations

- Integrate Kedro Pipeline with a project on a cloud environment for GPU-enabled finetuning.
- A training docker image is needed to move forward with this.
