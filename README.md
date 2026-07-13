# Technical Exercise

## Getting Started

1. Install the project:

```bash
uv sync --all-extras
```

## Training

Configuration

- The pipeline reads its settings from `conf/base/parameters.yml` under the `training` key.
- Edit that file to point to your raw data and choose `output_dir`.

Run via Kedro CLI

- If this repository is a Kedro project, run:

```bash
cd ../technical-exercise
uv run kedro run --pipelines training
```

Disclaimer: Locally, the training is currently not very performant.

## Running the API locally

```bash
cd ../technical-exercise
uv run uvicorn src.api.main:app
```
