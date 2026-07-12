# 01_General Project Architecture

- Training is wrapped in the Kedro Framework (see 02_Training).
- Deployment is through FastAPI locally (see 03_Depoyment).

# Structure

- .github: contains CI/CD pipeline configurations for github (Disclaimer: Examplary!)
- data: contains Amazon Reviews
- src: contains source code for preprocessing, training, API deployment
- notebooks: purely for experimentation, debugging
- pipelines: contains kedro pipeline for training, can be extended with other pipelines i.e. continuous evaluation
- test: tests for unit testing of the preprocessing (uses pytest), can be extended for integration testing, performance testing for the API

# Other considerations

- This project is extendable with prometheus monitoring on the API, and on the training pipelines for full-blown Monitoring solution necessary for resilient productionization.+
- This project is currently missing a Model Registry or Experiment tracking framework but could easily be extended with it.
