# 03_Deployment

## Local Deployment

- Model are currently stored locally in an outputs directory after training, and fetched through the API from a path.
- Model is wrapped up in an asynchronous endpoint for concurrent processing of requests, so that one requests is not blocking the processing of other requests at the same time (as synchronous processing does) and the production system remains responsive reducing request latency
- More efficient use of provided resources as one task does not block others.
- For proper productionisation, there would be a need for more advanced error handling, and request input safety to not cause high amounts of errors production.

## Deployment on Cloud

### 1. Depoy API in a containerized application on a Kubernetes cluster.

- higher control on the model deployment in terms or latency and throughput,
- can also set up fine-grained resource management depending on ML needs, horizontal scaling, and strong self-healing capabilities for high fault tolerance.
- this API could very simply be containerized and deployed with kubernetes configurations, even though the up front cost of deploying of a Kubernetes cluster may be high.
  Note: The API here needs a proper docker image for containerized deployments, the model needs to be stored somewhere where the API can access the model to download, e.g. some blob storage or a hosted Model registry like MlFlow, the deployment pipeline should manage all K8s deployments throughout the environments.

### 2. Deploying to a Managed Machine Learning Platform e.g. Vertex AI, Amazon Sagemaker

- faster time-to-market and higher standardization between projects if infrastructure is already set up during ml platform
- integrated suit of tools simplifying ML workflows
- may cause vendor lock in when everything is built for one provider (and consequently high transitioning cost).
  Note: this project would need to be transitioned to use a model registry that has access to the platform, and CD pipelines should exist to handle automatic deployment from the platform's registry to serving endpoints to reduce human errors.
