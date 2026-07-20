# llm-d-latency-predictor

[![CI](https://github.com/llm-d/llm-d-latency-predictor/actions/workflows/ci-pr-checks.yaml/badge.svg)](https://github.com/llm-d/llm-d-latency-predictor/actions/workflows/ci-pr-checks.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Latency prediction service for the llm-d ecosystem.**

## Overview

`llm-d-latency-predictor` provides continuous, online latency prediction for LLM inference scheduling. It predicts **TTFT** (time to first token) and **TPOT** (time per output token) for a request on each candidate vLLM pod, given the pod's live state (KV-cache utilization, queue depth, prefix-cache match score) and the request's features.

The predictions power **latency-based and SLO-aware routing** in the [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/guides/latency-based-predictor/) / [llm-d](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms): the Endpoint Picker (EPP) calls the prediction server for every candidate pod at scheduling time, routes the request to the pod with the best predicted outcome (or best SLO headroom), and feeds the observed latencies back to the training server after the response — so the models continuously learn from live traffic with no offline training step.

The project ships two FastAPI services:

- **Training server** — ingests observed latency samples, retrains XGBoost/LightGBM/Bayesian-Ridge quantile models on a stratified sliding window, and publishes model files over HTTP.
- **Prediction server** — syncs the latest models and serves bulk TTFT/TPOT predictions on the EPP's scheduling hot path.

## Prerequisites

- Python 3.11+
- Docker (for container builds)
- [pre-commit](https://pre-commit.com/) (for local development)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/llm-d/llm-d-latency-predictor.git
cd llm-d-latency-predictor

# Install pre-commit hooks
pre-commit install

# Install Python dependencies
make install

# Run tests
make test

# Run linters
make lint
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, coding standards, and how to submit changes.

### Common Commands

```bash
make help           # Show all available targets
make install        # Install Python dependencies
make test           # Run Python tests
make lint           # Run Python linter (ruff)
make fmt            # Format Python code
make image-build    # Build container images (prediction, training, test)
make pre-commit     # Run pre-commit hooks
```

### Layout

```
src/llm_d_latency_predictor/
  prediction_server.py    # FastAPI prediction server (serves latency predictions)
  training_server.py      # FastAPI training server (trains models from request traces)
tests/
  test_dual_server_client.py  # integration / load-test client exercising both servers
deploy/                   # Kubernetes manifests and kustomization
Dockerfile-prediction     # Image for the prediction server
Dockerfile-training       # Image for the training server
Dockerfile-test           # Image that runs the test client as a Job
build-deploy.sh           # Helper script for building images and deploying to GKE
```

The code is packaged as `llm_d_latency_predictor` — after `make install` (editable) or
`pip install .`, the servers can be run as Python modules:

```bash
uvicorn llm_d_latency_predictor.prediction_server:app --port 8001
uvicorn llm_d_latency_predictor.training_server:app --port 8000
```

## Architecture

See **[docs/architecture.md](docs/architecture.md)** for the full end-to-end architecture: the request flow from the user through Envoy, the EPP's flow-control layer and scheduler plugin pipeline, the latency predictor, and the vLLM pods; the continuous training feedback loop; batch inference internals; the llm-d/EPP plugin catalog; and deployment topologies.

At a glance:

```text
User -> Envoy (ext_proc) -> EPP [flow control -> predicted-latency-producer -> latency-scorer -> picker]
                                      |                                  ^
                                      v                                  |
                        Prediction server :8001  <-- model sync --  Training server :8000
                                                                          ^
User <- Envoy <- chosen vLLM pod (continuous batching)                    |
                     `----- observed TTFT/TPOT samples ------------------'
```

## Configuration

Both servers are configured entirely through environment variables (model type, quantile, retraining cadence, sync interval, ensemble mode, etc.). The complete reference lives in [docs/architecture.md](docs/architecture.md#configuration-reference).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

All commits must be signed off (DCO). See [PR_SIGNOFF.md](PR_SIGNOFF.md) for instructions.

## Security

To report a security vulnerability, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.
