# AGENTS.md

## Cursor Cloud specific instructions

This repo is a Python project (`llm-d-latency-predictor`) with a **dual FastAPI server** architecture. There is no frontend; both services expose HTTP/JSON APIs (with FastAPI Swagger UI at `/docs`).

### Services

| Service | Module | Default port | Role |
| --- | --- | --- | --- |
| Training server | `llm_d_latency_predictor.training_server:app` | 8000 | Ingests training data, periodically retrains TTFT/TPOT models, serves model files over HTTP |
| Prediction server | `llm_d_latency_predictor.prediction_server:app` | 8001 | Polls the training server, downloads models, serves latency predictions |

The prediction server syncs models **from** the training server, so start the training server first.

### Environment / setup

- Dependencies are installed by the update script (`make install` → editable package + `ruff pytest pytest-asyncio`). System Python 3.12 is used (project requires >=3.11); there is no virtualenv.
- Console scripts (`ruff`, `pytest`, `uvicorn`) install to `~/.local/bin`, which is added to `PATH` via `~/.bashrc`. If a non-login shell can't find them, use `~/.local/bin/<tool>` or `python3 -m <tool>`.

### Lint / test / build (standard commands)

- Lint: `make lint` (`ruff check .` + `ruff format --check .`). This is the only code gate CI runs (see `.github/workflows/ci-pr-checks.yaml`); CI also builds the three Dockerfiles but does **not** run pytest.
- Format: `make fmt`.
- Tests: `pytest tests/` — but see the important caveat below. Run commands are also in `README.md` and the `Makefile`.

### Running the servers (non-obvious caveats)

- The prediction server's module-level defaults are tuned for Kubernetes, not local dev. You **must** override two things or startup/sync fails:
  - `TRAINING_SERVER_URL=http://localhost:8000` (default is the k8s service name `http://training-service:8000`).
  - The model paths default to `/local_models/...` (not writable by a non-root user; `ModelSyncer()` does `os.makedirs` at import time). Redirect all of them to a writable dir, e.g. `/tmp/local_models/`: `LOCAL_TTFT_MODEL_PATH`, `LOCAL_TPOT_MODEL_PATH`, `LOCAL_TTFT_SCALER_PATH`, `LOCAL_TPOT_SCALER_PATH`, `LOCAL_TTFT_GATED_MODEL_PATH`, `LOCAL_TPOT_GATED_MODEL_PATH`.
- The training server's model paths default to `/tmp/models/` (writable) so it needs no overrides for local dev.
- On startup the training server creates a "default" stub model that predicts a constant `10.0` ms, so `/readyz` is immediately `ready` and predictions return `10.0` until a real model is trained.

### `tests/test_dual_server_client.py` (integration/e2e suite)

- This is an **e2e suite that runs against already-running servers**, not standalone unit tests. Point it at the local servers with `PREDICTION_SERVER_URL=http://localhost:8001` and `TRAINING_SERVER_URL=http://localhost:8000` (defaults are k8s placeholder IPs). In production it runs as a k8s Job built from `Dockerfile-test`.
- Real-model-quality tests (`*_learns_distribution`, `*_learns_equation`) submit thousands of samples then wait for retraining. The training loop only retrains every `LATENCY_RETRAINING_INTERVAL_SEC` (**default 1800s = 30 min**). For these to exercise a real (non-stub) model within test timeouts, start the training server with a short interval and lower sample threshold, e.g. `LATENCY_RETRAINING_INTERVAL_SEC=15 LATENCY_MIN_SAMPLES_FOR_RETRAIN=200`.
- `test_dual_server_quantile_regression_learns_distribution` is a **statistically flaky** assertion (requires ≥70% predictions within ±15% and coverage within ±0.05 of the 0.9 quantile). It can fail by a small margin on a freshly-trained local model even when everything works; it is not gated by CI. The other 37 tests pass and 2 skip by design (objective/model-type guards).
