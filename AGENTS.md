# AGENTS.md — llm-d-latency-predictor

Machine-readable onboarding for AI coding agents working in this repository.

## Commands

```bash
make install        # pip install -e . + requirements (editable)
make test           # pytest (tests/ — integration tests skip if servers unreachable)
make lint           # ruff check (config in pyproject.toml, line-length 120, py311)
make fmt            # ruff format
make image-build    # build prediction, training, and test container images
pre-commit run --all-files   # hooks incl. typos (_typos.toml)

# Run servers locally
uvicorn llm_d_latency_predictor.prediction_server:app --port 8001
uvicorn llm_d_latency_predictor.training_server:app --port 8000
```

## What this project is

Latency prediction service for the llm-d ecosystem. Predicts TTFT and TPOT per
(request, vLLM pod) pair so the Gateway API Inference Extension Endpoint Picker
(EPP) can do latency-based and SLO-aware routing. Position in the stack:

```text
User -> Envoy (ext_proc) -> EPP [flow control -> scheduler plugins] -> vLLM pod
                              |  predictions        ^ observed samples
                              v                     |
              Prediction server :8001 <-sync- Training server :8000   (THIS REPO)
```

Full architecture, diagrams, plugin catalog, and config reference:
**docs/architecture.md** (read it before answering architecture questions).

## File map

- `src/llm_d_latency_predictor/prediction_server.py` — FastAPI prediction server.
  `ModelSyncer` (background HTTP model sync), `LightweightPredictor` (inference,
  `/predict`, `/predict/bulk`, `/predict/bulk/strict` fast path).
- `src/llm_d_latency_predictor/training_server.py` — FastAPI training server.
  `LatencyPredictor` (stratified buckets, retraining loop, quantile metrics),
  model publishing endpoints (`/model/{name}/info|download`).
- `tests/test_dual_server_client.py` — integration/load-test client for both servers.
- `deploy/` — Kubernetes manifests (dual-server topology; ports 8000 training, 8001 prediction).
- `Dockerfile-prediction`, `Dockerfile-training`, `Dockerfile-test` — images.
- `.cursor/rules/` — scoped agent rules; `.cursor/agents/` — subagents
  (repo-guide, bug-hunter, perf-optimizer, pr-reviewer).

## Invariants (do not break)

1. **Feature-engineering parity**: `_prepare_features_with_interaction()` and
   `_prepare_features_for_ensemble()` are duplicated in BOTH servers and must
   produce identical columns in identical order. Change them together, never singly.
2. **Public API**: `/predict/bulk/strict`, `/add_training_data_bulk`, and
   `/model/{name}/info|download` schemas are consumed by the upstream EPP
   (`predicted-latency-producer`). Breaking changes need upstream coordination.
3. **Hot path**: `/predict/bulk/strict` sits on the EPP scheduling critical path.
   Keep the numpy fast path (no per-request `.dict()`, dict-of-arrays DataFrames,
   ORJSON responses); never do heavy work while holding the predictor lock —
   snapshot model references under the lock, run inference outside it.
4. **Model artifacts**: joblib files exchanged between servers. Renaming model
   names/paths or changing `QueueGatedModel` shape breaks the sync protocol.

## Code style

- Python 3.11+ (`StrEnum`, `X | None` unions). Ruff: E, W, F, I, UP; line length 120.
- Pydantic v1-style API is in use (`.dict()`, `min_items`); stay consistent.
- Config = env vars with defaults in `PredictSettings` / `Settings` classes;
  document new vars in docs/architecture.md configuration reference.
- Threading: daemon threads + `threading.Event` for shutdown; `RLock` in the
  prediction server, `Lock` in the training server.

## Testing

- `pytest` via `make test`. Integration tests call live servers via
  `PREDICTION_SERVER_URL` / `TRAINING_SERVER_URL` env vars and skip when unreachable.
- When changing feature engineering or endpoints, exercise both single and bulk
  prediction paths, and both ensemble regimes (queued / noqueue).

## Boundaries

- **Always**: run `make lint` and `make test` before finishing; keep the two
  servers' feature code in sync; sign off commits (DCO, see PR_SIGNOFF.md).
- **Ask first (never without explicit user permission)**: any `git commit`,
  `git push`, opening/updating a PR, filing a GitHub issue, or posting comments.
  Draft the content and present it to the user instead.
- **Never**: commit secrets or credentials; silently change model file formats
  or HTTP schemas consumed by the EPP; remove the fallback import guards for
  xgboost/lightgbm/orjson.
