# Deployment Guide

## 1. Docker (single host)

```bash
cp .env.example .env   # fill in GEMINI_API_KEY / GOOGLE_BOOKS_API_KEY
docker compose up --build -d
```

This brings up:
- `api` — the FastAPI service on `:8000`
- `redis` — cache backend on `:6379`
- `prometheus` — metrics scraping on `:9090`

Verify:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Building the image standalone

```bash
docker build -t book-recognition-system:latest .
docker run -p 8000:8000 --env-file .env book-recognition-system:latest
```

The image is a two-stage build (builder installs deps into `--user` site,
final stage copies only the installed packages + app code), runs as a
non-root `appuser`, and ships with a `HEALTHCHECK`.

## 2. Kubernetes

Reference manifests are in `deploy/k8s-deployment.yaml`:

- `Deployment` (3 replicas by default), with `readinessProbe` on `/ready`
  and `livenessProbe` on `/health`
- `Service` (ClusterIP, port 80 → 8000)
- `HorizontalPodAutoscaler` (CPU-based, 3–10 replicas)

Secrets and non-secret config are expected via:

```bash
kubectl create secret generic book-recognition-secrets \
  --from-literal=GEMINI_API_KEY=... \
  --from-literal=GOOGLE_BOOKS_API_KEY=...

kubectl create configmap book-recognition-config \
  --from-env-file=.env.example   # replace with your non-secret overrides
```

```bash
kubectl apply -f deploy/k8s-deployment.yaml
```

For production, replace the `emptyDir` volume with a `PersistentVolumeClaim`
if you need `data/checkpoints` (SQLite checkpointing) to survive pod
restarts, or switch `GRAPH_CHECKPOINT_BACKEND` off SQLite entirely and rely
on a managed Postgres/Redis-backed checkpointer.

## 3. Environment configuration

All configuration is environment-driven (`app/config/settings.py`). Key
production-relevant variables:

| Variable | Purpose | Production guidance |
|---|---|---|
| `CACHE_BACKEND` | `redis` or `memory` | Use `redis` behind more than one replica |
| `REDIS_URL` | Redis connection string | Point at a managed Redis (ElastiCache, Memorystore, etc.) |
| `GRAPH_CHECKPOINT_BACKEND` | `sqlite` or `memory` | `sqlite` on a persistent volume, or swap for a distributed checkpointer |
| `YOLO_DEVICE` / `SUPER_RES_DEVICE` | `cpu`/`cuda`/`mps` | Use GPU nodes for throughput; falls back gracefully on CPU |
| `RATE_LIMIT_PER_MINUTE` | Per-client throttling | Tune to your quota/cost budget |
| `API_KEY_ENABLED` / `API_KEYS` | Auth | Always `true` in production; rotate keys via secret manager |
| `OTEL_ENABLED` / `OTEL_EXPORTER_OTLP_ENDPOINT` | Tracing | Point at your collector (Tempo, Honeycomb, etc.) |

## 4. Model weights

- **YOLO**: place a `.pt` weights file at `YOLO_MODEL_PATH` (default
  `models/yolov8n.pt`). Without it, the tool logs a warning and falls back
  to a heuristic full-image detector — functional for a single-book photo,
  but you'll want a fine-tuned book-spine/cover model for multi-book shelf
  photos in production.
- **Real-ESRGAN**: place `RealESRGAN_x4plus.pth` under `weights/`. Without
  it, `SuperResolutionTool` falls back to a Lanczos upscale.

Both fallbacks are intentional so the service is always runnable, but
production accuracy depends on real weights being present.

## 5. Scaling considerations

- Each FastAPI worker/replica is stateless except for the DI `Container`
  singleton (tool clients); horizontal scaling behind a load balancer is
  safe.
- `GRAPH_CHECKPOINT_BACKEND=sqlite` is per-pod; if you need cross-pod
  resumability, run a single shared volume or move to a distributed
  checkpointer.
- Gemini and Google Books calls are the latency-dominant steps; both are
  cached (`CACHE_BACKEND=redis`) and batched (`GEMINI_BATCH_SIZE`) to
  control cost and throughput.

## 6. Observability in production

- `/metrics` — scrape with Prometheus (see `deploy/prometheus.yml`).
- Structured JSON logs go to stdout — ship with your log agent of choice
  (Fluent Bit, Vector, CloudWatch agent).
- Set `OTEL_ENABLED=true` to export traces via OTLP/gRPC.
