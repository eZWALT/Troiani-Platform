# Setup

Python 3.10+ is enough. A dedicated virtualenv keeps this package off the Troiani LLM environment.

## Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
troiani-platform control --with-tracking
```

In another shell:

```bash
source .venv/bin/activate
troiani-platform worker --node "$(hostname -s)"
```

| Service | Default |
|---------|---------|
| Control + dashboard | http://127.0.0.1:8787 |
| MLflow | http://127.0.0.1:5000 |
| TensorBoard | http://127.0.0.1:6006 |

State, checkpoints, and tracking files land under `var/` (gitignored). MLflow uses `var/mlflow.db` (SQLite). From a laptop, keep a **local port forward** (this is the dashboard pipe). Full story, including the hung-worker `Broken pipe` mix-up: [`ACCESS.md`](ACCESS.md).

```bash
ssh -N -L 8787:127.0.0.1:8787 -L 5000:127.0.0.1:5000 -L 6006:127.0.0.1:6006 atlas
```

## Atlas + Uranus

Atlas runs the control plane. Each GPU node runs one worker that heartbeats over the LAN.

```bash
# from a machine that can SSH to both
./scripts/deploy.sh
# nohup + exit — do not leave ssh attached to worker.sh (it execs the worker)
ssh atlas 'cd ~/Troiani-Platform && mkdir -p var/logs && nohup ./scripts/control.sh >> var/logs/control.out 2>&1 < /dev/null & echo $! > var/logs/control.pid'
ssh atlas 'cd ~/Troiani-Platform && nohup ./scripts/worker.sh atlas >> var/logs/worker-atlas.out 2>&1 < /dev/null & echo $! > var/logs/worker-atlas.pid'
ssh uranus 'cd ~/Troiani-Platform && nohup ./scripts/worker.sh uranus >> var/logs/worker-uranus.out 2>&1 < /dev/null & echo $! > var/logs/worker-uranus.pid'
```

Workers talk to `http://192.168.1.17:8787` by default (`control.public_url` in `config/platform.yaml`). Set `TROIANI_PLATFORM_TOKEN` on every process if the API is reachable beyond localhost.

The platform never claims exclusive ownership of a GPU. If another user is already on a device, discovery marks it `RESEARCHER` and the scheduler yields.

## Tests

```bash
pytest -q
```

Scheduler tests use the in-process simulator and do not need NVIDIA drivers.
