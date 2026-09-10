<div align="center">

  <a href="https://github.com/eZWALT/Troiani-Platform">
    <img src="resources/troiani_logo_white.png" alt="Troiani Platform" width="350"/>
  </a>

  <h1>Troiani Platform</h1>

  <p><strong>A cooperative, preemptible training platform for otherwise available GPUs.</strong></p>

  <p>
    <a href="https://github.com/eZWALT/Troiani-Platform/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg">
    </a>
    <a href="https://github.com/eZWALT/Troiani-Platform">
      <img alt="Status" src="https://img.shields.io/badge/status-pre--alpha-orange.svg">
    </a>
    <a href="https://github.com/eZWALT/Troiani-Platform/stargazers">
      <img alt="GitHub stars" src="https://img.shields.io/github/stars/eZWALT/Troiani-Platform?style=social">
    </a>
    <a href="https://github.com/eZWALT/Troiani-Platform/commits/main">
      <img alt="Last commit" src="https://img.shields.io/github/last-commit/eZWALT/Troiani-Platform?color=brightgreen">
    </a>
    <a href="https://github.com/eZWALT/Troiani-Platform/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/eZWALT/Troiani-Platform">
    </a>
  </p>

</div>

---

> **Troiani Platform** is the control plane for opportunistic Troiani training. It schedules preemptible jobs, checkpoints on demand, and releases GPUs the moment higher-priority work appears.

This repository is **just the platform**. Model code lives in [Troiani](https://github.com/eZWALT/Troiani).

## Why

Researchers always have priority. Troiani workloads are guests.

- **Researcher-first** — foreign GPU processes, logins, and intent signals preempt Troiani jobs. We never hide, disguise, or interfere with other work.
- **Easy to stop** — checkpoint, release, resume. That is the product.
- **Honest capacity** — use idle GPUs; yield when they are no longer idle.
- **Observable** — jobs, GPUs, checkpoints, and training health stay visible in one place.

## Install

```bash
git clone https://github.com/eZWALT/Troiani-Platform.git
cd Troiani-Platform
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

See [`docs/setup.md`](docs/setup.md) for Atlas / Uranus, MLflow, and TensorBoard.

## Run

**Local (no cluster required)**

```bash
troiani-platform control
# another terminal
troiani-platform worker --node local
troiani-platform status
```

Dashboard: `http://127.0.0.1:8787`  
MLflow: `http://127.0.0.1:5000`

**Atlas + Uranus**

Control plane on Atlas; one worker per node. Deploy and start with:

```bash
./scripts/deploy.sh
./scripts/control.sh          # atlas
./scripts/worker.sh atlas     # atlas
./scripts/worker.sh uranus    # uranus
```

Details in [`docs/setup.md`](docs/setup.md). From the laptop, keep the port-forward up (`docs/ACCESS.md`):

```bash
ssh -N -L 8787:127.0.0.1:8787 -L 5000:127.0.0.1:5000 -L 6006:127.0.0.1:6006 atlas
```

Then http://127.0.0.1:8787 (or LAN http://192.168.1.17:8787). That tunnel is not the same thing as a hung `ssh uranus '… worker …'` session.

## CLI

```bash
troiani-platform status
troiani-platform gpu list
troiani-platform job list
troiani-platform job submit config/jobs/example.yaml
troiani-platform job pause JOB_ID
troiani-platform job resume JOB_ID
troiani-platform job preempt JOB_ID
troiani-platform job cancel JOB_ID
troiani-platform release-all
troiani-platform run inspect RUN_ID
troiani-platform run lineage RUN_ID
troiani-platform reproduce RUN_ID
troiani-platform checkpoint list RUN_ID
```

`troiani platform …` is an alias for the same commands.

## Config

Default policy and paths live in [`config/platform.yaml`](config/platform.yaml). Job specs are YAML under [`config/jobs/`](config/jobs/). Override the file with `TROIANI_PLATFORM_CONFIG`.

## Docs

| Doc | What |
|-----|------|
| [`docs/setup.md`](docs/setup.md) | Environments, servers, tracking |
| [`docs/ACCESS.md`](docs/ACCESS.md) | Laptop tunnel vs LAN, worker start (no hung SSH) |
| [`docs/architecture.md`](docs/architecture.md) | Honest layout + [`architecture.png`](docs/architecture.png) |
| [`docs/usage.md`](docs/usage.md) | Jobs, checkpoints, preemption |
| [`docs/policy.md`](docs/policy.md) | Windows, limits, researcher-first rules |

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">

**[Star Troiani](https://github.com/eZWALT/Troiani)** for the model family · **[Star this repo](https://github.com/eZWALT/Troiani-Platform)** for the platform

</div>
