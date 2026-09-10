# Seeing the dashboard (and the SSH mix-up)

Two different SSH sessions got confused. Only one of them is how you view the UI on the laptop.

## The pipe you want — local port forward

From the laptop, keep a **local port forward** running (no remote command, just tunnels):

```bash
ssh -N -L 8787:127.0.0.1:8787 -L 5000:127.0.0.1:5000 -L 6006:127.0.0.1:6006 atlas
```

Then on the laptop:

| Service | URL |
|---------|-----|
| Atlas control + dashboard | http://127.0.0.1:8787 |
| MLflow | http://127.0.0.1:5000 |
| TensorBoard | http://127.0.0.1:6006 |

`-N` means “do not open a shell”. This process should stay up while you use the UI. If it dies, the laptop URLs stop working; Atlas itself is unchanged.

CLI through the same tunnel:

```bash
TROIANI_PLATFORM_URL=http://127.0.0.1:8787 troiani-platform status
```

### Same lab LAN (no tunnel needed for the dashboard)

Control binds `0.0.0.0:8787`. On the lab network:

```text
http://192.168.1.17:8787
```

MLflow (`:5000`) and TensorBoard (`:6006`) currently listen on **127.0.0.1 only** on Atlas. Those two still need the `-L` tunnel (or an ssh session on Atlas). They are not reachable as `http://192.168.1.17:5000`.

## The broken pipe you saw — not the dashboard

A **different** long-lived session looked like:

```bash
ssh uranus '… start worker …'
```

That SSH stayed open for hours (the remote command never exited, or stdout stayed attached). When the TCP session died, the laptop printed:

```text
client_loop: send disconnect: Broken pipe
```

Exit code **255**. That is OpenSSH giving up on a hung interactive/remote command. It is **not** how the dashboard is viewed.

If the worker was already started with `nohup` / `setsid`, it keeps running after the SSH disconnect. The broken pipe did not stop Troiani. Check with a pidfile or a **narrow** `ps` match (below), then open the UI via the **port-forward** command above.

`scripts/worker.sh` ends in `exec troiani-platform worker …`. So this **hangs SSH for the life of the worker**:

```bash
# wrong — SSH stays attached to the worker
ssh uranus 'cd ~/Troiani-Platform && ./scripts/worker.sh uranus'
```

## Start workers without leaving SSH hung

Background, redirect, drop stdin, write a pidfile, **exit**. Do this from the laptop; do not leave the session open.

```bash
# Atlas worker
ssh atlas 'cd ~/Troiani-Platform && mkdir -p var/logs && . .venv/bin/activate && nohup ./scripts/worker.sh atlas >> var/logs/worker-atlas.out 2>&1 < /dev/null & echo $! > var/logs/worker-atlas.pid'

# Uranus worker
ssh uranus 'cd ~/Troiani-Platform && mkdir -p var/logs && . .venv/bin/activate && nohup ./scripts/worker.sh uranus >> var/logs/worker-uranus.out 2>&1 < /dev/null & echo $! > var/logs/worker-uranus.pid'
```

Control (Atlas only), same pattern:

```bash
ssh atlas 'cd ~/Troiani-Platform && mkdir -p var/logs && . .venv/bin/activate && nohup ./scripts/control.sh >> var/logs/control.out 2>&1 < /dev/null & echo $! > var/logs/control.pid'
```

Workers already talk to `http://192.168.1.17:8787` (`control.public_url` in `config/platform.yaml`). They do not use your laptop tunnel.

Do **not** restart control/workers unless they are actually down. Never kill researcher PIDs (`gkoutr`, `csp`, vLLM).

### Never `pgrep -f` a pattern that is also on the SSH command line

```bash
# wrong — the remote bash -c line contains the same string, so pgrep matches itself
ssh uranus 'pgrep -f "troiani-platform worker"'
```

That can report a PID when no worker is running (the `pgrep` / `bash -c` process). Use the pidfile, or match the python argv only:

```bash
ssh uranus 'cat ~/Troiani-Platform/var/logs/worker-uranus.pid; ps -p "$(cat ~/Troiani-Platform/var/logs/worker-uranus.pid)" -o pid,etime,cmd'
# or
ssh uranus 'ps -eo pid,cmd | grep "[t]roiani-platform worker --node uranus"'
```

## Architecture

Honest layout (no fake shared filesystem): [`architecture.md`](architecture.md), image [`architecture.png`](architecture.png), script [`architecture.py`](architecture.py).

GPU interconnect facts: [`.agents/NETWORKING.md`](../.agents/NETWORKING.md).  
Where bytes live: [`.agents/STORAGE.md`](../.agents/STORAGE.md).
