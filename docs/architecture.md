# Architecture (this lab)

Render the figure with:

```bash
pip install -e ".[docs]"   # diagrams + graphviz
python docs/architecture.py            # readable Graphviz boxes (committed)
python docs/architecture.py --icons    # mingrammer/diagrams icons
```

Output: [`architecture.png`](architecture.png).

```mermaid
flowchart TB
  laptop["Laptop browser / CLI"]
  subgraph atlas["Atlas 192.168.1.17"]
    control["Control FastAPI + dashboard :8787\nSQLite var/platform.db"]
    mlflow["MLflow :5000 localhost"]
    tb["TensorBoard :6006 localhost"]
    aw["Atlas worker"]
    ackpt["var/checkpoints worker-local"]
    ag0["GPU0 A100-PCIE-40GB\ngkoutr RESEARCHER"]
    ag1["GPU1 A100-PCIE-40GB"]
  end
  subgraph uranus["Uranus 192.168.1.18"]
    uw["Uranus worker"]
    uckpt["var/checkpoints worker-local"]
    ug01["GPU0–1 A100 80GB PCIe NV12 pair"]
    ug23["GPU2–3 A100 80GB PCIe NV12 pair\ncsp vLLM RESEARCHER"]
  end
  laptop -->|"ssh -L 8787,5000,6006"| control
  laptop -->|"ssh -L"| mlflow
  laptop -->|"ssh -L"| tb
  laptop -->|"LAN http://192.168.1.17:8787"| control
  aw -->|"HTTP localhost heartbeats / logs / manifests"| control
  aw --> ackpt
  aw -.-> ag1
  ag0 -.->|"outside Troiani"| aw
  uw -->|"HTTP LAN 10GbE to :8787"| control
  uw --> uckpt
  uw -.-> ug01
  ug23 -.->|"outside Troiani"| uw
```

Honest constraints:

- Checkpoints stay on the node that wrote them (`var/checkpoints`). Atlas stores **manifests** in SQLite, not the weight files. Atlas-as-store is not implemented. There is no shared filesystem on this diagram.
- Placement is **same-node** only. A “2-node smoke” is two independent jobs. No NCCL across Atlas and Uranus (two chassis; 10GbE only).
- Researchers on GPUs are outside the platform. Never kill those PIDs.
- MLflow and TensorBoard bind loopback on Atlas; the dashboard binds `0.0.0.0:8787`.
