"""Render an honest Troiani Platform diagram.

Default renderer is Graphviz (readable boxes). Mingrammer ``diagrams``
icons: ``python docs/architecture.py --icons`` (needs ``pip install -e '.[docs]'``).

  python docs/architecture.py
  # writes docs/architecture.png

Mermaid fallback: docs/architecture.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).with_name("architecture.png")

MERMAID = """flowchart TB
  laptop["Laptop browser / CLI"]
  subgraph atlas["Atlas 192.168.1.17"]
    control["Control FastAPI + dashboard :8787"]
    aw["Atlas worker"]
    ackpt["var/checkpoints local"]
  end
  subgraph uranus["Uranus 192.168.1.18"]
    uw["Uranus worker"]
    uckpt["var/checkpoints local"]
  end
  laptop -->|"ssh -L 8787,5000,6006"| control
  aw -->|"HTTP localhost"| control
  aw --> ackpt
  uw -->|"HTTP LAN 10GbE :8787"| control
  uw --> uckpt
"""


def render(path: Path = OUT) -> Path:
    from graphviz import Digraph

    dest = path.with_suffix("")
    g = Digraph("troiani", format="png")
    g.attr(
        rankdir="LR",
        splines="true",
        fontsize="11",
        pad="0.4",
        nodesep="0.45",
        ranksep="0.7",
        bgcolor="white",
        fontname="Helvetica",
        label="No shared filesystem. Same-node jobs only. Researchers (gkoutr / csp) are outside Troiani.",
        labelloc="b",
    )
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize="11",
        fillcolor="white",
        color="#2c3e50",
    )
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#2c3e50")

    g.node("laptop", "Laptop\nbrowser / CLI", fillcolor="#eaf2f8")

    with g.subgraph(name="cluster_atlas") as cluster:
        cluster.attr(
            label="Atlas 192.168.1.17   ·   2× A100-PCIE-40GB   ·   topo NODE (no NVLink)",
            style="rounded",
            color="#1a5276",
            fontsize="11",
        )
        cluster.node(
            "control",
            "Control :8787\nFastAPI + dashboard\nSQLite var/platform.db",
            fillcolor="#d4e6f1",
        )
        cluster.node(
            "track",
            "MLflow :5000\nTensorBoard :6006\n127.0.0.1 only",
            fillcolor="#d5f5e3",
        )
        cluster.node("aw", "Atlas worker")
        cluster.node("ackpt", "var/checkpoints\nworker-local", fillcolor="#fdebd0")
        cluster.node(
            "agpu",
            "GPU1 Troiani\nGPU0 often gkoutr\n(outside)",
            fillcolor="#fadbd8",
        )

    with g.subgraph(name="cluster_uranus") as cluster:
        cluster.attr(
            label="Uranus 192.168.1.18   ·   4× A100 80GB PCIe   ·   NV12 pairs, no NVSwitch",
            style="rounded",
            color="#b9770e",
            fontsize="11",
        )
        cluster.node("uw", "Uranus worker")
        cluster.node("uckpt", "var/checkpoints\nworker-local", fillcolor="#fdebd0")
        cluster.node(
            "ugpu",
            "GPU0–1 NV12\nGPU2–3 often csp\n(outside)",
            fillcolor="#fadbd8",
        )

    g.edge("laptop", "control", label="ssh -N -L 8787,5000,6006\nor LAN :8787")
    g.edge("aw", "control", label="HTTP localhost")
    g.edge("control", "track", style="dashed")
    g.edge("aw", "ackpt")
    g.edge("aw", "agpu", style="dashed")
    g.edge(
        "uw",
        "control",
        label="HTTP 10GbE to :8787\nheartbeats / logs / manifests\n(no tensors, no shared FS)",
        color="#b9770e",
    )
    g.edge("uw", "uckpt")
    g.edge("uw", "ugpu", style="dashed")

    written = Path(g.render(filename=str(dest), cleanup=True))
    if not written.is_file():
        raise FileNotFoundError(f"graphviz did not write {written}")
    return written


def render_diagrams(path: Path = OUT) -> Path:
    """Icon version via mingrammer/diagrams (optional; layout is less controlled)."""
    from diagrams import Cluster, Diagram, Edge
    from diagrams.generic.storage import Storage
    from diagrams.onprem.client import Client
    from diagrams.onprem.compute import Server
    from diagrams.programming.framework import FastAPI

    dest = path.with_suffix("")
    with Diagram(
        "Troiani Platform — laptop tunnel, Atlas control, worker-local checkpoints",
        filename=str(dest),
        outformat="png",
        show=False,
        direction="LR",
        graph_attr={"pad": "0.4", "fontsize": "12", "bgcolor": "white", "splines": "spline"},
    ):
        laptop = Client("Laptop")
        with Cluster("Atlas 192.168.1.17  2x A100-PCIE-40GB  (NODE, no NVLink)"):
            control = FastAPI("Control :8787\nFastAPI + SQLite")
            tracking = Server("MLflow :5000\nTB :6006")
            atlas_worker = Server("Atlas worker")
            atlas_ckpt = Storage("var/checkpoints\nworker-local")
            atlas_gpu = Server("GPU1 / GPU0 gkoutr")
            control >> Edge(style="dashed") >> tracking
            atlas_worker >> Edge(label="HTTP localhost") >> control
            atlas_worker >> atlas_ckpt
            atlas_worker >> Edge(style="dashed") >> atlas_gpu
        with Cluster("Uranus 192.168.1.18  4x A100 80GB PCIe  (NV12 pairs, no NVSwitch)"):
            uranus_worker = Server("Uranus worker")
            uranus_ckpt = Storage("var/checkpoints\nworker-local")
            uranus_gpu = Server("GPU0-1 NV12 / GPU2-3 csp")
            uranus_worker >> uranus_ckpt
            uranus_worker >> Edge(style="dashed") >> uranus_gpu
        laptop >> Edge(label="ssh -L 8787,5000,6006") >> control
        uranus_worker >> Edge(label="HTTP 10GbE :8787 (no tensors)") >> control
    png = dest.with_suffix(".png")
    if not png.is_file():
        raise FileNotFoundError(f"diagrams did not write {png}")
    return png


if __name__ == "__main__":
    fn = render_diagrams if "--icons" in sys.argv else render
    print(fn())
