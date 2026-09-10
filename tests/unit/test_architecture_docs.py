import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs" / "architecture.py"


def test_architecture_script_compiles_and_stays_honest():
    text = SCRIPT.read_text(encoding="utf-8")
    compile(text, str(SCRIPT), "exec")
    assert "8787" in text
    assert "var/checkpoints" in text
    assert "10GbE" in text
    assert "worker-local" in text
    assert "no tensors" in text
    spec = importlib.util.spec_from_file_location("architecture_docs", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "var/checkpoints" in mod.MERMAID
    assert "8787" in mod.MERMAID
