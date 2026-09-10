from __future__ import annotations

from pathlib import Path


class TensorBoardSink:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, object] = {}

    def _writer(self, run_id: str):
        if run_id not in self._writers:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writers[run_id] = SummaryWriter(log_dir=str(self.root / run_id))
            except Exception:
                try:
                    from tensorboardX import SummaryWriter  # type: ignore

                    self._writers[run_id] = SummaryWriter(log_dir=str(self.root / run_id))
                except Exception:
                    self._writers[run_id] = None
        return self._writers[run_id]

    def log(self, run_id: str, name: str, value: float, step: int) -> None:
        writer = self._writer(run_id)
        if writer is None:
            return
        writer.add_scalar(name, value, step)

    def close(self) -> None:
        for writer in self._writers.values():
            if writer is not None:
                writer.close()
