from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Sequence, Optional, Any

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_csv(path: Path, rows: Iterable[Sequence[Any]], header: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(list(header))
        for r in rows:
            writer.writerow(list(r))
