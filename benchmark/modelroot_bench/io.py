"""Results I/O and JSONL serialization (M12).

Provides streaming serialization and deserialization for benchmark specifications,
ground-truth records, scenario results, and paired benchmark records.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from modelroot_bench.scenario import BenchmarkRecord

T = TypeVar("T", bound=BaseModel)


def write_jsonl(records: Iterable[Any], path: str | Path) -> None:
    """Write records to a JSONL file.

    Each record can be a Pydantic model, a dataclass, or a dict.

    Args:
        records: Iterable of objects to serialize.
        path: Destination path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, BaseModel):
                line = rec.model_dump_json()
            elif isinstance(rec, dict):
                line = json.dumps(rec)
            elif hasattr(rec, "__dict__"):
                line = json.dumps(rec.__dict__)
            else:
                line = json.dumps(rec)
            f.write(line + "\n")


def read_jsonl(path: str | Path, model_cls: type[T] | None = None) -> list[Any]:
    """Read records from a JSONL file.

    Args:
        path: Source file path.
        model_cls: Optional Pydantic model class to validate and instantiate.
            If None and model_cls is omitted, lines are returned as dicts.

    Returns:
        List of parsed models or dictionaries.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {in_path}")

    records: list[Any] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {in_path}: {exc}") from exc

            if model_cls is not None:
                records.append(model_cls.model_validate(data))
            else:
                records.append(data)
    return records


def read_benchmark_records(path: str | Path) -> list[BenchmarkRecord]:
    """Convenience helper to read a JSONL file as a list of :class:`BenchmarkRecord`."""
    return read_jsonl(path, model_cls=BenchmarkRecord)
