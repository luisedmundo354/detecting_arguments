"""Schema inspection utilities for Label Studio exports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .parser import iter_annotation_files


def describe_annotation_schema(
    annotation_dirs: Sequence[Path | str],
) -> Dict[str, Any]:
    """Return a JSON-friendly description of the observed export schema."""

    top_level_info = _FieldInfo()
    task_info = _FieldInfo()
    task_data_info = _FieldInfo()
    result_by_type: Dict[str, _FieldInfo] = defaultdict(_FieldInfo)
    result_value_by_type: Dict[str, _FieldInfo] = defaultdict(_FieldInfo)
    result_meta_by_type: Dict[str, _FieldInfo] = defaultdict(_FieldInfo)

    for annotation_dir in annotation_dirs:
        for path in iter_annotation_files(annotation_dir):
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            top_level_info.observe(payload)
            task = payload.get("task", {}) or {}
            if isinstance(task, dict):
                task_info.observe(task)
                task_data = task.get("data", {}) or {}
                if isinstance(task_data, dict):
                    task_data_info.observe(task_data)

            for item in payload.get("result", []) or []:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "unknown")
                result_by_type[item_type].observe(item)
                value = item.get("value", {}) or {}
                meta = item.get("meta", {}) or {}
                if isinstance(value, dict):
                    result_value_by_type[item_type].observe(value)
                if isinstance(meta, dict) and meta:
                    result_meta_by_type[item_type].observe(meta)

    result_items: List[Dict[str, Any]] = []
    for item_type in sorted(result_by_type):
        item_schema = result_by_type[item_type].to_schema_dict()
        item_schema["type"] = item_type
        if item_type in result_value_by_type:
            item_schema["value"] = result_value_by_type[item_type].to_schema_dict()
        if item_type in result_meta_by_type and result_meta_by_type[item_type].field_types:
            item_schema["meta"] = result_meta_by_type[item_type].to_schema_dict()
        result_items.append(item_schema)

    schema = top_level_info.to_schema_dict()
    schema["task"] = task_info.to_schema_dict()
    schema["task"]["data"] = task_data_info.to_schema_dict()
    schema["result"] = result_items
    return schema


class _FieldInfo:
    def __init__(self) -> None:
        self.field_types: Dict[str, set[str]] = defaultdict(set)
        self.field_presence: Dict[str, int] = defaultdict(int)
        self.sample_count = 0

    def observe(self, payload: Dict[str, Any]) -> None:
        self.sample_count += 1
        for key, value in payload.items():
            self.field_presence[key] += 1
            self.field_types[key].add(_describe_value_type(value))

    def to_schema_dict(self) -> Dict[str, str]:
        schema: Dict[str, str] = {}
        for key in sorted(self.field_types):
            type_name = " | ".join(sorted(self.field_types[key]))
            if self.field_presence[key] < self.sample_count:
                type_name = f"optional[{type_name}]"
            schema[key] = type_name
        return schema


def _describe_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        if not value:
            return "array[unknown]"
        element_types = sorted({_describe_value_type(item) for item in value})
        return f"array[{', '.join(element_types)}]"
    return type(value).__name__
