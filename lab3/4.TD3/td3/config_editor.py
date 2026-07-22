"""Safe source editor for the project's single-file ``config.py``.

The training panel uses this module to update literal values inside top-level
configuration dictionaries without rewriting the executable helper functions.
A timestamped ``config.py.bak`` copy is created before every successful write.
"""

from __future__ import annotations

import ast
import pprint
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple


class ConfigEditError(RuntimeError):
    pass


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute_offset(offsets: list[int], lineno: int, col_offset: int) -> int:
    return offsets[lineno - 1] + col_offset


def _top_level_values(tree: ast.Module) -> Dict[str, ast.AST]:
    result: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node.value
    return result


def _constant_key(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigEditError("Configuration dictionary keys must be literal values") from exc


def _find_nested_value(root: ast.AST, segments: Iterable[str]) -> ast.AST:
    current = root
    for segment in segments:
        if not isinstance(current, ast.Dict):
            raise ConfigEditError(f"Cannot descend through non-dict value at '{segment}'")
        match = None
        for key_node, value_node in zip(current.keys, current.values):
            if key_node is None:
                continue
            if str(_constant_key(key_node)) == segment:
                match = value_node
                break
        if match is None:
            raise ConfigEditError(f"Config key not found: {segment}")
        current = match
    return current


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (bool, int, str)) or value is None:
        return repr(value)
    return pprint.pformat(value, width=100, compact=True, sort_dicts=False)


def update_config_file(config_path: str | Path, updates: Dict[str, Any]) -> Path:
    """Apply ``{"SECTION.key.subkey": value}`` updates to ``config.py``.

    Only the target literal expressions are replaced. The rest of the source,
    including comments and helper functions, remains unchanged.
    """

    path = Path(config_path).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    top = _top_level_values(tree)
    offsets = _line_offsets(source)
    replacements: list[Tuple[int, int, str]] = []

    for dotted_path, value in updates.items():
        parts = dotted_path.split(".")
        root_name = parts[0]
        if root_name not in top:
            raise ConfigEditError(f"Top-level config object not found: {root_name}")
        target = _find_nested_value(top[root_name], parts[1:]) if len(parts) > 1 else top[root_name]
        if not hasattr(target, "end_lineno") or target.end_lineno is None:
            raise ConfigEditError(f"Python AST lacks source range for: {dotted_path}")
        start = _absolute_offset(offsets, target.lineno, target.col_offset)
        end = _absolute_offset(offsets, target.end_lineno, target.end_col_offset)
        replacements.append((start, end, _format_value(value)))

    new_source = source
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_source = new_source[:start] + replacement + new_source[end:]

    try:
        compile(new_source, str(path), "exec")
    except SyntaxError as exc:
        raise ConfigEditError(f"Edited config.py would be invalid: {exc}") from exc

    backup = path.with_name(path.name + "." + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
    shutil.copy2(path, backup)
    path.write_text(new_source, encoding="utf-8")
    return backup
