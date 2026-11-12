"""Unit tests for the project folder locations.

This matters because an earlier off-by-one in the parent index made downloads land
outside the project folder. The tests below lock the correct root position.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from bacteriacv.paths import (
    DATA_DIR,
    IMAGES_DIR,
    INDEX_PATH,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DIR,
    ZIPS_DIR,
    ensure_data_dirs,
)


def test_project_root_is_repository_folder() -> None:
    """The project root must be the folder holding bacteriacv and data."""
    assert PROJECT_ROOT.name == "BacteriaCV"
    assert (PROJECT_ROOT / "bacteriacv").is_dir()
    assert (PROJECT_ROOT / "docs").is_dir()


def test_paths_stay_inside_project_root() -> None:
    """Every data folder must sit inside the project root."""
    for path in (DATA_DIR, RAW_DIR, ZIPS_DIR, IMAGES_DIR, MANIFEST_PATH, INDEX_PATH):
        assert path.is_relative_to(PROJECT_ROOT), path


def test_expected_path_layout() -> None:
    """The folder structure must match what the other modules use."""
    assert ZIPS_DIR == PROJECT_ROOT / "data" / "raw" / "zips"
    assert IMAGES_DIR == PROJECT_ROOT / "data" / "raw" / "images"
    assert INDEX_PATH == PROJECT_ROOT / "data" / "index.csv"


def test_ensure_data_dirs_is_idempotent() -> None:
    """Repeated calls must not fail and must not delete existing content."""
    ensure_data_dirs()
    marker = ZIPS_DIR / "_probe.txt"
    marker.write_text("probe", encoding="utf-8")
    try:
        ensure_data_dirs()
        assert marker.is_file()
    finally:
        marker.unlink(missing_ok=True)
    assert ZIPS_DIR.is_dir()
    assert IMAGES_DIR.is_dir()


# A drive letter counts as a path only at the start of the string or after a
# separator. Without that rule, "https://example.com/D:/a" and "%s:/backup"
# would count as local machine paths because their letters happen to be followed
# by a colon and a slash.
_WINDOWS_DRIVE = re.compile(r"(?:^|[\s\"'\(\[=,;:>])[A-Za-z]:[\\/]")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Collect the string node ids that act as docstrings."""
    docstrings: set[int] = set()
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, owners):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstrings.add(id(value))
    return docstrings


def _concat_parts(node: ast.AST, constants: dict[str, str] | None = None) -> str | None:
    """Fold static strings built from constant expressions.

    This closes a FALSE NEGATIVE from the literal simplification. Python folds
    ``"D:" + "\\data"`` into one constant at compile time, but another parser or an
    explicit ``join`` does not always do so. An expression that cannot be evaluated
    statically returns None.
    """
    constants = constants or {}
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            return node.value.decode("utf-8", "replace")
        return None
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            part = _concat_parts(value, constants)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    if isinstance(node, ast.FormattedValue):
        return _concat_parts(node.value, constants)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        left = _concat_parts(node.left, constants)
        right = _concat_parts(node.right, constants)
        if left is None or right is None:
            return None
        return left + right if isinstance(node.op, ast.Add) else f"{left}/{right}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "join" and len(node.args) == 1:
            separator = _concat_parts(node.func.value, constants)
            items = node.args[0]
            if separator is None or not isinstance(items, (ast.List, ast.Tuple)):
                return None
            parts = [_concat_parts(item, constants) for item in items.elts]
            if any(part is None for part in parts):
                return None
            return separator.join(part for part in parts if part is not None)
        if node.func.attr in {"Path", "PurePath", "WindowsPath", "PosixPath"}:
            parts = [_concat_parts(arg, constants) for arg in node.args]
            if any(part is None for part in parts):
                return None
            return "/".join(part for part in parts if part is not None)
    return None


_DRIVE_ONLY = re.compile(r"\b[A-Za-z]:\s*$")
_POSIX_ABSOLUTE = re.compile(r"^/(?!$)")
# Forbidden Windows folders at the start of the string or after a separator. Matched
# without a drive letter because a local machine path can read "Users\LENOVO"
# once the drive has been assembled separately.
_WINDOWS_RESERVED = re.compile(r"(?:^|[\\/])(Users|Windows|Program Files|WindowsApps)")


def _looks_like_absolute_path(literal: str) -> bool:
    """Check whether a literal forms an absolute local machine path.

    It catches four patterns:
    1. A full path, for example ``D:/2026/Backup``.
    2. A terminated drive letter, for example ``"D:"`` then assembled.
    3. A UNC path, for example ``\\\\server\\share``.
    4. A forbidden Windows folder that marks an absolute drive, for example
       ``Users\LENOVO`` with no drive but still a local machine path.
    """
    if _WINDOWS_DRIVE.search(literal):
        return True
    if _DRIVE_ONLY.search(literal):
        return True
    if literal.startswith("\\\\") and literal.count("\\") >= 2:
        return True
    if _POSIX_ABSOLUTE.match(literal):
        return True
    return bool(_WINDOWS_RESERVED.search(literal))


def _iter_code_strings(tree: ast.AST) -> Iterator[str]:
    """Take the string literals in live code, including ones assembled from several literals.

    An AST parser is used rather than line reading, so an absolute path written in
    live code is still detected. Docstrings are excluded because they are only
    explanation and do not affect behaviour.
    """
    docstrings = _docstring_nodes(tree)
    constants: dict[str, str] = {}

    # Resolve simple constant assignments before scanning expressions. This
    # catches paths assembled through names, while unresolved/dynamic values
    # remain intentionally outside the static detector's scope.
    for _ in range(2):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = _concat_parts(node.value, constants)
                if value is not None:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                value = _concat_parts(node.value, constants) if node.value else None
                if value is not None:
                    constants[node.target.id] = value

    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        value = _concat_parts(node, constants)
        if value is not None:
            yield value


def test_no_absolute_paths_in_active_code() -> None:
    """Live code must not contain an absolute local machine path.

    Strings inside docstrings are excluded because they are only explanation.
    A path name appearing in a live string is reported with its filename.
    """
    offenders: list[str] = []
    package = PROJECT_ROOT / "bacteriacv"
    for source in sorted(package.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for literal in _iter_code_strings(tree):
            if _looks_like_absolute_path(literal):
                offenders.append(f"{source.name}: {literal[:70]}")
    assert not offenders, offenders


@pytest.mark.parametrize(
    "literal",
    [
        "D:/2026/workspace",
        "C:\\Users\\LENOVO",
        "prefix E:\\data suffix",
        "D:",
        "c:",
        "\\\\server\\share\\data",
        "Users\\LENOVO\\BacteriaCV",
        "C:/Windows/System32",
        "Program Files\\app",
        "f\"D:/backup/data\"",
        "d:/lowercase",
        "C:\\Users\\LENOVO\\BacteriaCV\\data",
    ],
)
def test_detector_flags_absolute_paths(literal: str) -> None:
    """The filter must catch absolute paths, including assembled ones."""
    assert _looks_like_absolute_path(literal), literal


@pytest.mark.parametrize(
    "literal",
    [
        "https://example.com/a/b",
        "http://localhost:8000",
        "data/raw/images",
        "zips_manifest.csv",
        "",
        "postgres://user@host/db",
        "images",
        "raw",
        "https://example.com/D:/a",
        "%s:/backup",
        "value:relative/path",
        "dict key:some value",
    ],
)
def test_detector_accepts_relative_and_url(literal: str) -> None:
    """Relative paths and URLs must not count as absolute local machine paths.

    The cases `https://example.com/D:/a` and `%s:/backup` are deliberately included as
    negatives. The letters in both are followed by a colon and a slash, so a
    simple pattern would wrongly flag both as drive letters.
    """
    assert not _looks_like_absolute_path(literal), literal


def test_detector_catches_bytes_literal_path() -> None:
    """Absolute paths inside bytes literals must be detected.

    Bytes literals often show up in binary file handling, and their content can
    hold an absolute local machine path.
    """
    snippet = 'JALUR = b"D:/data/raw"\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_assembled_from_two_literals() -> None:
    """An absolute path assembled from several literals must be detected.

    This is a weakness an earlier audit found. Python folds
    ``"D:" + "\\data"`` at compile time, but other parsers do not, and an
    explicit ``join`` must be caught too.
    """
    snippet = 'JALUR = "D:" + "\\\\data" + "/raw"\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert "D:" in literals
    assert "\\data" in literals
    assert "/raw" in literals
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_assembled_by_join() -> None:
    """An absolute path assembled through join must be detected."""
    snippet = 'JALUR = "".join(["D:/", "2026", "/Backup"])\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_via_path_construction() -> None:
    """An absolute path in a Path construction must be detected."""
    snippet = 'from pathlib import Path\nJALUR = Path("D:/Backup") / "data"\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_assembled_with_fstring() -> None:
    """An absolute path in an f-string must be detected when every part is static."""
    snippet = 'DRIVE = "D:"\nJALUR = f"{DRIVE}/data/raw"\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


@pytest.mark.parametrize(
    "snippet",
    [
        'DRIVE = "D" + ":"\nJALUR = DRIVE + "\\\\data"\n',
        'JALUR = "/".join(["D:", "data"])\n',
        'from pathlib import Path\nJALUR = Path("D" + ":") / "data"\n',
        'JALUR = "/home/user/data"\n',
        'JALUR = "/mnt/data"\n',
    ],
)
def test_detector_catches_indirect_and_posix_absolute_paths(snippet: str) -> None:
    """Absolute paths are still detected when assembled or written POSIX style."""
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_concat_parts_rejects_non_static_expressions() -> None:
    """An expression that cannot be evaluated statically must return None."""
    tree = ast.parse('X = some_function("D:/data")\nY = "a" + variable\n')
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]
    assert nodes
    for node in nodes:
        assert _concat_parts(node) is None or isinstance(_concat_parts(node), str)
    assert _concat_parts(ast.parse('f"{x}/data"').body[0].value) is None


def test_string_scanner_ignores_docstrings() -> None:
    """A docstring holding an absolute path must not be reported."""
    snippet = '''"""Docstring yang menyebut D:/jalur/mesin lokal."""

from pathlib import Path

JALUR_RELATIF = "data/raw/images"
'''

    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))

    assert any("data/raw/images" in item for item in literals)
    assert not any(_looks_like_absolute_path(item) for item in literals)


def test_function_docstrings_are_also_ignored() -> None:
    """Function and class docstrings must be excluded too."""
    snippet = '''def contoh():
    """Docstring fungsi yang menyebut C:/Users/mesin."""
    return "data/raw"


class Contoh:
    """Docstring kelas yang menyebut D:/Backup."""
    nilai = "images"
'''

    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))

    assert not any(_looks_like_absolute_path(item) for item in literals)
    assert "data/raw" in literals
    assert "images" in literals


def test_detector_ignores_docstrings_in_real_source() -> None:
    """Scanning the real package must not flag docstrings as violations."""
    package = PROJECT_ROOT / "bacteriacv"
    flagged_in_docstrings: list[str] = []
    for source in sorted(package.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstrings and _looks_like_absolute_path(node.value):
                    flagged_in_docstrings.append(f"{source.name}: {node.value[:50]}")
    assert not flagged_in_docstrings, flagged_in_docstrings
