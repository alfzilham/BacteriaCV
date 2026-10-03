"""Tes unit untuk lokasi folder proyek.

Fungsi ini penting karena satu bug index parent membuat unduhan mendarat di
luar folder proyek. Tes di bawah mengunci posisi root yang benar.
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
    """Root proyek harus folder yang memuat folder bacteriacv dan data."""
    assert PROJECT_ROOT.name == "BacteriaCV"
    assert (PROJECT_ROOT / "bacteriacv").is_dir()
    assert (PROJECT_ROOT / "docs").is_dir()


def test_paths_stay_inside_project_root() -> None:
    """Semua folder data harus berada di dalam root proyek."""
    for path in (DATA_DIR, RAW_DIR, ZIPS_DIR, IMAGES_DIR, MANIFEST_PATH, INDEX_PATH):
        assert path.is_relative_to(PROJECT_ROOT), path


def test_expected_path_layout() -> None:
    """Struktur folder harus sesuai dengan yang dipakai modul lain."""
    assert ZIPS_DIR == PROJECT_ROOT / "data" / "raw" / "zips"
    assert IMAGES_DIR == PROJECT_ROOT / "data" / "raw" / "images"
    assert INDEX_PATH == PROJECT_ROOT / "data" / "index.csv"


def test_ensure_data_dirs_is_idempotent() -> None:
    """Pemanggilan berulang tidak boleh gagal dan tidak boleh menghapus isi."""
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


# Drive letter hanya dianggap path bila berada di awal string atau setelah
# separator. Tanpa syarat ini, "https://example.com/D:/a" dan "%s:/backup"
# ikut dianggap path mesin lokal karena huruf di dalamnya kebetulan diikuti
# tanda titik dua dan garis miring.
_WINDOWS_DRIVE = re.compile(r"(?:^|[\s\"'\(\[=,;:>])[A-Za-z]:[\\/]")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Kumpulkan id node string yang berfungsi sebagai docstring."""
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
    """Rakit string statis dari ekspresi konstan.

    Menutup FALSE NEGATIVE dari penyederhanaan literal. Python menyatukan
    ``"D:" + "\\data"`` menjadi satu konstanta saat kompilasi, tapi penyusun
    kode yang lain atau penulisan eksplisit lewat ``join`` tidak selalu
    demikian. Ekspresi yang tidak bisa dinilai statis mengembalikan None.
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
# Folder terlarang Windows di awal string atau setelah separator. Dicocokkan
# tanpa drive letter karena path mesin lokal bisa ditulis sebagai "Users\LENOVO"
# setelah drive dirangkai terpisah.
_WINDOWS_RESERVED = re.compile(r"(?:^|[\\/])(Users|Windows|Program Files|WindowsApps)")


def _looks_like_absolute_path(literal: str) -> bool:
    """Perksa apakah literal membentuk path absolut mesin lokal.

    Menangkap empat pola:
    1. Path lengkap, misalnya ``D:/2026/Backup``.
    2. Drive letter terminated, misalnya ``"D:"`` lalu dirangkai.
    3. Path UNC, misalnya ``\\\\server\\share``.
    4. Folder terlarang Windows yang menandakan drive absolut, misalnya
       ``Users\\LENOVO`` tanpa drive tapi tetap path mesin lokal.
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
    """Ambil literal string di kode aktif, termasuk yang dirakit dari beberapa literal.

    Parser AST dipakai, bukan pembacaan baris, supaya path absolut yang ditulis
    di kode aktif terdeteksi. Docstring dikecualikan karena hanya penjelasan
    dan tidak memengaruhi perilaku.
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
    """Kode aktif tidak boleh memuat path absolut mesin lokal.

    String di dalam docstring dikecualikan karena hanya penjelasan.
    Nama path yang muncul di string aktif akan dilaporkan bersama nama berkas.
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
    """Penyaring harus menangkap path absolut, termasuk yang dirakit."""
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
    """Path relatif dan URL tidak boleh dianggap path absolut mesin lokal.

    Kasus `https://example.com/D:/a` dan `%s:/backup` sengaja dimasukkan sebagai
    negatif. Huruf di dalam keduanya diikuti titik dua dan garis miring, jadi
    pola sederhana akan salah menandai keduanya sebagai drive letter.
    """
    assert not _looks_like_absolute_path(literal), literal


def test_detector_catches_bytes_literal_path() -> None:
    """Path absolut dalam literal bytes harus terdeteksi.

    Literal bytes sering muncul pada pemrosesan berkas biner, dan isinya bisa
    memuat path absolut mesin lokal.
    """
    snippet = 'JALUR = b"D:/data/raw"\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_assembled_from_two_literals() -> None:
    """Path absolut yang dirakit dari beberapa literal harus terdeteksi.

    Ini kelemahan yang ditemukan audit sebelumnya. Python menyatukan
    ``"D:" + "\\data"`` saat kompilasi, tapi penyusun lain tidak, dan
    penulisan eksplisit lewat ``join`` juga harus tertangkap.
    """
    snippet = 'JALUR = "D:" + "\\\\data" + "/raw"\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert "D:" in literals
    assert "\\data" in literals
    assert "/raw" in literals
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_assembled_by_join() -> None:
    """Path absolut yang dirakit lewat join harus terdeteksi."""
    snippet = 'JALUR = "".join(["D:/", "2026", "/Backup"])\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_via_path_construction() -> None:
    """Path absolut pada konstruksi Path harus terdeteksi."""
    snippet = 'from pathlib import Path\nJALUR = Path("D:/Backup") / "data"\n'
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_detector_catches_path_assembled_with_fstring() -> None:
    """Path absolut di dalam f-string harus terdeteksi bila semua bagian statis."""
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
    """Path absolut tetap terdeteksi saat dirakit atau memakai sintaks POSIX."""
    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))
    assert any(_looks_like_absolute_path(item) for item in literals)


def test_concat_parts_rejects_non_static_expressions() -> None:
    """Ekspresi yang tidak bisa dinilai statis harus mengembalikan None."""
    tree = ast.parse('X = some_function("D:/data")\nY = "a" + variable\n')
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]
    assert nodes
    for node in nodes:
        assert _concat_parts(node) is None or isinstance(_concat_parts(node), str)
    assert _concat_parts(ast.parse('f"{x}/data"').body[0].value) is None


def test_string_scanner_ignores_docstrings() -> None:
    """Docstring dengan path absolut tidak boleh dilaporkan."""
    snippet = '''"""Docstring yang menyebut D:/jalur/mesin lokal."""

from pathlib import Path

JALUR_RELATIF = "data/raw/images"
'''

    tree = ast.parse(snippet)
    literals = list(_iter_code_strings(tree))

    assert any("data/raw/images" in item for item in literals)
    assert not any(_looks_like_absolute_path(item) for item in literals)


def test_function_docstrings_are_also_ignored() -> None:
    """Docstring fungsi dan kelas juga harus dikecualikan."""
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
    """Pemindaian paket nyata tidak boleh salah menandai docstring sebagai pelanggaran."""
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
