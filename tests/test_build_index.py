"""Tes unit untuk pembangunan index dan pemeriksaan kebocoran data."""

from __future__ import annotations

import csv
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from bacteriacv.datasets.build_index import (
    N_FOLDS,
    RANDOM_SEED,
    TEST_FOLD,
    IndexRow,
    _relative_posix,
    _split_counts,
    build_rows,
    collect_images,
    report,
    verify_no_leakage,
    write_index,
)
from bacteriacv.datasets.extract import IMAGE_SUFFIXES
from bacteriacv.datasets.species_map import SPECIES, TRAINABLE_SPECIES

# Jumlah citra per spesies dari hitungan struktural arsip DIBaS. Angka ini yang
# dipakai untuk menguji proporsi pada skala data sebenarnya, bukan 660 yang
# disebut literatur lama dan tidak cocok dengan isi arsip. Candida albicans
# dikecualikan karena jamur, sehingga total efektif 672 dari 692 citra.
REAL_SPECIES_COUNTS: dict[str, int] = {
    "acinetobacter_baumannii": 20,
    "actinomyces_israelii": 23,
    "bacteroides_fragilis": 23,
    "bifidobacterium_spp": 23,
    "clostridium_perfringens": 23,
    "enterococcus_faecium": 20,
    "enterococcus_faecalis": 20,
    "escherichia_coli": 20,
    "fusobacterium_spp": 23,
    "lactobacillus_casei": 20,
    "lactobacillus_crispatus": 20,
    "lactobacillus_delbrueckii": 20,
    "lactobacillus_gasseri": 20,
    "lactobacillus_johnsonii_a": 20,
    "lactobacillus_johnsonii_b": 20,
    "lactobacillus_paracasei": 20,
    "lactobacillus_plantarum": 20,
    "lactobacillus_reuteri": 20,
    "lactobacillus_rhamnosus": 20,
    "lactobacillus_salivarius": 20,
    "listeria_monocytogenes": 23,
    "micrococcus_spp": 23,
    "neisseria_gonorrhoeae": 23,
    "porphyromonas_gingivalis": 23,
    "propionibacterium_acnes": 23,
    "proteus_spp": 20,
    "pseudomonas_aeruginosa": 20,
    "staphylococcus_aureus": 20,
    "staphylococcus_epidermidis": 20,
    "staphylococcus_saprophyticus": 20,
    "streptococcus_agalactiae": 20,
    "veillonella_spp": 22,
}


def _fake_images(project_tmp_dir: Path, count: int = 20) -> list[Path]:
    """Buat berkas citra palsu untuk satu spesies."""
    directory = project_tmp_dir / "escherichia_coli"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"Escherichia.coli_{index:04d}.tif"
        path.write_bytes(b"\x00")
        paths.append(path)
    return paths


def _full_fake_dataset(project_tmp_dir: Path) -> dict[str, list[Path]]:
    """Buat seluruh spesies trainable dengan jumlah citra sesuai data DIBaS."""
    grouped: dict[str, list[Path]] = {}
    for species in TRAINABLE_SPECIES:
        count = REAL_SPECIES_COUNTS[species.species_id]
        directory = project_tmp_dir / species.species_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(count):
            path = directory / f"{species.zip_name}_{index:04d}.tif"
            path.write_bytes(b"\x00")
            paths.append(path)
        grouped[species.species_id] = paths
    return grouped


# --------------------------------------------------------------------------
# Proporsi split
# --------------------------------------------------------------------------


def test_split_counts_respect_70_20_10() -> None:
    """Jumlah train dan val mengikuti rasio 70:20 dan menyisakan test."""
    n_train, n_val = _split_counts(20)
    assert n_train == 14
    assert n_val == 4
    assert 20 - n_train - n_val == 2


def test_split_counts_handle_small_species() -> None:
    """Spesies dengan sedikit citra tetap menyisakan train, val, dan test."""
    for total in (3, 5, 10, 15, 20, 22, 23):
        n_train, n_val = _split_counts(total)
        assert n_train >= 1, total
        assert n_val >= 1, total
        assert n_train + n_val <= total - 1, total


@pytest.mark.parametrize("total", [10, 20, 22, 23, 33, 50, 100])
def test_split_proportions_stay_close_to_target(total: int) -> None:
    """Proporsi train dan val harus dekat dengan 70:20 pada semua ukuran."""
    n_train, n_val = _split_counts(total)
    n_test = total - n_train - n_val
    assert abs(n_train / total - 0.70) <= 0.06
    assert abs(n_val / total - 0.20) <= 0.06
    assert abs(n_test / total - 0.10) <= 0.06


def test_full_dataset_totals_match_expected(project_tmp_dir: Path) -> None:
    """Dataset efektif harus menghasilkan 469 train, 138 val, 65 test.

    Candida albicans dikecualikan karena jamur, sehingga total turun dari 692 ke
    672. Angka 660 dari literatur lama tidak dipakai karena jumlah citra per
    spesies tidak seragam.
    """
    grouped = _full_fake_dataset(project_tmp_dir)
    assert sum(len(paths) for paths in grouped.values()) == 672

    rows = build_rows(grouped)
    counts = Counter(row.split for row in rows)

    assert counts["train"] == 469
    assert counts["val"] == 138
    assert counts["test"] == 65
    assert len(rows) == 672


def test_full_dataset_has_all_trainable_species(project_tmp_dir: Path) -> None:
    """Semua 32 spesies trainable harus muncul di index."""
    grouped = _full_fake_dataset(project_tmp_dir)
    rows = build_rows(grouped)
    assert len({row.species_id for row in rows}) == len(TRAINABLE_SPECIES) == 32


def test_full_dataset_is_leak_free(project_tmp_dir: Path) -> None:
    """Dataset lengkap harus lolos pemeriksaan kebocoran."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    verify_no_leakage(rows)


# --------------------------------------------------------------------------
# Aturan lipatan: hanya data latih
# --------------------------------------------------------------------------


def test_folds_exist_only_on_train_rows(project_tmp_dir: Path) -> None:
    """Hanya baris train boleh punya nomor lipatan."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    for row in rows:
        if row.split != "train":
            assert row.fold == TEST_FOLD, f"{row.path} split={row.split} fold={row.fold}"


def test_every_train_row_has_a_fold(project_tmp_dir: Path) -> None:
    """Setiap baris train harus punya nomor lipatan yang valid."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    for row in rows:
        if row.split == "train":
            assert 0 <= row.fold < N_FOLDS, f"{row.path} fold={row.fold}"


def test_val_and_test_never_receive_fold(project_tmp_dir: Path) -> None:
    """Data validasi dan uji harus selalu fold -1, termasuk pada 20 citra."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    for row in rows:
        if row.split in {"val", "test"}:
            assert row.fold == TEST_FOLD


def test_fold_counts_come_only_from_train(project_tmp_dir: Path) -> None:
    """Jumlah per lipatan harus sama dengan jumlah baris train."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    per_fold = Counter(row.fold for row in rows)
    n_train = sum(1 for row in rows if row.split == "train")
    assert sum(per_fold[fold] for fold in range(N_FOLDS)) == n_train


def test_produced_train_folds_are_within_range(project_tmp_dir: Path) -> None:
    """Lipatan yang dihasilkan build_rows() sendiri harus berada di rentang 0 sampai 4.

    Tes ini memeriksa sisi produksi. Sisi verifikasi, yaitu bahwa nilai di luar
    rentang ditolak, diuji oleh test_verify_rejects_fold_outside_range.
    """
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    train_rows = [row for row in rows if row.split == "train"]
    assert train_rows
    for row in train_rows:
        assert 0 <= row.fold < N_FOLDS


@pytest.mark.parametrize("bad_fold", [N_FOLDS, TEST_FOLD, 99, -3])
def test_verify_rejects_fold_outside_range(project_tmp_dir: Path, bad_fold: int) -> None:
    """Nomor lipatan di luar rentang harus ditolak pada baris train."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    source = next(row for row in rows if row.split == "train")
    broken = rows + [source.__class__(**{**source.__dict__, "fold": bad_fold})]
    with pytest.raises(RuntimeError, match="Nomor lipatan di luar rentang"):
        verify_no_leakage(broken)


def test_verify_error_message_names_the_offending_fold(
    project_tmp_dir: Path,
) -> None:
    """Pesan error harus menyebut nilai lipatan yang ditemukan."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    source = next(row for row in rows if row.split == "train")
    broken = rows + [source.__class__(**{**source.__dict__, "fold": 7})]
    with pytest.raises(RuntimeError, match="fold 7"):
        verify_no_leakage(broken)


def test_folds_are_balanced(project_tmp_dir: Path) -> None:
    """Lipatan harus terdistribusi merata pada satu spesies."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    counts = {fold: 0 for fold in range(N_FOLDS)}
    for row in rows:
        if row.split == "train":
            counts[row.fold] += 1
    assert max(counts.values()) - min(counts.values()) <= 1, counts


def test_verify_rejects_fold_on_val(project_tmp_dir: Path) -> None:
    """Lipatan pada data validasi harus ditolak."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    val_row = next(row for row in rows if row.split == "val")
    broken = [val_row.__class__(**{**val_row.__dict__, "fold": 2})] + [
        row for row in rows if row is not val_row
    ]
    with pytest.raises(RuntimeError, match="Pelanggaran lipatan"):
        verify_no_leakage(broken)


def test_verify_rejects_fold_on_test(project_tmp_dir: Path) -> None:
    """Lipatan pada data uji harus ditolak."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    test_row = next(row for row in rows if row.split == "test")
    broken = [test_row.__class__(**{**test_row.__dict__, "fold": 3})] + [
        row for row in rows if row is not test_row
    ]
    with pytest.raises(RuntimeError, match="Pelanggaran lipatan"):
        verify_no_leakage(broken)


# --------------------------------------------------------------------------
# Pemeriksaan kebocoran
# --------------------------------------------------------------------------


def test_verify_no_leakage_accepts_valid_rows(project_tmp_dir: Path) -> None:
    """Index yang benar harus lolos pemeriksaan kebocoran."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(paths)})
    verify_no_leakage(rows)


def test_verify_no_leakage_detects_duplicate_path(project_tmp_dir: Path) -> None:
    """Citra yang sama di dua split harus dianggap kebocoran.

    Pesan error harus menyebut citra berulang, bukan overlap antar lipatan.
    """
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(paths)})
    with pytest.raises(RuntimeError, match="Citra berulang"):
        verify_no_leakage(rows + [rows[0]])


def test_verify_no_leakage_detects_shared_fold(project_tmp_dir: Path) -> None:
    """Citra yang sama pada dua lipatan harus dianggap kebocoran antar lipatan.

    Satu-satunya cara satu path masuk dua lipatan adalah path itu muncul dua
    kali dengan nomor lipatan berbeda, jadi duplikasi path memang prasyarat
    overlap. Yang dibedakan di sini adalah pesan error: pemeriksaan antar
    lipatan dijalankan lebih dulu, jadi Rules ini harus gagal pada aturan
    overlap, bukan pada aturan duplikasi path.
    """
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(paths)})
    source = next(row for row in rows if row.split == "train")
    other_fold = (source.fold + 1) % N_FOLDS
    clash = source.__class__(**{**source.__dict__, "fold": other_fold})
    with pytest.raises(RuntimeError, match="Kebocoran antar lipatan"):
        verify_no_leakage(rows + [clash])


def test_duplicate_path_in_same_fold_is_caught_by_path_rule(
    project_tmp_dir: Path,
) -> None:
    """Citra yang sama pada lipatan sama harus kena aturan citra berulang."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(paths)})
    source = next(row for row in rows if row.split == "train")
    twin = source.__class__(**{**source.__dict__, "split": "val", "fold": TEST_FOLD})
    with pytest.raises(RuntimeError, match="Citra berulang"):
        verify_no_leakage(rows + [twin])


# --------------------------------------------------------------------------
# Path relatif dan path di luar root
# --------------------------------------------------------------------------


def test_relative_posix_strips_project_root(project_tmp_dir: Path) -> None:
    """Path di dalam root proyek jadi relatif dengan garis miring maju."""
    from bacteriacv.paths import PROJECT_ROOT

    target = PROJECT_ROOT / "data" / "raw" / "images" / "x" / "y.tif"
    assert _relative_posix(target) == "data/raw/images/x/y.tif"


def test_relative_posix_rejects_path_outside_root() -> None:
    """Citra di luar root proyek harus ditolak, bukan ditulis sebagai path absolut."""
    outside = Path(tempfile.gettempdir()) / "bacteriacv_probe" / "citra.tif"
    with pytest.raises(ValueError, match="di luar root proyek"):
        _relative_posix(outside)


def test_build_rows_rejects_images_outside_root() -> None:
    """build_rows harus gagal bila diberi citra di luar root proyek."""
    grouped = {"escherichia_coli": [Path(tempfile.gettempdir()) / "keluar" / "a.tif"]}
    with pytest.raises(ValueError, match="di luar root proyek"):
        build_rows(grouped)


def test_index_never_contains_absolute_path(project_tmp_dir: Path) -> None:
    """Tidak ada baris index yang boleh memuat drive absolut."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    for row in rows:
        assert not row.path.startswith("C:")
        assert not row.path.startswith("D:")
        assert ":" not in row.path
        assert not row.path.startswith("/")


# --------------------------------------------------------------------------
# Determinisme
# --------------------------------------------------------------------------


def test_build_rows_is_deterministic(project_tmp_dir: Path) -> None:
    """Seed tetap harus menghasilkan index identik."""
    grouped = _full_fake_dataset(project_tmp_dir)
    first = build_rows(grouped)
    second = build_rows(grouped)
    assert first == second


def test_different_seed_changes_assignment(project_tmp_dir: Path) -> None:
    """Seed berbeda harus menghasilkan pembagian berbeda, kalau tidak seed tidak dipakai."""
    grouped = _full_fake_dataset(project_tmp_dir)
    first = build_rows(grouped, seed=RANDOM_SEED)
    second = build_rows(grouped, seed=RANDOM_SEED + 1)
    assert [row.path for row in first] != [row.path for row in second] or [
        row.split for row in first
    ] != [row.split for row in second]


def test_random_seed_is_pinned() -> None:
    """Seed harus berupa konstanta tertulis agar dapat direproduksi."""
    assert isinstance(RANDOM_SEED, int)
    assert RANDOM_SEED == 20260203


# --------------------------------------------------------------------------
# Pengumpulan citra dan keluaran berkas
# --------------------------------------------------------------------------


def test_collect_images_ignores_non_images(project_tmp_dir: Path) -> None:
    """Berkas non-gambar di folder spesies harus diabaikan."""
    paths = _fake_images(project_tmp_dir, count=3)
    (project_tmp_dir / "escherichia_coli" / "catatan.txt").write_text("bukan citra", encoding="utf-8")
    grouped = collect_images(project_tmp_dir)
    collected = grouped.get(SPECIES[8].species_id, [])
    assert len(collected) == 3
    assert all(path.suffix.lower() in IMAGE_SUFFIXES for path in collected)
    assert set(collected) == set(paths)


def test_index_roundtrip_keeps_columns(project_tmp_dir: Path) -> None:
    """index.csv harus memuat semua kolom yang diminta."""
    _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(_fake_images(project_tmp_dir))})
    index_path = project_tmp_dir / "index.csv"
    write_index(rows, index_path)

    with index_path.open(newline="", encoding="utf-8") as handle:
        read_back = list(csv.DictReader(handle))

    assert len(read_back) == len(rows)
    assert set(read_back[0]) == {"path", "species", "species_id", "split", "fold"}


def test_index_is_utf8_without_bom_and_uses_lf(project_tmp_dir: Path) -> None:
    """index.csv harus UTF-8 tanpa BOM dan akhir baris LF."""
    _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(_fake_images(project_tmp_dir))})
    index_path = project_tmp_dir / "index.csv"
    write_index(rows, index_path)

    raw = index_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw


def test_report_runs_without_error(project_tmp_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Laporan ringkasan harus tercetak tanpa exception."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    report(rows)
    captured = capsys.readouterr()
    assert "Total 672 citra" in captured.out
    assert "Lipatan (hanya data latih)" in captured.out


def test_index_row_fields_are_stable(project_tmp_dir: Path) -> None:
    """IndexRow harus menyimpan kelima field sesuai urutan deklarasi."""
    row = IndexRow(
        path="data/raw/images/x/a.tif",
        species="Escherichia coli",
        species_id="escherichia_coli",
        split="train",
        fold=0,
    )
    assert row.path.endswith("a.tif")
    assert row.split == "train"
    assert row.fold == 0