"""Unit tests for index building and the data leakage check."""

from __future__ import annotations

import csv
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from bacteriacv.config import INDEX_SEED
from bacteriacv.datasets.build_index import (
    N_FOLDS,
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
from bacteriacv.datasets.build_index import _relative_posix
from bacteriacv.datasets.extract import IMAGE_SUFFIXES, UnreadableImage, write_unreadable_report
from bacteriacv.datasets.species_map import SPECIES, TRAINABLE_SPECIES

# Image counts per species from a structural count of the DIBaS archives. These are the
# figures used to test the proportions at real dataset scale, not the 660 quoted in the
# older literature, which does not match the archive contents. Candida albicans
# is excluded because it is a fungus, and three damaged images are discarded, so the
# effective total is 669 out of 692 files.
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
    "listeria_monocytogenes": 22,
    "micrococcus_spp": 21,
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
    """Build a valid TIFF image file for one species."""
    directory = project_tmp_dir / "escherichia_coli"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"Escherichia.coli_{index:04d}.tif"
        _write_tiff(path)
        paths.append(path)
    return paths


def _write_tiff(path: Path) -> None:
    """Build a valid TIFF that cv2.imread can genuinely open."""
    assert cv2.imwrite(str(path), np.full((32, 32, 3), 150, dtype=np.uint8))


def _full_fake_dataset(project_tmp_dir: Path) -> dict[str, list[Path]]:
    """Build every trainable species with the image count the DIBaS data has."""
    grouped: dict[str, list[Path]] = {}
    for species in TRAINABLE_SPECIES:
        count = REAL_SPECIES_COUNTS[species.species_id]
        directory = project_tmp_dir / species.species_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(count):
            path = directory / f"{species.zip_name}_{index:04d}.tif"
            _write_tiff(path)
            paths.append(path)
        grouped[species.species_id] = paths
    return grouped


# --------------------------------------------------------------------------
# Split proportions
# --------------------------------------------------------------------------


def test_split_counts_respect_70_20_10() -> None:
    """The train and val counts follow a 70:20 ratio and leave the rest for test."""
    n_train, n_val = _split_counts(20)
    assert n_train == 14
    assert n_val == 4
    assert 20 - n_train - n_val == 2


def test_split_counts_handle_small_species() -> None:
    """A species with few images still keeps train, val, and test."""
    for total in (3, 5, 10, 15, 20, 22, 23):
        n_train, n_val = _split_counts(total)
        assert n_train >= 1, total
        assert n_val >= 1, total
        assert n_train + n_val <= total - 1, total


@pytest.mark.parametrize("total", [10, 20, 22, 23, 33, 50, 100])
def test_split_proportions_stay_close_to_target(total: int) -> None:
    """The train and val proportions must sit close to 70:20 at every size."""
    n_train, n_val = _split_counts(total)
    n_test = total - n_train - n_val
    assert abs(n_train / total - 0.70) <= 0.06
    assert abs(n_val / total - 0.20) <= 0.06
    assert abs(n_test / total - 0.10) <= 0.06


def test_full_dataset_totals_match_expected(project_tmp_dir: Path) -> None:
    """The effective dataset must yield 467 train, 136 val, 66 test.

    Candida albicans is excluded because it is a fungus, so the total drops from
    692 to 669. The figure 660 from the older literature is not used because the
    images per species are not uniform.
    """
    grouped = _full_fake_dataset(project_tmp_dir)
    assert sum(len(paths) for paths in grouped.values()) == 669

    rows = build_rows(grouped)
    counts = Counter(row.split for row in rows)

    assert counts["train"] == 467
    assert counts["val"] == 136
    assert counts["test"] == 66
    assert len(rows) == 669


def test_full_dataset_has_all_trainable_species(project_tmp_dir: Path) -> None:
    """All 32 trainable species must appear in the index."""
    grouped = _full_fake_dataset(project_tmp_dir)
    rows = build_rows(grouped)
    assert len({row.species_id for row in rows}) == len(TRAINABLE_SPECIES) == 32


def test_full_dataset_is_leak_free(project_tmp_dir: Path) -> None:
    """A complete dataset must pass the leakage check."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    verify_no_leakage(rows)


# --------------------------------------------------------------------------
# Fold rules: train data only
# --------------------------------------------------------------------------


def test_folds_exist_only_on_train_rows(project_tmp_dir: Path) -> None:
    """Only train rows may carry a fold number."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    for row in rows:
        if row.split != "train":
            assert row.fold == TEST_FOLD, f"{row.path} split={row.split} fold={row.fold}"


def test_every_train_row_has_a_fold(project_tmp_dir: Path) -> None:
    """Every train row must carry a valid fold number."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    for row in rows:
        if row.split == "train":
            assert 0 <= row.fold < N_FOLDS, f"{row.path} fold={row.fold}"


def test_val_and_test_never_receive_fold(project_tmp_dir: Path) -> None:
    """Validation and test rows must always use fold -1, including at 20 images."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    for row in rows:
        if row.split in {"val", "test"}:
            assert row.fold == TEST_FOLD


def test_fold_counts_come_only_from_train(project_tmp_dir: Path) -> None:
    """The counts per fold must add up to the train row count."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    per_fold = Counter(row.fold for row in rows)
    n_train = sum(1 for row in rows if row.split == "train")
    assert sum(per_fold[fold] for fold in range(N_FOLDS)) == n_train


def test_produced_train_folds_are_within_range(project_tmp_dir: Path) -> None:
    """The folds build_rows() produces must sit in the range 0 to 4.

    This test checks the production side. The verification side, that an out of
    range value is rejected, is covered by test_verify_rejects_fold_outside_range.
    """
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    train_rows = [row for row in rows if row.split == "train"]
    assert train_rows
    for row in train_rows:
        assert 0 <= row.fold < N_FOLDS


@pytest.mark.parametrize("bad_fold", [N_FOLDS, TEST_FOLD, 99, -3])
def test_verify_rejects_fold_outside_range(project_tmp_dir: Path, bad_fold: int) -> None:
    """A fold number out of range must be rejected on a train row."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    source = next(row for row in rows if row.split == "train")
    broken = rows + [source.__class__(**{**source.__dict__, "fold": bad_fold})]
    with pytest.raises(RuntimeError, match="Nomor lipatan di luar rentang"):
        verify_no_leakage(broken)


def test_verify_error_message_names_the_offending_fold(
    project_tmp_dir: Path,
) -> None:
    """The error message must name the fold value it found."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    source = next(row for row in rows if row.split == "train")
    broken = rows + [source.__class__(**{**source.__dict__, "fold": 7})]
    with pytest.raises(RuntimeError, match="fold 7"):
        verify_no_leakage(broken)


def test_folds_are_balanced(project_tmp_dir: Path) -> None:
    """Folds must be distributed evenly within one species."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    counts = {fold: 0 for fold in range(N_FOLDS)}
    for row in rows:
        if row.split == "train":
            counts[row.fold] += 1
    assert max(counts.values()) - min(counts.values()) <= 1, counts


def test_verify_rejects_fold_on_val(project_tmp_dir: Path) -> None:
    """A fold on validation data must be rejected."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    val_row = next(row for row in rows if row.split == "val")
    broken = [val_row.__class__(**{**val_row.__dict__, "fold": 2})] + [
        row for row in rows if row is not val_row
    ]
    with pytest.raises(RuntimeError, match="Pelanggaran lipatan"):
        verify_no_leakage(broken)


def test_verify_rejects_fold_on_test(project_tmp_dir: Path) -> None:
    """A fold on test data must be rejected."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: list(paths)})
    test_row = next(row for row in rows if row.split == "test")
    broken = [test_row.__class__(**{**test_row.__dict__, "fold": 3})] + [
        row for row in rows if row is not test_row
    ]
    with pytest.raises(RuntimeError, match="Pelanggaran lipatan"):
        verify_no_leakage(broken)


# --------------------------------------------------------------------------
# Leakage check
# --------------------------------------------------------------------------


def test_build_rows_verifies_itself(project_tmp_dir: Path) -> None:
    """build_rows must run the leakage check before returning.

    A caller that imports build_rows directly must not be able to skip the gate
    that main() applies.
    """
    import inspect

    source = inspect.getsource(build_rows)
    assert "verify_no_leakage(rows)" in source


def test_verify_no_leakage_accepts_valid_rows(project_tmp_dir: Path) -> None:
    """A correct index must pass the leakage check."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(paths)})
    verify_no_leakage(rows)


def test_verify_no_leakage_detects_duplicate_path(project_tmp_dir: Path) -> None:
    """The same image in two splits must count as leakage.

    The error message must name the repeated image, not cross fold overlap.
    """
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(paths)})
    with pytest.raises(RuntimeError, match="Citra berulang"):
        verify_no_leakage(rows + [rows[0]])


def test_verify_no_leakage_detects_shared_fold(project_tmp_dir: Path) -> None:
    """The same image in two folds must count as cross fold leakage.

    The only way one path lands in two folds is for that path to appear twice with
    different fold numbers, so path duplication is a prerequisite for overlap.
    What is distinguished here is the error message: the cross fold check runs
    first, so these tests must fail on the overlap rule, not on the path
    duplication rule.
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
    """The same image in the same fold must hit the repeated image rule."""
    paths = _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(paths)})
    source = next(row for row in rows if row.split == "train")
    twin = source.__class__(**{**source.__dict__, "split": "val", "fold": TEST_FOLD})
    with pytest.raises(RuntimeError, match="Citra berulang"):
        verify_no_leakage(rows + [twin])


# --------------------------------------------------------------------------
# Relative paths and paths outside the root
# --------------------------------------------------------------------------


def test_relative_posix_strips_project_root(project_tmp_dir: Path) -> None:
    """A path inside the project root becomes relative with forward slashes."""
    from bacteriacv.paths import PROJECT_ROOT

    target = PROJECT_ROOT / "data" / "raw" / "images" / "x" / "y.tif"
    assert _relative_posix(target) == "data/raw/images/x/y.tif"


def test_relative_posix_rejects_path_outside_root() -> None:
    """An image outside the project root must be rejected, not written as an absolute path."""
    outside = Path(tempfile.gettempdir()) / "bacteriacv_probe" / "citra.tif"
    with pytest.raises(ValueError, match="di luar root proyek"):
        _relative_posix(outside)


def test_build_rows_rejects_images_outside_root() -> None:
    """build_rows must fail when given an image outside the project root."""
    grouped = {"escherichia_coli": [Path(tempfile.gettempdir()) / "keluar" / "a.tif"]}
    with pytest.raises(ValueError, match="di luar root proyek"):
        build_rows(grouped)


def test_main_writes_index_when_record_is_in_sync(project_tmp_dir: Path) -> None:
    """main must write the index when unreadable.csv matches the disk."""
    from bacteriacv.datasets import build_index as build_index_module
    from bacteriacv.datasets.build_index import main

    images = project_tmp_dir / "images"
    report = project_tmp_dir / "unreadable.csv"
    index = project_tmp_dir / "index.csv"
    _full_fake_dataset(images)

    broken = images / "listeria_monocytogenes" / "rusak_9999.tif"
    broken.write_bytes(b"")
    write_unreadable_report(
        [
            UnreadableImage(
                path=_relative_posix(broken),
                species_id="listeria_monocytogenes",
                size_bytes=0,
                reason="berkas 0 byte",
            )
        ],
        report,
    )

    exit_code = main(
        [
            "--images-dir", str(images),
            "--index-path", str(index),
            "--unreadable", str(report),
        ]
    )

    assert exit_code == 0
    assert index.is_file()
    written = list(csv.DictReader(index.open(newline="", encoding="utf-8")))
    assert len(written) == 669
    assert all("rusak_9999" not in row["path"] for row in written)
    assert build_index_module.TEST_FOLD == -1


def test_main_refuses_when_record_is_out_of_sync(project_tmp_dir: Path) -> None:
    """main must refuse to write the index when there is unrecorded damage."""
    from bacteriacv.datasets.build_index import main

    images = project_tmp_dir / "images"
    report = project_tmp_dir / "unreadable.csv"
    index = project_tmp_dir / "index.csv"
    _full_fake_dataset(images)
    write_unreadable_report([], report)

    broken = images / "listeria_monocytogenes" / "rusak_9998.tif"
    broken.write_bytes(b"tidak boleh bocor ke index")

    exit_code = main(
        [
            "--images-dir", str(images),
            "--index-path", str(index),
            "--unreadable", str(report),
        ]
    )

    assert exit_code == 1
    assert not index.exists(), "index tidak boleh ditulis saat rekonsiliasi gagal"


def test_main_keeps_existing_index_when_refused(project_tmp_dir: Path) -> None:
    """An existing index must not be overwritten when reconciliation fails."""
    from bacteriacv.datasets.build_index import main

    images = project_tmp_dir / "images"
    report = project_tmp_dir / "unreadable.csv"
    index = project_tmp_dir / "index.csv"
    _full_fake_dataset(images)
    index.write_text("path,species,species_id,split,fold\n", encoding="utf-8")
    before = index.read_text(encoding="utf-8")
    write_unreadable_report([], report)

    (images / "listeria_monocytogenes" / "rusak_9997.tif").write_bytes(b"")

    exit_code = main(
        [
            "--images-dir", str(images),
            "--index-path", str(index),
            "--unreadable", str(report),
        ]
    )

    assert exit_code == 1
    assert index.read_text(encoding="utf-8") == before


def test_index_never_contains_absolute_path(project_tmp_dir: Path) -> None:
    """No index row may carry an absolute drive."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    for row in rows:
        assert not row.path.startswith("C:")
        assert not row.path.startswith("D:")
        assert ":" not in row.path
        assert not row.path.startswith("/")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_build_rows_is_deterministic(project_tmp_dir: Path) -> None:
    """A fixed seed must produce an identical index."""
    grouped = _full_fake_dataset(project_tmp_dir)
    first = build_rows(grouped)
    second = build_rows(grouped)
    assert first == second


def test_different_seed_changes_assignment(project_tmp_dir: Path) -> None:
    """A different seed must produce a different split, otherwise the seed is unused."""
    grouped = _full_fake_dataset(project_tmp_dir)
    first = build_rows(grouped, seed=INDEX_SEED)
    second = build_rows(grouped, seed=INDEX_SEED + 1)
    assert [row.path for row in first] != [row.path for row in second] or [
        row.split for row in first
    ] != [row.split for row in second]


def test_index_seed_is_pinned() -> None:
    """The seed must come from config.py so the result is reproducible."""
    from bacteriacv.config import INDEX_SEED

    assert isinstance(INDEX_SEED, int)
    assert INDEX_SEED == 20260203


# --------------------------------------------------------------------------
# Image collection and file output
# --------------------------------------------------------------------------


def test_collect_images_ignores_non_images(project_tmp_dir: Path) -> None:
    """Non image files in a species folder must be ignored."""
    paths = _fake_images(project_tmp_dir, count=3)
    (project_tmp_dir / "escherichia_coli" / "catatan.txt").write_text(
        "bukan citra", encoding="utf-8"
    )
    grouped, skipped = collect_images(project_tmp_dir)
    collected = grouped.get(SPECIES[8].species_id, [])
    assert len(collected) == 3
    assert all(path.suffix.lower() in IMAGE_SUFFIXES for path in collected)
    assert set(collected) == set(paths)
    assert skipped == []


def test_collect_images_skips_unreadable_files(project_tmp_dir: Path) -> None:
    """A file that cannot be opened must be skipped, not fail the run."""
    paths = _fake_images(project_tmp_dir, count=5)
    broken = project_tmp_dir / "escherichia_coli" / "rusak_9999.tif"
    broken.write_bytes(b"bukan tiff" * 100)

    grouped, skipped = collect_images(project_tmp_dir)

    collected = grouped.get(SPECIES[8].species_id, [])
    assert broken not in collected
    assert len(collected) == 5
    assert len(skipped) == 1
    assert skipped[0].endswith("rusak_9999.tif")


def test_collect_images_keeps_unreadable_when_asked(project_tmp_dir: Path) -> None:
    """With skip_unreadable=False every file is returned."""
    _fake_images(project_tmp_dir, count=5)
    broken = project_tmp_dir / "escherichia_coli" / "rusak_9998.tif"
    broken.write_bytes(b"")

    grouped, skipped = collect_images(project_tmp_dir, skip_unreadable=False)

    assert len(grouped.get(SPECIES[8].species_id, [])) == 6
    assert skipped == []


def test_index_roundtrip_keeps_columns(project_tmp_dir: Path) -> None:
    """index.csv must carry every requested column."""
    _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(_fake_images(project_tmp_dir))})
    index_path = project_tmp_dir / "index.csv"
    write_index(rows, index_path)

    with index_path.open(newline="", encoding="utf-8") as handle:
        read_back = list(csv.DictReader(handle))

    assert len(read_back) == len(rows)
    assert set(read_back[0]) == {"path", "species", "species_id", "split", "fold"}


def test_index_is_utf8_without_bom_and_uses_lf(project_tmp_dir: Path) -> None:
    """index.csv must be UTF-8 without BOM and LF line endings."""
    _fake_images(project_tmp_dir)
    rows = build_rows({SPECIES[8].species_id: sorted(_fake_images(project_tmp_dir))})
    index_path = project_tmp_dir / "index.csv"
    write_index(rows, index_path)

    raw = index_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw


def test_report_runs_without_error(project_tmp_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The summary report must print without raising."""
    rows = build_rows(_full_fake_dataset(project_tmp_dir))
    report(rows)
    captured = capsys.readouterr()
    assert "Total 669 citra" in captured.out
    assert "Lipatan (hanya data latih)" in captured.out


def test_index_row_fields_are_stable(project_tmp_dir: Path) -> None:
    """IndexRow must store all five fields in declaration order."""
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