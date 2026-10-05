"""Build ``data/index.csv`` from the extracted images.

The data split follows SPEC:
1. A 70:20:10 split stratified per species with a fixed seed.
2. The five cross validation folds are distributed only within the train data.
   Validation and test data use the fold -1 marker.
3. Test data is never used for any decision during training.
4. Class weights are computed from train data only. This module computes no
   weights at all, it only prepares metadata labels.

Usage:
    python -m bacteriacv.datasets.build_index
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..config import INDEX_SEED, N_FOLDS, TRAIN_FRACTION, VAL_FRACTION
from ..paths import IMAGES_DIR, INDEX_PATH, PROJECT_ROOT, UNREADABLE_PATH
from .extract import IMAGE_SUFFIXES, read_unreadable_report
from .species_map import (
    EXCLUDED_FROM_TRAINING,
    SPECIES,
    TRAINABLE_SPECIES,
    display_name,
)

INDEX_FIELDS = ("path", "species", "species_id", "split", "fold")

TEST_FOLD = -1


@dataclass(frozen=True)
class IndexRow:
    """One index row.

    Attributes:
        path: The image path relative to the project root, with forward slashes.
        species: The display species name.
        species_id: The species key.
        split: train, val, or test.
        fold: The fold number 0 to 4, or -1 for test.
    """

    path: str
    species: str
    species_id: str
    split: str
    fold: int


def collect_images(
    images_dir: Path, skip_unreadable: bool = True
) -> tuple[dict[str, list[Path]], list[str]]:
    """Collect the readable images per species, in a deterministic order.

    Args:
        images_dir: The folder of extracted images.
        skip_unreadable: When True, files cv2.imread cannot open are
            skipped and returned as a second list.

    Returns:
        A (species_id to path list map, skipped path list) pair.
    """
    grouped: dict[str, list[Path]] = defaultdict(list)
    skipped: list[str] = []
    for species in TRAINABLE_SPECIES:
        directory = images_dir / species.species_id
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if skip_unreadable and cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
                skipped.append(_relative_posix(path))
                continue
            grouped[species.species_id].append(path)
    return dict(grouped), sorted(skipped)


def _split_counts(total: int) -> tuple[int, int]:
    """Determine the train and val counts for one species.

    Args:
        total: The image count of one species.

    Returns:
        A (train_count, val_count) pair.
    """
    n_train = int(round(total * TRAIN_FRACTION))
    n_val = int(round(total * VAL_FRACTION))
    n_train = min(max(n_train, 1), total - 2)
    n_val = min(max(n_val, 1), total - n_train - 1)
    return n_train, n_val


def build_rows(
    grouped: dict[str, list[Path]],
    seed: int = INDEX_SEED,
    n_folds: int = N_FOLDS,
) -> list[IndexRow]:
    """Build the full index rows with split and fold.

    Folds are distributed only within the train data. Validation and test data
    use ``TEST_FOLD``. The reason is that if folds covered validation data,
    validation data would become training data in some folds, and
    optimistic bias would appear when the same validation data is used for
    checkpoint selection.

    Args:
        grouped: The images per species_id.
        seed: The shuffling seed, so the result is reproducible.
        n_folds: The number of cross validation folds.

    Returns:
        A list of IndexRow sorted by species_id then filename. The leakage
        check runs at the end so a caller cannot skip it.

    Raises:
        ValueError: When an image lies outside the project root.
        RuntimeError: When the split result violates the split or fold rules.
    """
    rng = random.Random(seed)
    rows: list[IndexRow] = []

    for species in TRAINABLE_SPECIES:
        paths = grouped.get(species.species_id, [])
        if not paths:
            continue

        shuffled = list(paths)
        rng.shuffle(shuffled)

        n_train, n_val = _split_counts(len(shuffled))
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        folds = [index % n_folds for index in range(len(train))]
        rng.shuffle(folds)
        fold_of = dict(zip(train, folds))

        for split, members in (("train", train), ("val", val), ("test", test)):
            for path in members:
                rows.append(
                    IndexRow(
                        path=_relative_posix(path),
                        species=species.display_name,
                        species_id=species.species_id,
                        split=split,
                        fold=fold_of.get(path, TEST_FOLD),
                    )
                )

    rows.sort(key=lambda row: (row.species_id, row.path))
    verify_no_leakage(rows)
    return rows


def _relative_posix(path: Path) -> str:
    """Turn an image path into a project relative path with forward slashes.

    Args:
        path: The image file location.

    Returns:
        A path relative to the project root.

    Raises:
        ValueError: When the image lies outside the project root. Writing an
            absolute path into index.csv breaks AGENT.md section 3 rule 5, so
            failing is better than silently leaking a local machine location.
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(
            f"Citra di luar root proyek: {resolved}. "
            f"Index hanya menerima citra di dalam {PROJECT_ROOT.name}."
        )
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def write_index(rows: list[IndexRow], index_path: Path = INDEX_PATH) -> None:
    """Write index.csv."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "species": row.species,
                    "species_id": row.species_id,
                    "split": row.split,
                    "fold": row.fold,
                }
            )


def verify_no_leakage(rows: list[IndexRow]) -> None:
    """Check the integrity of the data split.

Four things are checked:
    1. Only ``train`` rows may carry a fold number. ``val`` and ``test``
       rows must use ``TEST_FOLD``.
    2. The fold number on a train row must lie in the range 0 to
       ``N_FOLDS - 1``.
    3. No image appears in two folds.
    4. No image appears in two splits.

    Raises:
        RuntimeError: When leakage or a fold rule violation is found.
    """
    for row in rows:
        if row.split != "train" and row.fold != TEST_FOLD:
            raise RuntimeError(
                f"Pelanggaran lipatan: {row.path} berlabel split {row.split} "
                f"lalu mendapat fold {row.fold}. Hanya data latih yang boleh "
                f"punya nomor lipatan."
            )
        if row.split == "train" and not 0 <= row.fold < N_FOLDS:
            raise RuntimeError(
                f"Nomor lipatan di luar rentang: {row.path} pada split train "
                f"mendapat fold {row.fold}. Rentang yang sah adalah "
                f"0 sampai {N_FOLDS - 1}."
            )

    fold_members: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if row.fold != TEST_FOLD:
            fold_members[row.fold].add(row.path)

    folds = sorted(fold_members)
    for i, left in enumerate(folds):
        for right in folds[i + 1 :]:
            overlap = fold_members[left] & fold_members[right]
            if overlap:
                raise RuntimeError(
                    f"Kebocoran antar lipatan: {len(overlap)} citra muncul di "
                    f"lipatan {left} dan lipatan {right}. Contoh: "
                    f"{sorted(overlap)[0]}."
                )

    seen: dict[str, str] = {}
    for row in rows:
        if row.path in seen:
            raise RuntimeError(
                f"Citra berulang: {row.path} muncul di split {seen[row.path]} "
                f"dan split {row.split}."
            )
        seen[row.path] = row.split


def report(rows: list[IndexRow]) -> None:
    """Print a summary of image counts per species, split and fold."""
    per_species = Counter(row.species_id for row in rows)
    per_split = Counter(row.split for row in rows)
    per_fold = Counter(row.fold for row in rows)

    print("Jumlah citra per spesies")
    for species in TRAINABLE_SPECIES:
        print(f"  {species.species_id:38s} {per_species.get(species.species_id, 0):3d}")

    total = len(rows)
    print(f"\nTotal {total} citra")
    print("Split")
    for name in ("train", "val", "test"):
        count = per_split[name]
        print(f"  {name:6s} {count:4d}  {count / total * 100:5.1f}%")

    print("Lipatan (hanya data latih)")
    for fold in range(N_FOLDS):
        print(f"  fold {fold}  {per_fold[fold]:4d}")
    print(f"  val    {per_split['val']:4d}  (fold -1)")
    print(f"  test   {per_split['test']:4d}  (fold -1)")


def report_exclusions() -> None:
    """Print the species excluded from training along with their reasons.

    The exclusion is reported explicitly so it does not vanish from the report.
    """
    if not EXCLUDED_FROM_TRAINING:
        return
    print("\nSpesies dikecualikan dari pelatihan")
    for species in SPECIES:
        reason = EXCLUDED_FROM_TRAINING.get(species.species_id)
        if reason:
            print(f"  {species.display_name} ({species.species_id})")
            print(f"    alasan: {reason}")


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    The reconciliation gate is here: the index is only written when the set of
    unreadable images matches exactly what unreadable.csv records.
    New damage or an unrecorded change to the list makes the process
    stop without touching the index.
    """
    parser = argparse.ArgumentParser(description="Bangun data/index.csv dari citra DIBaS.")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--unreadable", type=Path, default=UNREADABLE_PATH)
    parser.add_argument("--seed", type=int, default=INDEX_SEED)
    args = parser.parse_args(argv)

    grouped, skipped = collect_images(args.images_dir)
    if not grouped:
        print("GAGAL: tidak ada citra ditemukan.", file=sys.stderr)
        return 1

    expected_skipped = {row["path"] for row in read_unreadable_report(args.unreadable)}
    actually_skipped = set(skipped)
    if actually_skipped != expected_skipped:
        missing = actually_skipped - expected_skipped
        stale = expected_skipped - actually_skipped
        print(
            "GAGAL: daftar citra tidak terbaca tidak cocok dengan "
            f"{args.unreadable.name}. Jalankan ekstraksi dengan verifikasi lebih dulu."
        )
        if missing:
            print(f"  tidak tercatat di laporan: {sorted(missing)}")
        if stale:
            print(f"  tercatat tapi ternyata terbaca: {sorted(stale)}")
        return 1

    rows = build_rows(grouped, seed=args.seed)

    try:
        verify_no_leakage(rows)
    except RuntimeError as error:
        print(f"GAGAL: {error}", file=sys.stderr)
        return 1

    write_index(rows, args.index_path)
    report(rows)
    report_exclusions()
    if skipped:
        print(f"\n{len(skipped)} citra tidak terbaca dan tidak masuk index:")
        for name in skipped:
            print(f"  {name}")
    print(f"\nIndex ditulis: {args.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())