"""Bangun ``data/index.csv`` dari citra hasil ekstraksi.

Pembagian data mengikuti SPEC:
1. Split 70:20:10 stratified per spesies dengan seed tetap.
2. Lima lipatan validasi silang hanya dibagikan di dalam data latih. Data
   validasi dan data uji memakai penanda fold -1.
3. Data uji tidak pernah dipakai untuk keputusan apa pun selama pelatihan.
4. Bobot kelas dihitung dari data latih saja. Modul ini tidak menghitung
   bobot apa pun, hanya menyiapkan label metadata.

Pemakaian:
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

from ..paths import IMAGES_DIR, INDEX_PATH, PROJECT_ROOT
from .extract import IMAGE_SUFFIXES
from .species_map import (
    EXCLUDED_FROM_TRAINING,
    SPECIES,
    TRAINABLE_SPECIES,
    display_name,
)

INDEX_FIELDS = ("path", "species", "species_id", "split", "fold")

RANDOM_SEED = 20260203
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20
N_FOLDS = 5

TEST_FOLD = -1


@dataclass(frozen=True)
class IndexRow:
    """Satu baris index.

    Attributes:
        path: Path citra relatif terhadap root proyek, dengan garis miring maju.
        species: Nama spesies tampilan.
        species_id: Kunci spesies.
        split: train, val, atau test.
        fold: Nomor lipatan 0 sampai 4, atau -1 untuk test.
    """

    path: str
    species: str
    species_id: str
    split: str
    fold: int


def collect_images(images_dir: Path) -> dict[str, list[Path]]:
    """Kumpulkan citra per spesies, diurutkan agar hasilnya deterministik."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for species in SPECIES:
        directory = images_dir / species.species_id
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                grouped[species.species_id].append(path)
    return dict(grouped)


def _split_counts(total: int) -> tuple[int, int]:
    """Tentukan jumlah train dan val untuk satu spesies.

    Args:
        total: Jumlah citra satu spesies.

    Returns:
        Pasangan (jumlah_train, jumlah_val).
    """
    n_train = int(round(total * TRAIN_FRACTION))
    n_val = int(round(total * VAL_FRACTION))
    n_train = min(max(n_train, 1), total - 2)
    n_val = min(max(n_val, 1), total - n_train - 1)
    return n_train, n_val


def build_rows(
    grouped: dict[str, list[Path]],
    seed: int = RANDOM_SEED,
    n_folds: int = N_FOLDS,
) -> list[IndexRow]:
    """Bangun baris index lengkap dengan split dan lipatan.

    Lipatan hanya dibagikan di dalam data latih. Data validasi dan data uji
    memakai ``TEST_FOLD``. Alasannya, bila lipatan mencakup data validasi
    maka data validasi ikut menjadi data latihan pada sebagian lipatan, dan
    optimistic bias muncul saat data validasi yang sama dipakai untuk
    pemilihan checkpoint.

    Args:
        grouped: Citra per species_id.
        seed: Seed untuk pengacakan, agar hasil dapat direproduksi.
        n_folds: Jumlah lipatan validasi silang.

    Returns:
        Daftar IndexRow terurut berdasarkan species_id lalu nama berkas.
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
    return rows


def _relative_posix(path: Path) -> str:
    """Ubah path citra menjadi path relatif proyek dengan garis miring maju.

    Args:
        path: Lokasi berkas citra.

    Returns:
        Path relatif terhadap root proyek.

    Raises:
        ValueError: Bila citra berada di luar root proyek. Menulis path absolut
            ke index.csv melanggar AGENT.md bagian 3 aturan 5, jadi lebih baik
            gagal Than diam-diam membocorkan lokasi mesin lokal.
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(
            f"Citra di luar root proyek: {resolved}. "
            f"Index hanya menerima citra di dalam {PROJECT_ROOT.name}."
        )
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def write_index(rows: list[IndexRow], index_path: Path = INDEX_PATH) -> None:
    """Tulis index.csv."""
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
    """Periksa integritas pembagian data.

    Empat hal diperiksa:
    1. Hanya baris ``train`` yang boleh punya nomor lipatan. Baris ``val``
       dan ``test`` harus memakai ``TEST_FOLD``.
    2. Nomor lipatan pada baris train harus berada di rentang 0 sampai
       ``N_FOLDS - 1``.
    3. Tidak ada citra yang muncul di dua lipatan.
    4. Tidak ada citra yang muncul di dua split.

    Raises:
        RuntimeError: Bila ditemukan kebocoran atau pelanggaran aturan lipatan.
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
    """Cetak ringkasan jumlah citra per spesies, split, dan lipatan."""
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
    """Cetak spesies yang dikecualikan dari pelatihan beserta alasannya.

    Pengecualian dilaporkan eksplisit agar tidak hilang diam-diam dari laporan.
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
    """Titik masuk baris perintah."""
    parser = argparse.ArgumentParser(description="Bangun data/index.csv dari citra DIBaS.")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    grouped = collect_images(args.images_dir)
    if not grouped:
        print("GAGAL: tidak ada citra ditemukan.", file=sys.stderr)
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
    print(f"\nIndex ditulis: {args.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())