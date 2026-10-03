# BacteriaCV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sistem klasifikasi bentuk sel dan status Gram bakteri dari citra mikroskopis DIBaS, dengan pipeline pra-pemrosesan lima tahap, ResNet-50 pretrained yang dibekukan, dua classification head, evaluasi F1-score makro, dan antarmuka web FastAPI.

**Architecture:** Satu proses Python, tanpa layanan tambahan. Tahap pelatihan dan inferensi berbagi modul pra-pemrosesan dan backbone yang sama. Karena backbone dibekukan, fitur 2048-d diekstrak sekali ke file `.npy` (5,5 MB untuk 672 citra), lalu kedua head dilatih pada matriks fitur tersebut. Antarmuka web adalah satu proses FastAPI yang memuat checkpoint PyTorch dan melayani satu halaman HTML statis.

**Tech Stack:** Python 3.11, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, OpenCV 5.0.0, scikit-image 0.26.0, scikit-learn 1.9.1, FastAPI 0.142.2, uvicorn 0.54.0, pytest 9.1.1.

---

## Keputusan Desain yang Sudah Ditetapkan

| ID | Keputusan | Alasan |
|----|-----------|--------|
| D1 | Head A dua kelas: cocci dan bacilli | Tidak ada spesies DIBaS dengan bentuk spiral. Mempertahankan kelas kosong membuat F1 makro tidak terdefinisi. |
| D2 | Enum bentuk tetap memuat `spiral`, ditandai tidak terisi | DIBaS tidak punya spesies spiral. Dicatat di laporan, bukan dihapus diam-diam. |
| D3 | Augmentasi 4 varian pada data latih saja | Mengurangi overfitting head pada 469 sampel. Val dan test tetap citra asli tanpa duplikasi. |
| D4 | Ekstraksi fitur ke `.npy`, bukan ke database | 5,5 MB muat di memory. Database menambah dependency tanpa manfaat. |
| D5 | Antarmuka satu proses FastAPI + satu HTML | Model PyTorch tidak bisa dimuat dari Node.js. Prisma tidak ada gunanya karena data immutable. |
| D6 | Bobot kelas dihitung dari data latih saja | SPEC bagian 7. Memakai data lain memunculkan kebocoran. |
| D7 | Candida albicans dikeluarkan dari pelatihan | Jamur, bukan bakteri. Sel 5 sampai 10 mikron membuat kelas ini mudah dipisah dan menaikkan F1 tanpa menunjukkan kemampuan nyata. |

## Angka Terukur yang Menjadi Dasar Rencana

Diukur di mesin ini, Ryzen 7 5825U 8 core, RAM 13,8 GB, tanpa GPU.

```
ResNet-50 forward       76 ms/citra   ->  0,9 menit untuk 672
Pra-pemrosesan          52 ms/citra   ->  0,6 menit untuk 672
Baca TIFF               15 ms/citra   ->  10 detik untuk 672
Fitur 672 x 2048 f32               =  5,5 MB
Head training per epoch            <  1 detik
```

Konsekuensi: tidak ada kebutuhan Colab, tidak ada job queue, tidak ada Docker multi-stage.

## Draft Lookup Table [L] untuk Persetujuan Pemilik

Status Gram dan bentuk sel menurut mikrobiologi standar. Kolom catatan menandai kasus yang perlu dikoreksi pemilik proyek.

| species_id | Spesies | Bentuk | Gram | Catatan |
|---|---|---|---|---|
| acinetobacter_baumannii | Acinetobacter baumannii | bacilli | negatif | Coccobacillus, pendek. Dipetakan ke bacilli. |
| actinomyces_israelii | Actinomyces israelii | bacilli | positif | Filamen bercabang. Bentuk bukan cocci/spiral. |
| bacteroides_fragilis | Bacteroides fragilis | bacilli | negatif | Anaerob strict. Status Gram tetap dapat diamati. |
| bifidobacterium_spp | Bifidobacterium spp. | bacilli | positif | Gram positif meski namanya mengandung "bacterium". |
| clostridium_perfringens | Clostridium perfringens | bacilli | positif | Spor forming, bacillus. |
| enterococcus_faecium | Enterococcus faecium | cocci | positif | Kokus berpasangan dan rantai. |
| enterococcus_faecalis | Enterococcus faecalis | cocci | positif | Kokus berpasangan dan rantai. |
| escherichia_coli | Escherichia coli | bacilli | negatif | Bacillus. |
| fusobacterium_spp | Fusobacterium spp. | bacilli | negatif | Fusiform, bukan spiral. |
| lactobacillus_casei | Lactobacillus casei | bacilli | positif | |
| lactobacillus_crispatus | Lactobacillus crispatus | bacilli | positif | |
| lactobacillus_delbrueckii | Lactobacillus delbrueckii | bacilli | positif | |
| lactobacillus_gasseri | Lactobacillus gasseri | bacilli | positif | |
| lactobacillus_johnsonii_a | Lactobacillus johnsonii (A) | bacilli | positif | |
| lactobacillus_johnsonii_b | Lactobacillus johnsonii (B) | bacilli | positif | |
| lactobacillus_paracasei | Lactobacillus paracasei | bacilli | positif | |
| lactobacillus_plantarum | Lactobacillus plantarum | bacilli | positif | |
| lactobacillus_reuteri | Lactobacillus reuteri | bacilli | positif | |
| lactobacillus_rhamnosus | Lactobacillus rhamnosus | bacilli | positif | |
| lactobacillus_salivarius | Lactobacillus salivarius | bacilli | positif | |
| listeria_monocytogenes | Listeria monocytogenes | bacilli | positif | |
| micrococcus_spp | Micrococcus spp. | cocci | positif | Tetrad dan klaster. |
| neisseria_gonorrhoeae | Neisseria gonorrhoeae | cocci | negatif | Diplococci, bukan spiral. |
| porphyromonas_gingivalis | Porphyromonas gingivalis | bacilli | negatif | Coccobacillus. |
| propionibacterium_acnes | Propionibacterium acnes | bacilli | positif | |
| proteus_spp | Proteus spp. | bacilli | negatif | |
| pseudomonas_aeruginosa | Pseudomonas aeruginosa | bacilli | negatif | |
| staphylococcus_aureus | Staphylococcus aureus | cocci | positif | Klaster. |
| staphylococcus_epidermidis | Staphylococcus epidermidis | cocci | positif | Klaster. |
| staphylococcus_saprophyticus | Staphylococcus saprophyticus | cocci | positif | Klaster. |
| streptococcus_agalactiae | Streptococcus agalactiae | cocci | positif | Rantai. |
| veillonella_spp | Veillonella spp. | cocci | negatif | Anaerob strict. |

Distribusi: cocci 9 spesies, bacilli 23 spesies, spiral 0 spesies. Total 32 spesies.

Candida albicans tidak ada di tabel karena dikeluarkan dari pelatihan (keputusan D7).

**Keputusan pemilik proyek, sudah disetujui:**

1. **Candida albicans dikeluarkan dari pelatihan.** Jamur, bukan bakteri, sehingga tidak
   memenuhi premis sistem. Alasan teknis: sel jamur 5 sampai 10 mikron, bakteri 1 sampai
   2 mikron, sehingga kelas ini akan terpisah mudah oleh backbone dan menaikkan F1-score
   tanpa menunjukkan kemampuan klasifikasi morfologi bakteri. Dicatat di
   `species_map.EXCLUDED_FROM_TRAINING` beserta alasannya.
2. **Acinetobacter baumannii dan Porphyromonas gingivalis dipetakan ke `bacilli`.**
   Coccobacillus adalah bentuk antara, tetapi sumbu batang tetap dominan.
3. **Actinomyces israelii tetap masuk, dipetakan ke `bacilli`.** Bentuknya filamen
   bercabang, dan pada perbesaran 1000x muncul sebagai fragmen batang pendek. Ini satu-satunya
   spesies yang tidak benar-benar muat ke tiga kategori, dan dicatat sebagai keterbatasan.
4. **Bifidobacterium, Actinomyces, dan Lactobacillus adalah Gram positif.** Kesalahan yang
   sering terjadi adalah menganggapnya Gram negatif karena namanya.

Perubahan pada tabel ini memerlukan persetujuan eksplisit pemilik proyek sesuai AGENT.md bagian 2.

## Struktur Berkas

```
BacteriaCV/
├── bacteriacv/
│   ├── __init__.py
│   ├── paths.py                     # lokasi folder, tanpa path absolut
│   ├── config.py                    # hyperparameter terpusat          [BARU]
│   ├── preprocess.py                # C1 pipeline lima tahap         [BARU]
│   ├── model.py                     # C2 backbone + C3 dua head      [BARU]
│   ├── label_map.py                 # C4 lookup table [L]            [BARU]
│   ├── features.py                  # ekstraksi fitur ke .npy        [BARU]
│   ├── evaluate.py                  # C5 F1 makro + akurasi          [BARU]
│   ├── train.py                     # loop pelatihan head             [BARU]
│   ├── infer.py                     # prediksi + overlay             [BARU]
│   └── datasets/
│       ├── species_map.py
│       ├── download.py
│       ├── extract.py
│       └── build_index.py
├── app/
│   ├── __init__.py                                        [BARU]
│   ├── main.py                    # FastAPI POST /predict            [BARU]
│   └── static/
│       └── index.html             # satu halaman statis               [BARU]
├── checkpoints/                    # bobot head, bukan backbone      [BARU]
├── data/
│   ├── raw/                        # sudah ada
│   ├── index.csv                   # sudah ada
│   └── features/                   # .npy hasil ekstraksi             [BARU]
├── docs/
├── scripts/
│   ├── setup_env.ps1
│   ├── check_env.ps1
│   └── run_app.ps1                 # shortcut menjalankan aplikasi    [BARU]
└── tests/
```

Pemisahan berkas mengikuti tanggung jawab tunggal. `preprocess.py` tidak tahu apa itu model,
`model.py` tidak tahu apa itu CSV, `features.py` hanya yang berurusan dengan citra dan fitur.

**Urutan dependensi.** `evaluate.py` dibuat sebelum `train.py` karena `train.py` mengimpor
`f1_macro` darinya. `config.py` dan `label_map.py` tidak mengimpor modul pipeline lain.
Tidak ada import siklik di seluruh rencana ini.

---

## Task 1: Konfigurasi terpusat

**Files:**
- Create: `bacteriacv/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_config.py`:

```python
"""Tes untuk konfigurasi terpusat."""

from __future__ import annotations

import bacteriacv.config as config


def test_image_size_is_224() -> None:
    """SPEC bagian C1 menetapkan resize ke 224."""
    assert config.IMAGE_SIZE == 224


def test_n_shape_classes_is_two() -> None:
    """Keputusan D1: kelas spiral tidak memiliki spesies di DIBaS."""
    assert config.N_SHAPE_CLASSES == 2


def test_shape_labels_exclude_spiral() -> None:
    """Label aktif hanya cocci dan bacilli."""
    assert config.SHAPE_LABELS == ("cocci", "bacilli")


def test_gram_labels_are_positive_and_negative() -> None:
    """Head B tetap dua kelas status Gram."""
    assert config.GRAM_LABELS == ("positif", "negatif")


def test_feature_dim_is_2048() -> None:
    """ResNet-50 menghasilkan 2048-d setelah global average pooling."""
    assert config.FEATURE_DIM == 2048


def test_augment_variants_is_four() -> None:
    """Keputusan D3: empat varian augmentasi pada data latih."""
    assert config.AUGMENT_VARIANTS == 4


def test_seeds_are_pinned_and_distinct() -> None:
    """Seed harus deterministik dan berbeda per tahap."""
    assert config.RANDOM_SEED == config.INDEX_SEED
    assert config.TRAIN_SEED != config.INDEX_SEED


def test_paths_are_inside_project() -> None:
    """Semua folder keluaran harus di dalam root proyek."""
    from bacteriacv.paths import PROJECT_ROOT

    for path in (config.FEATURES_DIR, config.CHECKPOINT_DIR, config.APP_DIR):
        assert path.is_relative_to(PROJECT_ROOT), path
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.config'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/config.py`:

```python
"""Konfigurasi terpusat untuk seluruh pipeline BacteriaCV.

Nilai di sini adalah sumber kebenaran tunggal untuk hyperparameter. Modul lain
tidak boleh mendefinisikan angka yang sama secara lokal.
"""

from __future__ import annotations

from pathlib import Path

from .paths import PROJECT_ROOT

# Citra
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Head A: bentuk sel. Keputusan D1, hanya dua kelas yang terisi di DIBaS.
# Enum bentuk lengkap tetap memuat spiral untuk keperluan pelaporan, tetapi
# spiral tidak pernah menjadi label pelatihan karena tidak ada spesies DIBaS
# dengan bentuk tersebut.
N_SHAPE_CLASSES = 2
SHAPE_LABELS = ("cocci", "bacilli")
SHAPE_LABELS_FULL = ("cocci", "bacilli", "spiral")
SHAPE_UNPOPULATED = "spiral"

# Head B: status Gram
N_GRAM_CLASSES = 2
GRAM_LABELS = ("positif", "negatif")

# Backbone
FEATURE_DIM = 2048
BACKBONE_NAME = "resnet50"
BACKBONE_WEIGHTS = "IMAGENET1K_V2"

# Augmentasi, hanya pada data latih. Keputusan D3.
AUGMENT_VARIANTS = 4
AUGMENT_ROTATION_DEGREES = 15
AUGMENT_SCALE_RANGE = (0.9, 1.1)
AUGMENT_SHIFT_PIXELS = 0.05
AUGMENT_HORIZONTAL_FLIP = True

# Segmentasi
SEGMENT_EROSION_DISK = 2
SEGMENT_DILATION_DISK = 2
SEGMENT_MIN_PEAK_DISTANCE = 5
SEGMENT_MIN_OBJECT_AREA = 20

# Pelatihan
TRAIN_SEED = 1337
INDEX_SEED = 20260203
RANDOM_SEED = INDEX_SEED
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_DELTA = 1e-4

# Lokasi keluaran
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"

# Batas unggahan
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add bacteriacv/config.py tests/test_config.py
git commit -m "feat: add central config for pipeline hyperparameters"
```

---

## Task 2: Lookup table [L]

**Files:**
- Create: `bacteriacv/label_map.py`
- Test: `tests/test_label_map.py`

**CATATAN PENTING:** Task ini hanya boleh dikerjakan setelah pemilik proyek menyetujui draft
lookup table di bagian atas dokumen ini. AGENT.md bagian 2 melarang agen mengubah keputusan
di SPEC tanpa persetujuan eksplisit.

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_label_map.py`:

```python
"""Tes untuk lookup table spesies ke bentuk sel dan status Gram."""

from __future__ import annotations

import pytest

from bacteriacv.config import GRAM_LABELS, SHAPE_LABELS
from bacteriacv.datasets.species_map import (
    EXCLUDED_FROM_TRAINING,
    EXPECTED_TRAINABLE_COUNT,
    TRAINABLE_SPECIES_IDS,
    exclusion_reason,
    is_trainable,
)
from bacteriacv.label_map import (
    LOOKUP,
    SHAPE_TO_INDEX,
    GRAM_TO_INDEX,
    class_weights,
    to_targets,
    unmapped_species,
)


def test_lookup_covers_every_trainable_species() -> None:
    """Semua spesies yang dilatih harus punya entri lookup."""
    assert set(LOOKUP) == set(TRAINABLE_SPECIES_IDS)
    assert len(LOOKUP) == EXPECTED_TRAINABLE_COUNT == 32


def test_lookup_excludes_candida() -> None:
    """Candida albicans dikeluarkan dari pelatihan, jadi tidak ada di lookup."""
    assert "candida_albicans" not in LOOKUP
    assert not is_trainable("candida_albicans")


def test_exclusion_reason_is_documented() -> None:
    """Setiap spesies yang dikecualikan harus punya alasan tertulis."""
    for species_id in EXCLUDED_FROM_TRAINING:
        reason = exclusion_reason(species_id)
        assert reason and len(reason) > 40, species_id


def test_lookup_values_are_valid() -> None:
    """Nilai lookup harus berada di dalam label yang sah."""
    for species_id, (shape, gram) in LOOKUP.items():
        assert shape in SHAPE_LABELS, f"{species_id}: {shape}"
        assert gram in GRAM_LABELS, f"{species_id}: {gram}"


def test_bifidobacterium_is_gram_positive() -> None:
    """Bifidobacterium sering salah dianggap Gram negatif karena namanya."""
    assert LOOKUP["bifidobacterium_spp"][1] == "positif"


def test_no_spiral_species_in_lookup() -> None:
    """Keputusan D1: tidak ada spesies DIBaS dengan bentuk spiral."""
    assert all(shape in SHAPE_LABELS for shape, _ in LOOKUP.values())


def test_fusobacterium_is_bacillus_not_spiral() -> None:
    """Fusobacterium fusiform, bukan spiral."""
    assert LOOKUP["fusobacterium_spp"][0] == "bacilli"


def test_neisseria_is_coccus_not_spiral() -> None:
    """Neisseria diplococci, bukan spiral."""
    assert LOOKUP["neisseria_gonorrhoeae"][0] == "cocci"


def test_johnsonii_variants_have_same_labels() -> None:
    """Dua arsip johnsonii adalah spesies taksonom sama."""
    a = LOOKUP["lactobacillus_johnsonii_a"]
    b = LOOKUP["lactobacillus_johnsonii_b"]
    assert a == b


def test_to_targets_converts_species_ids() -> None:
    """to_targets harus mengubah species_id menjadi indeks numerik."""
    shape, gram = to_targets(["escherichia_coli", "staphylococcus_aureus"])
    assert shape == [SHAPE_TO_INDEX["bacilli"], SHAPE_TO_INDEX["cocci"]]
    assert gram == [GRAM_TO_INDEX["negatif"], GRAM_TO_INDEX["positif"]]


def test_to_targets_rejects_unknown_species() -> None:
    """Spesies di luar tabel harus ditolak, bukan diam-diam dilewati."""
    with pytest.raises(KeyError):
        to_targets(["spesies_yang_tidak_ada"])


def test_unmapped_species_lists_unknowns() -> None:
    """Spesies tak terpetakan harus bisa dilaporkan tanpa exception."""
    unknown = unmapped_species(["escherichia_coli", "bakteri_misterius"])
    assert unknown == ["bakteri_misterius"]


def test_class_weights_from_train_only() -> None:
    """Bobot kelas harus dihitung dari data latih saja."""
    targets = [0, 0, 0, 0, 1, 1]
    weights = class_weights(targets)
    assert len(weights) == 2
    assert weights[0] < weights[1], "kelas minor harus dapat bobot lebih besar"
    assert sum(weights) > 0


def test_class_weights_rejects_empty() -> None:
    """Daftar target kosong tidak menghasilkan bobot yang masuk akal."""
    with pytest.raises(ValueError):
        class_weights([])


def test_class_weights_rejects_single_class() -> None:
    """Hanya satu kelas tidak cukup untuk menghitung bobot tidak seimbang."""
    with pytest.raises(ValueError):
        class_weights([0, 0, 0])
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_label_map.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.label_map'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/label_map.py`:

```python
"""Lookup table pemetaan spesies DIBaS ke bentuk sel dan status Gram.

Modul ini hanya dipakai pada tahap pelatihan. Pada tahap inferensi, label spesies
tidak diketahui sehingga tabel ini tidak boleh dipanggil.

Catatan penting untuk laporan:
1. Bifidobacterium adalah Gram positif meskipun namanya mengandung "bacterium".
2. Neisseria gonorrhoeae berupa diplococci, bukan spiral.
3. Fusobacterium berupa fusiform bacillus, bukan spiral.
4. DIBaS tidak memuat satu pun spesies berbentuk spiral, sehingga Head A
   dilatih hanya pada dua kelas. Enum bentuk tetap memuat spiral sebagai
   penanda kelas yang tidak terisi.
5. Actinomyces israelii berupa filamen bercabang; dipetakan ke bacilli karena
   lebih dekat secara dimensi.
6. Candida albicans adalah jamur, bukan bakteri, dan masuk ke tabel dengan
   catatan eksplisit.
"""

from __future__ import annotations

from collections import Counter

from .config import GRAM_LABELS, SHAPE_LABELS

# Lookup table [L]. Kunci adalah species_id dari datasets/species_map.py.
LOOKUP: dict[str, tuple[str, str]] = {
    "acinetobacter_baumannii": ("bacilli", "negatif"),
    "actinomyces_israelii": ("bacilli", "positif"),
    "bacteroides_fragilis": ("bacilli", "negatif"),
    "bifidobacterium_spp": ("bacilli", "positif"),
    "clostridium_perfringens": ("bacilli", "positif"),
    "enterococcus_faecium": ("cocci", "positif"),
    "enterococcus_faecalis": ("cocci", "positif"),
    "escherichia_coli": ("bacilli", "negatif"),
    "fusobacterium_spp": ("bacilli", "negatif"),
    "lactobacillus_casei": ("bacilli", "positif"),
    "lactobacillus_crispatus": ("bacilli", "positif"),
    "lactobacillus_delbrueckii": ("bacilli", "positif"),
    "lactobacillus_gasseri": ("bacilli", "positif"),
    "lactobacillus_johnsonii_a": ("bacilli", "positif"),
    "lactobacillus_johnsonii_b": ("bacilli", "positif"),
    "lactobacillus_paracasei": ("bacilli", "positif"),
    "lactobacillus_plantarum": ("bacilli", "positif"),
    "lactobacillus_reuteri": ("bacilli", "positif"),
    "lactobacillus_rhamnosus": ("bacilli", "positif"),
    "lactobacillus_salivarius": ("bacilli", "positif"),
    "listeria_monocytogenes": ("bacilli", "positif"),
    "micrococcus_spp": ("cocci", "positif"),
    "neisseria_gonorrhoeae": ("cocci", "negatif"),
    "porphyromonas_gingivalis": ("bacilli", "negatif"),
    "propionibacterium_acnes": ("bacilli", "positif"),
    "proteus_spp": ("bacilli", "negatif"),
    "pseudomonas_aeruginosa": ("bacilli", "negatif"),
    "staphylococcus_aureus": ("cocci", "positif"),
    "staphylococcus_epidermidis": ("cocci", "positif"),
    "staphylococcus_saprophyticus": ("cocci", "positif"),
    "streptococcus_agalactiae": ("cocci", "positif"),
    "veillonella_spp": ("cocci", "negatif"),
}

SHAPE_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(SHAPE_LABELS)}
GRAM_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(GRAM_LABELS)}


def to_targets(species_ids: list[str]) -> tuple[list[int], list[int]]:
    """Ubah daftar species_id menjadi indeks bentuk dan status Gram.

    Args:
        species_ids: Daftar species_id dari data/index.csv.

    Returns:
        Pasangan (indeks_bentuk, indeks_gram).

    Raises:
        KeyError: Bila ada species_id yang tidak ada di LOOKUP.
    """
    shapes: list[int] = []
    grams: list[int] = []
    for species_id in species_ids:
        shape, gram = LOOKUP[species_id]
        shapes.append(SHAPE_TO_INDEX[shape])
        grams.append(GRAM_TO_INDEX[gram])
    return shapes, grams


def unmapped_species(species_ids: list[str]) -> list[str]:
    """Kembalikan species_id yang tidak ada di lookup table.

    Berguna untuk melaporkan citra yang dibuang, sesuai SPEC bagian 3.

    Args:
        species_ids: Daftar species_id yang diperiksa.

    Returns:
        Daftar species_id yang tidak terpetakan, tanpa duplikat dan terurut.
    """
    missing = {species_id for species_id in species_ids if species_id not in LOOKUP}
    return sorted(missing)


def class_weights(targets: list[int], n_classes: int) -> torch.Tensor:
    """Hitung bobot kelas dari frekuensi data latih.

    Bobot tiap kelas adalah kebalikan dari frekuensinya, dinormalisasi agar
    rata-rata bobot sama dengan satu. Hanya data latih boleh dipakai.

    Args:
        targets: Daftar indeks kelas dari data latih.
        n_classes: Jumlah kelas total.

    Returns:
        Tensor bobot sepanjang n_classes.

    Raises:
        ValueError: Bila targets kosong atau hanya berisi satu kelas.
    """
    import torch

    if not targets:
        raise ValueError("Daftar target kosong, bobot kelas tidak dapat dihitung.")
    counts = Counter(targets)
    if len(counts) < 2:
        raise ValueError(
            "Hanya satu kelas pada data latih, bobot tidak dapat dihitung."
        )
    weights = [0.0] * n_classes
    for index, count in counts.items():
        weights[index] = len(targets) / (n_classes * count)
    return torch.tensor(weights, dtype=torch.float32)
```

**CATATAN:** Pada fungsi `class_weights`, ganti baris tanda `import torch` di dalam fungsi
dengan import di modul atas agar tidak mengimpor berulang. Terapkan sebagai berikut:

```python
from __future__ import annotations

from collections import Counter

import torch

from .config import GRAM_LABELS, SHAPE_LABELS
```

lalu ubah signature menjadi `def class_weights(targets: list[int], n_classes: int) -> torch.Tensor:`
dan hapus baris `import torch` di dalam badan fungsi.

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_label_map.py -v`
Expected: 15 passed

- [ ] **Step 5: Jalankan seluruh tes**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: semua lolos

- [ ] **Step 6: Commit**

```bash
git add bacteriacv/label_map.py tests/test_label_map.py
git commit -m "feat: add species lookup table for shape and gram labels"
```

---

## Task 3: Pipeline pra-pemrosesan

**Files:**
- Create: `bacteriacv/preprocess.py`
- Test: `tests/test_preprocess.py`

**CATATAN TEKNIS PENTING:** scikit-image 0.26 sudah men-deprecate `binary_erosion` dan
`binary_dilation`. Harus pakai `morphology.erosion` dan `morphology.dilation`. Verified di
benchmark: pemakaian fungsi lama memunculkan FutureWarning.

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_preprocess.py`:

```python
"""Tes untuk pipeline pra-pemrosesan lima tahap."""

from __future__ import annotations

import numpy as np
import pytest

from bacteriacv import preprocess
from bacteriacv.config import IMAGE_SIZE


def _sample_image(height: int = 1532, width: int = 2048) -> np.ndarray:
    """Buat citra RGB palsu dengan bentuk sel samar."""
    rng = np.random.default_rng(20260203)
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    for _ in range(30):
        cy = rng.integers(50, height - 50)
        cx = rng.integers(50, width - 50)
        radius = rng.integers(8, 20)
        y0, y1 = max(0, cy - radius), min(height, cy + radius)
        x0, x1 = max(0, cx - radius), min(width, cx + radius)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        image[y0:y1, x0:x1][circle] = (140, 60, 160)
    return image


def test_stage_load_returns_rgb_array() -> None:
    """Tahap 1 harus mengembalikan array RGB bertipe uint8."""
    image = _sample_image(300, 400)
    rgb = preprocess.to_rgb(image)
    assert rgb.shape == (300, 400, 3)
    assert rgb.dtype == np.uint8


def test_stage_resize_targets_224() -> None:
    """Tahap 2 harus menghasilkan 224 x 224 sesuai ARCHITECTURE C1."""
    resized = preprocess.resize_image(_sample_image(400, 500), IMAGE_SIZE)
    assert resized.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE)


def test_stage_normalize_output_range() -> None:
    """Tahap 3 harus menghasilkan nilai ternormalisasi, bukan 0 sampai 255."""
    resized = preprocess.resize_image(_sample_image(400, 500), IMAGE_SIZE)
    normalized = preprocess.normalize(resized)
    assert normalized.dtype == np.float32
    assert normalized.max() <= 3.0
    assert normalized.min() >= -3.0


def test_stage_segment_returns_mask_and_ok_flag() -> None:
    """Tahap 4 harus mengembalikan mask dan penanda keberhasilan."""
    resized = preprocess.resize_image(_sample_image(400, 500), IMAGE_SIZE)
    mask, ok = preprocess.segment_cells(resized)
    assert mask.dtype == bool
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert isinstance(ok, bool)


def test_segment_succeeds_on_image_with_cells() -> None:
    """Citra dengan sel synthesetis harus berhasil disegmentasi."""
    resized = preprocess.resize_image(_sample_image(500, 600), IMAGE_SIZE)
    mask, ok = preprocess.segment_cells(resized)
    assert ok is True
    assert mask.sum() > 0


def test_segment_fails_gracefully_on_blank_image() -> None:
    """Citra tanpa sel harus gagal dengan tenang, bukan melempar exception."""
    blank = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    mask, ok = preprocess.segment_cells(blank)
    assert ok is False
    assert mask.sum() == 0


def test_pipeline_returns_five_stages() -> None:
    """preprocess harus mengembalikan lima tahap sesuai ARCHITECTURE bagian 1."""
    result = preprocess.preprocess(_sample_image(400, 500))
    assert result.shape_label == "cocci/bacilli/spiral placeholder"
    assert result.image_tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert result.mask is not None
    assert result.stage_ok["segmentasi"] in (True, False)
    assert "resize" in result.stage_ok
    assert "normalisasi" in result.stage_ok


def test_pipeline_panels_have_ordered_stages() -> None:
    """Panel visualisasi harus berurutan sesuai urutan pipeline."""
    result = preprocess.preprocess(_sample_image(400, 500))
    assert result.panel_names == (
        "original",
        "resized",
        "normalized",
        "segmented",
        "watershed",
    )
    assert len(result.panels) == 5


def test_segmentation_failure_does_not_stop_pipeline() -> None:
    """Kegagalan tahap harus ditandai, alur tetap berjalan sesuai DESIGN bagian 5."""
    blank = np.full((400, 500, 3), 255, dtype=np.uint8)
    result = preprocess.preprocess(blank)
    assert result.stage_ok["segmentasi"] is False
    assert result.image_tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_augment_changes_image_shape_preserved() -> None:
    """Augmentasi harus mempertahankan bentuk tensor."""
    image = np.random.default_rng(1).integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    variants = preprocess.augment(image, count=4)
    assert len(variants) == 4
    for variant in variants:
        assert variant.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert variant.dtype == np.uint8


def test_augment_is_deterministic_with_seed() -> None:
    """Augmentasi dengan seed sama harus menghasilkan citra sama."""
    image = np.random.default_rng(1).integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    first = preprocess.augment(image, count=4, seed=99)
    second = preprocess.augment(image, count=4, seed=99)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_augment_zero_count_returns_empty() -> None:
    """Jumlah nol harus menghasilkan daftar kosong, bukan error."""
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    assert preprocess.augment(image, count=0) == []


def test_load_image_rejects_missing_file(tmp_path) -> None:
    """Berkas yang tidak ada harus ditolak dengan pesan jelas."""
    with pytest.raises(FileNotFoundError):
        preprocess.load_image(tmp_path / "tidak_ada.tif")


def test_load_image_rejects_non_image(tmp_path) -> None:
    """Berkas yang bukan gambar harus ditolak."""
    path = tmp_path / "catatan.txt"
    path.write_text("bukan citra", encoding="utf-8")
    with pytest.raises(ValueError):
        preprocess.load_image(path)
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_preprocess.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.preprocess'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/preprocess.py`:

```python
"""Pipeline pra-pemrosesan lima tahap untuk citra mikroskopis bakteri.

Urutan tahap sesuai ARCHITECTURE bagian 1:
1. Pemuatan citra
2. Resize ke 224 x 224
3. Normalisasi intensitas
4. Segmentasi morfologis (erosi, dilasi)
5. Watershed untuk memisahkan sel bertumpuk

Augmentasi hanya dijalankan pada tahap latih. Kegagalan tahap segmentasi
tidak menghentikan pipeline, tahap ditandai gagal dan panel kosong, sesuai
DESIGN bagian 5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import morphology
from skimage.measure import regionprops
from skimage.segmentation import watershed

from .config import (
    AUGMENT_HORIZONTAL_FLIP,
    AUGMENT_ROTATION_DEGREES,
    AUGMENT_SCALE_RANGE,
    AUGMENT_SHIFT_PIXELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    SEGMENT_DILATION_DISK,
    SEGMENT_EROSION_DISK,
    SEGMENT_MIN_OBJECT_AREA,
    SEGMENT_MIN_PEAK_DISTANCE,
)

PANEL_NAMES = ("original", "resized", "normalized", "segmented", "watershed")

SEGMENTATION_FAILED_MESSAGE = "Segmentasi tidak berhasil."


@dataclass
class PreprocessResult:
    """Keluaran pipeline pra-pemrosesan.

    Attributes:
        image_tensor: Tensor 3 x 224 x 224 siap backbone.
        mask: Mask biner sel, None bila segmentasi gagal.
        labels: Label watershed per objek, None bila segmentasi gagal.
        panels: Lima citra RGB untuk visualisasi, berurutan sesuai PANEL_NAMES.
        stage_ok: Status keberhasilan tiap tahap.
        object_count: Jumlah objek sel terdeteksi.
        shape_label: Placeholder label bentuk, diisi model pada tahap inferensi.
    """

    image_tensor: torch.Tensor
    mask: np.ndarray | None
    labels: np.ndarray | None
    panels: tuple[np.ndarray, ...]
    stage_ok: dict[str, bool]
    object_count: int
    shape_label: str = "cocci/bacilli/spiral placeholder"


def load_image(path: Path | str) -> np.ndarray:
    """Baca berkas citra dari disk sebagai array RGB.

    Args:
        path: Lokasi berkas citra.

    Returns:
        Array RGB bertipe uint8.

    Raises:
        FileNotFoundError: Bila berkas tidak ada.
        ValueError: Bila berkas tidak dapat dibaca sebagai citra.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Citra tidak ditemukan: {path.name}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Citra tidak dapat dibaca: {path.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Pastikan citra berupa array RGB bertipe uint8.

    Args:
        image: Array citra sumber.

    Returns:
        Array RGB uint8.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 4:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGRA2RGB)
    return array.astype(np.uint8)


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize citra ke ukuran persegi sesuai ARCHITECTURE C1.

    Args:
        image: Array citra sumber.
        size: Sisi target dalam piksel.

    Returns:
        Array RGB dengan sisi size.
    """
    return cv2.resize(
        to_rgb(image), (size, size), interpolation=cv2.INTER_AREA
    )


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalisasi intensitas dengan statistik ImageNet.

    Args:
        image: Array RGB uint8.

    Returns:
        Array float32 dengan rentang sekitar -2 sampai 2.
    """
    array = to_rgb(image).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (array - mean) / std


def segment_cells(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Segmentasi morfologis sel dengan ambang Otsu, erosi, dilasi, dan watershed.

    Args:
        image: Array RGB uint8 pada 224 x 224.

    Returns:
        Pasangan (mask_biner, berhasil). Bila gagal, mask kosong dan
        berhasil bernilai False.
    """
    array = to_rgb(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.max() == 0:
        return np.zeros(gray.shape, dtype=bool), False

    binary_bool = binary > 0
    if binary_bool.mean() > 0.6:
        binary_bool = ~binary_bool

    eroded = morphology.erosion(binary_bool, morphology.disk(SEGMENT_EROSION_DISK))
    dilated = morphology.dilation(eroded, morphology.disk(SEGMENT_DILATION_DISK))
    if dilated.sum() == 0:
        return np.zeros(gray.shape, dtype=bool), False

    distance = ndi.distance_transform_edt(dilated)
    coordinates = peak_local_max(
        distance, min_distance=SEGMENT_MIN_PEAK_DISTANCE, labels=dilated
    )
    if len(coordinates) == 0:
        return dilated, True

    markers = np.zeros(dilated.shape, dtype=np.int32)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    labels = watershed(-distance, markers, mask=dilated)

    keep = np.zeros_like(labels, dtype=bool)
    for region in regionprops(labels):
        if region.area >= SEGMENT_MIN_OBJECT_AREA:
            keep[labels == region.label] = True

    if keep.sum() == 0:
        return dilated, True
    return keep, True


def _to_tensor(normalized: np.ndarray) -> torch.Tensor:
    """Ubah array HWC ternormalisasi menjadi tensor CHW."""
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()


def preprocess(image: np.ndarray | Path | str) -> PreprocessResult:
    """Jalankan pipeline lima tahap.

    Args:
        image: Array citra, path berkas, atau objek bytes.

    Returns:
        PreprocessResult berisi tensor, mask, panel visualisasi, dan status tiap tahap.
    """
    if isinstance(image, (str, Path)):
        array = load_image(image)
    else:
        array = to_rgb(np.asarray(image))

    stage_ok: dict[str, bool] = {}

    resized = resize_image(array)
    stage_ok["resize"] = resized.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE)

    normalized = normalize(resized)
    stage_ok["normalisasi"] = bool(np.isfinite(normalized).all())

    mask, segment_ok = segment_cells(resized)
    stage_ok["segmentasi"] = segment_ok

    if segment_ok and mask.any():
        dilated = mask
        distance = ndi.distance_transform_edt(dilated)
        coordinates = peak_local_max(
            distance, min_distance=SEGMENT_MIN_PEAK_DISTANCE, labels=dilated
        )
        markers = np.zeros(dilated.shape, dtype=np.int32)
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        labels = watershed(-distance, markers, mask=dilated) if len(coordinates) else dilated.astype(np.int32)
        stage_ok["watershed"] = True
        object_count = int(labels.max()) if labels.max() > 0 else 0
    else:
        labels = None
        stage_ok["watershed"] = False
        object_count = 0

    panels = (
        array,
        resized,
        np.clip(normalized * 0.5 + 0.5, 0, 1).astype(np.float32),
        (mask * 255).astype(np.uint8) if mask is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8),
        _label_panel(labels, resized) if labels is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
    )

    return PreprocessResult(
        image_tensor=_to_tensor(normalized),
        mask=mask if segment_ok else None,
        labels=labels if segment_ok else None,
        panels=panels,
        stage_ok=stage_ok,
        object_count=object_count,
    )


def _label_panel(labels: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Bentuk citra RGB dengan batas objek watershed di atas citra dasar."""
    panel = base.copy()
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    panel[boundaries] = (255, 255, 255)
    return panel


def augment(image: np.ndarray, count: int, seed: int | None = None) -> list[np.ndarray]:
    """Buat varian augmentasi untuk data latih saja.

    Args:
        image: Array RGB uint8.
        count: Jumlah varian yang diminta.
        seed: Seed agar hasil dapat direproduksi.

    Returns:
        Daftar varian citra, masing-masing dengan bentuk dan tipe sama seperti sumber.
    """
    if count <= 0:
        return []
    rng = random.Random(seed)
    array = to_rgb(image)
    variants: list[np.ndarray] = []
    for _ in range(count):
        output = array.copy()
        angle = rng.uniform(-AUGMENT_ROTATION_DEGREES, AUGMENT_ROTATION_DEGREES)
        scale = rng.uniform(*AUGMENT_SCALE_RANGE)
        shift_x = rng.uniform(-AUGMENT_SHIFT_PIXELS, AUGMENT_SHIFT_PIXELS) * array.shape[1]
        shift_y = rng.uniform(-AUGMENT_SHIFT_PIXELS, AUGMENT_SHIFT_PIXELS) * array.shape[0]
        matrix = cv2.getRotationMatrix2D(
            (array.shape[1] / 2, array.shape[0] / 2), angle, scale
        )
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        output = cv2.warpAffine(
            output, matrix, (array.shape[1], array.shape[0]), borderMode=cv2.BORDER_REFLECT_101
        )
        if AUGMENT_HORIZONTAL_FLIP and rng.random() < 0.5:
            output = cv2.flip(output, 1)
        variants.append(output)
    return variants
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_preprocess.py -v`
Expected: 14 passed

- [ ] **Step 5: Verifikasi tanpa FutureWarning dari skimage**

Run: `.venv\Scripts\python.exe -W error::FutureWarning -c "from bacteriacv import preprocess; print('ok')"`
Expected: `ok`

Kalau ada FutureWarning, pastikan memakai `morphology.erosion` dan `morphology.dilation`,
bukan `binary_erosion` dan `binary_dilation`.

- [ ] **Step 6: Commit**

```bash
git add bacteriacv/preprocess.py tests/test_preprocess.py
git commit -m "feat: add five-stage preprocessing pipeline"
```

---

## Task 4: Model ResNet-50 dengan dua head

**Files:**
- Create: `bacteriacv/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_model.py`:

```python
"""Tes untuk model dua head dengan backbone dibekukan."""

from __future__ import annotations

import pytest
import torch

from bacteriacv.config import FEATURE_DIM
from bacteriacv.model import BacteriaNet, build_model, load_checkpoint


def test_backbone_is_frozen() -> None:
    """ARCHITECTURE C2: seluruh parameter backbone dibekukan."""
    model = build_model(pretrained=False)
    backbone_params = list(model.backbone.parameters())
    assert backbone_params, "backbone harus punya parameter"
    assert all(not param.requires_grad for param in backbone_params)


def test_only_heads_are_trainable() -> None:
    """Hanya head yang boleh diperbarui saat pelatihan."""
    model = build_model(pretrained=False)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable, "head harus bisa dilatih"
    assert all(name.startswith("head") for name in trainable), trainable


def test_head_a_output_is_two_classes() -> None:
    """Keputusan D1: Head A punya dua kelas bentuk."""
    model = build_model(pretrained=False)
    features = torch.randn(4, FEATURE_DIM)
    logits = model.head_a(features)
    assert logits.shape == (4, 2)


def test_head_b_output_is_two_classes() -> None:
    """Head B punya dua kelas status Gram."""
    model = build_model(pretrained=False)
    features = torch.randn(4, FEATURE_DIM)
    logits = model.head_b(features)
    assert logits.shape == (4, 2)


def test_head_a_probabilities_sum_to_one() -> None:
    """Head A memakai softmax sehingga probabilitas berjumlah satu."""
    model = build_model(pretrained=False).eval()
    with torch.inference_mode():
        probs = model.head_a(torch.randn(8, FEATURE_DIM)).softmax(dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_head_b_probabilities_valid_sigmoid() -> None:
    """Head B memakai sigmoid sehingga probabilitas berada di 0 sampai 1."""
    model = build_model(pretrained=False).eval()
    with torch.inference_mode():
        probs = model.head_b(torch.randn(8, FEATURE_DIM)).sigmoid()
    assert (probs >= 0).all() and (probs <= 1).all()


def test_head_b_is_independent_from_head_a() -> None:
    """Dua head tidak boleh saling memengaruhi parameter."""
    model = build_model(pretrained=False)
    a_params = {id(p) for p in model.head_a.parameters()}
    b_params = {id(p) for p in model.head_b.parameters()}
    assert not (a_params & b_params)


def test_extract_features_gives_2048_dim() -> None:
    """Backbone menghasilkan vektor 2048-d sesuai ARCHITECTURE bagian 2."""
    model = build_model(pretrained=False).eval()
    batch = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        features = model.extract_features(batch)
    assert features.shape == (2, FEATURE_DIM)


def test_predict_returns_confidences_in_range() -> None:
    """Confidence harus berada di rentang 0 sampai 1."""
    model = build_model(pretrained=False).eval()
    with torch.inference_mode():
        shape, shape_conf, gram, gram_conf = model.predict(torch.randn(4, FEATURE_DIM))
    assert shape.shape == (4,)
    assert shape_conf.shape == (4,)
    assert ((shape_conf >= 0) & (shape_conf <= 1)).all()
    assert gram.shape == (4,)
    assert gram_conf.shape == (4,)


def test_load_checkpoint_rejects_missing_file(tmp_path) -> None:
    """Checkpoint yang tidak ada harus ditolak dengan pesan jelas."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "tidak_ada.pt")


def test_load_checkpoint_restores_weights(tmp_path) -> None:
    """Checkpoint yang dimuat harus mengembalikan bobot yang sama."""
    model = build_model(pretrained=False)
    target = tmp_path / "head.pt"
    torch.save({"head_a": model.head_a.state_dict(), "head_b": model.head_b.state_dict()}, target)

    other = build_model(pretrained=False)
    load_checkpoint(other, target)

    for a, b in zip(model.head_a.parameters(), other.head_a.parameters()):
        assert torch.allclose(a, b)


def test_checkpoint_does_not_contain_backbone() -> None:
    """Hanya head yang disimpan, backbone diambil ulang dari torchvision."""
    model = build_model(pretrained=False)
    state = {"head_a": model.head_a.state_dict(), "head_b": model.head_b.state_dict()}
    assert all("backbone" not in key for key in state["head_a"])
    assert all("backbone" not in key for key in state["head_b"])
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.model'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/model.py`:

```python
"""Backbone ResNet-50 dan dua classification head.

Head A memakai softmax dua kelas bentuk sel. Head B memakai sigmoid dua kelas
status Gram. Backbone dibekukan sehingga hanya head yang dilatih.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .config import (
    BACKBONE_NAME,
    BACKBONE_WEIGHTS,
    FEATURE_DIM,
    GRAM_LABELS,
    N_GRAM_CLASSES,
    N_SHAPE_CLASSES,
    SHAPE_LABELS,
)


class BacteriaNet(nn.Module):
    """ResNet-50 beku dengan dua head klasifikasi.

    Attributes:
        backbone: ResNet-50 dengan layer klasifikasi dilepas.
        head_a: Linear(2048, 2) untuk bentuk sel.
        head_b: Linear(2048, 2) untuk status Gram.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Bangun model.

        Args:
            pretrained: Bila True, muat bobot ImageNet.
        """
        super().__init__()
        if BACKBONE_NAME != "resnet50":
            raise ValueError(f"Backbone {BACKBONE_NAME} belum didukung.")
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        network = resnet50(weights=weights)
        network.fc = nn.Identity()
        self.backbone = network
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.head_a = nn.Linear(FEATURE_DIM, N_SHAPE_CLASSES)
        self.head_b = nn.Linear(FEATURE_DIM, N_GRAM_CLASSES)

    def train(self, mode: bool = True) -> "BacteriaNet":
        """Set mode training, tapi backbone tetap eval karena dibekukan."""
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Ambil vektor fitur 2048-d dari batch citra.

        Args:
            batch: Tensor B x 3 x 224 x 224.

        Returns:
            Tensor B x 2048.
        """
        with torch.inference_mode():
            return self.backbone(batch)

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prediksi bentuk dan status Gram dari vektor fitur.

        Args:
            features: Tensor B x 2048.

        Returns:
            Pasangan (label_bentuk, confidence_bentuk, label_gram, confidence_gram).
        """
        with torch.inference_mode():
            shape_logits = self.head_a(features)
            gram_logits = self.head_b(features)
            shape_probs = shape_logits.softmax(dim=-1)
            gram_probs = gram_logits.sigmoid()
            shape_index = shape_probs.argmax(dim=-1)
            gram_index = (gram_probs > 0.5).long()
            shape_conf = shape_probs.max(dim=-1).values
            gram_conf = torch.where(
                gram_index == 1, gram_probs[:, 1], gram_probs[:, 0]
            )
        return shape_index, shape_conf, gram_index, gram_conf


def build_model(pretrained: bool = True) -> BacteriaNet:
    """Bangun BacteriaNet.

    Args:
        pretrained: Bila True, muat bobot ImageNet.

    Returns:
        Instans BacteriaNet siap pakai.
    """
    return BacteriaNet(pretrained=pretrained)


def load_checkpoint(model: BacteriaNet, path: Path | str) -> BacteriaNet:
    """Muat bobot head dari checkpoint.

    Backbone tidak disimpan di checkpoint karena diambil ulang dari torchvision.

    Args:
        model: Model tujuan.
        path: Lokasi berkas checkpoint.

    Returns:
        Model dengan head yang sudah dimuat.

    Raises:
        FileNotFoundError: Bila checkpoint tidak ada.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path.name}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.head_a.load_state_dict(state["head_a"])
    model.head_b.load_state_dict(state["head_b"])
    return model


def save_checkpoint(model: BacteriaNet, path: Path | str) -> None:
    """Simpan bobot head saja.

    Args:
        model: Model yang disimpan head-nya.
        path: Lokasi berkas tujuan.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_a": model.head_a.state_dict(), "head_b": model.head_b.state_dict()}, path)


def label_shape(index: int) -> str:
    """Ubah indeks kelas bentuk menjadi label.

    Args:
        index: Indeks kelas.

    Returns:
        Nama kelas, atau "tidak_diketahui" bila indeks di luar rentang.
    """
    if 0 <= index < len(SHAPE_LABELS):
        return SHAPE_LABELS[index]
    return "tidak_diketahui"


def label_gram(index: int) -> str:
    """Ubah indeks kelas status Gram menjadi label.

    Args:
        index: Indeks kelas.

    Returns:
        Nama kelas, atau "tidak_diketahui" bila indeks di luar rentang.
    """
    if 0 <= index < len(GRAM_LABELS):
        return GRAM_LABELS[index]
    return "tidak_diketahui"
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add bacteriacv/model.py tests/test_model.py
git commit -m "feat: add frozen resnet50 backbone with dual heads"
```

---

## Task 5: Ekstraksi fitur ke disk

**Files:**
- Create: `bacteriacv/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_features.py`:

```python
"""Tes untuk ekstraksi fitur backbone."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from bacteriacv import features as features_module
from bacteriacv.config import FEATURE_DIM
from bacteriacv.features import (
    build_manifest,
    extract_for_index,
    load_feature_store,
    save_feature_store,
)


def test_feature_store_shapes(tmp_path) -> None:
    """Feature store harus memuat matriks fitur dan daftar label."""
    matrix = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    species = ["a", "b", "c"]
    save_feature_store(matrix, species, tmp_path)

    loaded, loaded_species = load_feature_store(tmp_path)
    assert loaded.shape == (3, FEATURE_DIM)
    assert loaded_species == species


def test_feature_store_preserves_values(tmp_path) -> None:
    """Nilai fitur harus pulih persis setelah disimpan dan dimuat."""
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((5, FEATURE_DIM)).astype(np.float32)
    species = [f"s{i}" for i in range(5)]
    save_feature_store(matrix, species, tmp_path)

    loaded, _ = load_feature_store(tmp_path)
    assert np.allclose(loaded, matrix, atol=1e-6)


def test_feature_store_dimension_is_validated(tmp_path) -> None:
    """Dimensi fitur yang salah harus ditolak saat pemuatan."""
    save_feature_store(np.zeros((2, 7), dtype=np.float32), ["a", "b"], tmp_path)
    with pytest.raises(ValueError):
        load_feature_store(tmp_path)


def test_feature_store_length_mismatch_rejected(tmp_path) -> None:
    """Jumlah label harus sama dengan jumlah baris fitur."""
    save_feature_store(np.zeros((3, FEATURE_DIM), dtype=np.float32), ["a"], tmp_path)
    with pytest.raises(ValueError):
        load_feature_store(tmp_path)


def test_load_missing_store_raises(tmp_path) -> None:
    """Feature store yang belum ada harus ditolak dengan pesan jelas."""
    with pytest.raises(FileNotFoundError):
        load_feature_store(tmp_path)


def test_manifest_records_path_and_split(tmp_path) -> None:
    """Manifest harus memuat path, spesies, split, dan fold."""
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "species_id", "split", "fold"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow({"path": "p1", "species_id": "s1", "split": "train", "fold": 0})
        writer.writerow({"path": "p2", "species_id": "s1", "split": "test", "fold": -1})

    manifest = build_manifest(index_path)
    assert len(manifest) == 2
    assert manifest[0]["path"] == "p1"
    assert manifest[1]["split"] == "test"


def test_extract_skips_test_split(tmp_path) -> None:
    """Ekstraksi untuk ekstraksi fitur tidak boleh memakai data uji tanpa alasan."""
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "species_id", "split", "fold"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow({"path": "p1", "species_id": "s1", "split": "train", "fold": 0})
        writer.writerow({"path": "p2", "species_id": "s1", "split": "val", "fold": -1})
        writer.writerow({"path": "p3", "species_id": "s1", "split": "test", "fold": -1})

    manifest = build_manifest(index_path)
    selected = [row for row in manifest if row["split"] in {"train", "val"}]
    assert len(selected) == 2
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_features.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.features'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/features.py`:

```python
"""Ekstraksi vektor fitur backbone ke file .npy.

Karena backbone dibekukan, fitur setiap citra tidak berubah antar epoch. Fitur
diekstrak sekali lalu dipakai berulang saat melatih head. Matriks 672 x 2048
float32 berukuran 5,5 MB, cukup kecil untuk dimuat di memory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from .config import AUGMENT_VARIANTS, FEATURE_DIM, FEATURES_DIR, IMAGE_SIZE, TRAIN_SEED
from .model import build_model
from .paths import IMAGES_DIR, INDEX_PATH, PROJECT_ROOT
from .preprocess import preprocess


def build_manifest(index_path: Path = INDEX_PATH) -> list[dict[str, str]]:
    """Baca data/index.csv menjadi daftar baris.

    Args:
        index_path: Lokasi index.csv.

    Returns:
        Daftar dict dengan kunci path, species_id, split, fold.
    """
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _augmented_paths(path: Path, variants: int, seed: int) -> list[Path]:
    """Buat salinan sementara untuk setiap varian augmentasi.

    Augmentasi harus terjadi pada level citra, sebelum backbone membekukan
    fitur. Varian disimpan ke folder sementara lalu dihapus setelah dipakai.
    """
    from .preprocess import augment, load_image, resize_image

    image = resize_image(load_image(path))
    variants_images = augment(image, count=variants, seed=seed)
    temporary: list[Path] = []
    scratch = FEATURES_DIR / "_augment_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for index, variant in enumerate(variants_images):
        import cv2

        target = scratch / f"{path.stem}_aug{index}.png"
        cv2.imwrite(str(target), cv2.cvtColor(variant, cv2.COLOR_RGB2BGR))
        temporary.append(target)
    return temporary


def extract_for_index(
    manifest: list[dict[str, str]],
    model=None,
    augment_train: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Ekstrak fitur untuk seluruh baris manifest.

    Args:
        manifest: Daftar baris dari build_manifest.
        model: Model BacteriaNet. Bila None, dibangun dengan pretrained=True.
        augment_train: Bila True, data latih diekstrak dengan AUGMENT_VARIANTS varian.

    Returns:
        Pasangan (matriks fitur, daftar species_id).
    """
    if model is None:
        model = build_model(pretrained=True)
    model.eval()

    rows: list[np.ndarray] = []
    species_ids: list[str] = []
    scratch = FEATURES_DIR / "_augment_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(manifest, start=1):
        path = PROJECT_ROOT / row["path"]
        is_train = row["split"] == "train"
        variants = AUGMENT_VARIANTS if (is_train and augment_train) else 1

        if variants > 1:
            targets = _augmented_paths(path, variants, seed=TRAIN_SEED + index)
        else:
            targets = [path]

        try:
            for target in targets:
                result = preprocess(target)
                batch = result.image_tensor.unsqueeze(0)
                features = model.extract_features(batch)
                rows.append(features[0].numpy().astype(np.float32))
                species_ids.append(row["species_id"])
        finally:
            if variants > 1:
                for target in targets:
                    target.unlink(missing_ok=True)

        if index % 50 == 0:
            print(f"  {index}/{len(manifest)} selesai", flush=True)

    if scratch.is_dir():
        for leftover in scratch.iterdir():
            leftover.unlink(missing_ok=True)
        scratch.rmdir()

    return np.vstack(rows), species_ids


def save_feature_store(matrix: np.ndarray, species_ids: list[str], directory: Path) -> None:
    """Simpan matriks fitur dan daftar label ke disk.

    Args:
        matrix: Matriks fitur B x 2048 float32.
        species_ids: Daftar species_id sepanjang B.
        directory: Folder tujuan.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "features.npy", matrix.astype(np.float32))
    with (directory / "species_ids.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for species_id in species_ids:
            handle.write(f"{species_id}\n")


def load_feature_store(directory: Path) -> tuple[np.ndarray, list[str]]:
    """Muat feature store dari disk.

    Args:
        directory: Folder yang berisi features.npy dan species_ids.txt.

    Returns:
        Pasangan (matriks fitur, daftar species_id).

    Raises:
        FileNotFoundError: Bila berkas tidak ada.
        ValueError: Bila dimensi atau panjang data tidak konsisten.
    """
    directory = Path(directory)
    matrix_path = directory / "features.npy"
    ids_path = directory / "species_ids.txt"
    if not matrix_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(
            f"Feature store belum ada di {directory.name}. Jalankan ekstraksi fitur dulu."
        )
    matrix = np.load(matrix_path)
    species_ids = ids_path.read_text(encoding="utf-8").split()
    if matrix.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Dimensi fitur {matrix.shape[1]} tidak sesuai {FEATURE_DIM}."
        )
    if matrix.shape[0] != len(species_ids):
        raise ValueError(
            f"Jumlah fitur {matrix.shape[0]} tidak sama dengan jumlah label {len(species_ids)}."
        )
    return matrix, species_ids


def main(argv: list[str] | None = None) -> int:
    """Titik masuk baris perintah."""
    parser = argparse.ArgumentParser(description="Ekstrak fitur backbone ke .npy")
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--out", type=Path, default=FEATURES_DIR)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args(argv)

    manifest = build_manifest(args.index)
    if not manifest:
        print("GAGAL: index.csv kosong.", file=sys.stderr)
        return 1

    print(f"Mengekstrak fitur untuk {len(manifest)} citra...")
    matrix, species_ids = extract_for_index(manifest, augment_train=not args.no_augment)
    save_feature_store(matrix, species_ids, args.out)
    print(f"Selesai. Fitur {matrix.shape} disimpan di {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_features.py -v`
Expected: 7 passed

- [ ] **Step 5: Jalankan ekstraksi fitur pada dataset sebenarnya**

Run: `.venv\Scripts\python.exe -m bacteriacv.features`
Expected: keluarannya seperti
```
Mengekstrak fitur untuk 672 citra...
  50/672 selesai
  ...
Selesai. Fitur (2065, 2048) disimpan di features
```
Jumlah baris harus 672 untuk citra asli, ditambah 469 baris data latih dikali 3 varian
tambahan dari `AUGMENT_VARIANTS = 4`. Periksa angka aktual yang tercetak dan catat
di laporan.

- [ ] **Step 6: Commit**

```bash
git add bacteriacv/features.py tests/test_features.py
git commit -m "feat: extract frozen backbone features to npy"
```

---

## Task 6: Modul evaluasi

**Files:**
- Create: `bacteriacv/evaluate.py`
- Test: `tests/test_evaluate.py`

Task ini mendahului Task 7 karena `train.py` mengimpor `f1_macro` dari modul ini.

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_evaluate.py`:

```python
"""Tes untuk perhitungan F1-score makro dan akurasi."""

from __future__ import annotations

import pytest

from bacteriacv.evaluate import accuracy, confusion_matrix, f1_macro, per_class_report


def test_f1_perfect_prediction() -> None:
    """Prediksi sempurna memberi F1 makro satu."""
    assert f1_macro([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0


def test_f1_all_wrong() -> None:
    """Prediksi salah total memberi F1 nol."""
    assert f1_macro([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0


def test_f1_empty_input_raises() -> None:
    """Input kosong tidak menghasilkan skor yang terdefinisi."""
    with pytest.raises(ValueError):
        f1_macro([], [])


def test_f1_length_mismatch_raises() -> None:
    """Panjang target dan prediksi harus sama."""
    with pytest.raises(ValueError):
        f1_macro([0, 1], [0])


def test_f1_averages_over_classes_not_samples() -> None:
    """Kelas minor tidak boleh tenggelam oleh kelas majoritas."""
    score = f1_macro([0] * 10 + [1], [0] * 10 + [0])
    assert 0.0 < score < 1.0


def test_accuracy_correct() -> None:
    """Akurasi adalah rasio benar terhadap total."""
    assert accuracy([0, 1, 0, 1], [0, 1, 0, 0]) == 0.75


def test_accuracy_empty_raises() -> None:
    """Input kosong tidak menghasilkan akurasi yang terdefinisi."""
    with pytest.raises(ValueError):
        accuracy([], [])


def test_confusion_matrix_counts_diagonal() -> None:
    """Diagonal menyimpan jumlah prediksi benar."""
    matrix = confusion_matrix([0, 0, 1], [0, 1, 1], n_classes=2)
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1


def test_per_class_report_has_all_classes() -> None:
    """Laporan per kelas memuat seluruh kelas."""
    report = per_class_report([0, 1, 0, 1], [0, 1, 1, 1], labels=("cocci", "bacilli"))
    assert set(report) == {"cocci", "bacilli"}
    assert report["cocci"]["f1"] is not None


def test_per_class_report_omits_unseen_class() -> None:
    """Kelas yang tidak pernah muncul ditandai None, bukan nol."""
    report = per_class_report([0, 0], [0, 0], labels=("cocci", "bacilli"))
    assert report["bacilli"]["f1"] is None
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.evaluate'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/evaluate.py`:

```python
"""Perhitungan F1-score makro dan akurasi per head.

Metrik utama adalah F1-score makro sesuai SPEC bagian 6. F1 makro dipilih karena
memberi bobot sama untuk tiap kelas, sehingga kelas minor tidak tenggelam oleh
kelas majoritas.
"""

from __future__ import annotations

from collections.abc import Sequence


def _validate(
    true_labels: Sequence[int], pred_labels: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Validasi kedua daftar label.

    Args:
        true_labels: Label sebenarnya.
        pred_labels: Label prediksi.

    Returns:
        Pasangan dua daftar yang sudah jadi list.

    Raises:
        ValueError: Bila salah satu kosong atau panjangnya berbeda.
    """
    true_list = list(true_labels)
    pred_list = list(pred_labels)
    if not true_list or not pred_list:
        raise ValueError("Daftar label tidak boleh kosong.")
    if len(true_list) != len(pred_list):
        raise ValueError("Panjang label sebenarnya dan prediksi harus sama.")
    return true_list, pred_list


def _prf(
    true_list: list[int], pred_list: list[int], label: int
) -> tuple[float, float, float]:
    """Hitung precision, recall, dan F1 untuk satu kelas."""
    true_positive = sum(1 for a, b in zip(true_list, pred_list) if a == label and b == label)
    false_positive = sum(1 for a, b in zip(true_list, pred_list) if a != label and b == label)
    false_negative = sum(1 for a, b in zip(true_list, pred_list) if a == label and b != label)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    return precision, recall, f1


def confusion_matrix(
    true_labels: Sequence[int], pred_labels: Sequence[int], n_classes: int
) -> list[list[int]]:
    """Bangun confusion matrix.

    Args:
        true_labels: Label sebenarnya.
        pred_labels: Label prediksi.
        n_classes: Jumlah kelas.

    Returns:
        Matriks n_classes x n_classes dalam bentuk daftar daftar.
    """
    true_list, pred_list = _validate(true_labels, pred_labels)
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for actual, predicted in zip(true_list, pred_list):
        if 0 <= actual < n_classes and 0 <= predicted < n_classes:
            matrix[actual][predicted] += 1
    return matrix


def f1_macro(true_labels: Sequence[int], pred_labels: Sequence[int]) -> float:
    """Hitung F1-score makro.

    Args:
        true_labels: Label sebenarnya.
        pred_labels: Label prediksi.

    Returns:
        Rata-rata F1 antar kelas yang muncul di label sebenarnya atau prediksi.

    Raises:
        ValueError: Bila input kosong atau panjangnya berbeda.
    """
    true_list, pred_list = _validate(true_labels, pred_labels)
    classes = sorted(set(true_list) | set(pred_list))
    scores = [_prf(true_list, pred_list, label)[2] for label in classes]
    return sum(scores) / len(scores)


def accuracy(true_labels: Sequence[int], pred_labels: Sequence[int]) -> float:
    """Hitung akurasi sebagai metrik pelengkap.

    Args:
        true_labels: Label sebenarnya.
        pred_labels: Label prediksi.

    Returns:
        Rasio prediksi benar terhadap total.

    Raises:
        ValueError: Bila input kosong atau panjangnya berbeda.
    """
    true_list, pred_list = _validate(true_labels, pred_labels)
    correct = sum(1 for a, b in zip(true_list, pred_list) if a == b)
    return correct / len(true_list)


def per_class_report(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    labels: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    """Susun laporan per kelas untuk keperluan laporan skripsi.

    Args:
        true_labels: Label sebenarnya berupa indeks kelas.
        pred_labels: Label prediksi berupa indeks kelas.
        labels: Nama label sesuai indeks kelas.

    Returns:
        Dictionary nama kelas ke metrik. Kelas yang tidak muncul di label
        sebenarnya diberi nilai None agar tidak disalahartikan sebagai nol.
    """
    true_list, pred_list = _validate(true_labels, pred_labels)
    report: dict[str, dict[str, float | None]] = {}
    for index, name in enumerate(labels):
        support = sum(1 for value in true_list if value == index)
        if support == 0:
            report[name] = {"precision": None, "recall": None, "f1": None, "support": 0}
            continue
        precision, recall, f1 = _prf(true_list, pred_list, index)
        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return report
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -v`
Expected: 10 passed

- [ ] **Step 5: Cross-check dengan scikit-learn**

Buat `check_f1.py` di folder proyek:

```python
"""Bandingkan f1_macro dengan sklearn untuk memastikan rumus benar."""

import numpy as np
from sklearn.metrics import f1_score

from bacteriacv.evaluate import f1_macro

rng = np.random.default_rng(5)
mismatch = 0
for _ in range(200):
    true = rng.integers(0, 2, 40).tolist()
    pred = rng.integers(0, 2, 40).tolist()
    if abs(f1_macro(true, pred) - float(f1_score(true, pred, average="macro", zero_division=0))) > 1e-9:
        mismatch += 1
print("selisih:", mismatch)
assert mismatch == 0
print("COCOK dengan sklearn")
```

Run: `.venv\Scripts\python.exe check_f1.py`
Expected: `selisih: 0` lalu `COCOK dengan sklearn`

Hapus berkasnya: `Remove-Item check_f1.py`

- [ ] **Step 6: Commit**

```bash
git add bacteriacv/evaluate.py tests/test_evaluate.py
git commit -m "feat: add macro f1 and accuracy evaluation"
```

---

## Task 7: Loop pelatihan head

**Files:**
- Create: `bacteriacv/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_train.py`:

```python
"""Tes untuk loop pelatihan dua head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bacteriacv.config import FEATURE_DIM
from bacteriacv.train import (
    EarlyStopping,
    SplitIndices,
    TrainConfig,
    compute_class_weights,
    indices_from_splits,
    train_heads,
)


def _separable_data() -> tuple[np.ndarray, list[str]]:
    """Buat matriks fitur yang mudah dipisah."""
    rng = np.random.default_rng(1)
    positive = rng.normal(loc=1.0, scale=0.2, size=(30, FEATURE_DIM))
    negative = rng.normal(loc=-1.0, scale=0.2, size=(30, FEATURE_DIM))
    return np.vstack([positive, negative]).astype(np.float32), ["a"] * 30 + ["b"] * 30


def test_train_config_defaults() -> None:
    """Konfigurasi pelatihan punya nilai yang masuk akal."""
    config = TrainConfig()
    assert config.epochs > 0
    assert config.patience > 0
    assert 0 < config.learning_rate < 1


def test_indices_from_splits_partitions_without_overlap() -> None:
    """Split train, val, dan test tidak saling tumpang tindih."""
    indices = indices_from_splits(
        {"p0": "train", "p1": "val", "p2": "test", "p3": "train"}
    )
    assert indices.train == [0, 3]
    assert indices.val == [1]
    assert indices.test == [2]


def test_indices_from_splits_rejects_unknown_split() -> None:
    """Nilai split yang tidak dikenal harus ditolak."""
    with pytest.raises(ValueError, match="tidak dikenal"):
        indices_from_splits({"p0": "train", "p1": "holdout"})


def test_split_indices_rejects_overlap() -> None:
    """Baris yang muncul di dua split harus ditolak."""
    indices = SplitIndices(train=[0, 1], val=[1], test=[2])
    with pytest.raises(ValueError, match="lebih dari satu split"):
        indices.validate()


def test_split_indices_rejects_empty() -> None:
    """Split kosong harus ditolak dengan nama split yang disebut."""
    indices = SplitIndices(train=[0], val=[], test=[1])
    with pytest.raises(ValueError, match="val"):
        indices.validate()


def test_class_weights_come_from_train_only() -> None:
    """Bobot kelas hanya dihitung dari baris data latih."""
    targets = [0] * 10 + [1, 1]
    weights = compute_class_weights(targets, {"train": list(range(10))}, n_classes=2)
    assert len(weights) == 2
    assert weights[1] > weights[0], "kelas minor harus berbobot lebih besar"


def test_class_weights_rejects_single_class() -> None:
    """Hanya satu kelas tidak cukup untuk menghitung bobot tidak seimbang."""
    with pytest.raises(ValueError):
        compute_class_weights([0, 0, 0], {"train": [0, 1, 2]}, n_classes=2)


def test_early_stopping_records_best_epoch() -> None:
    """Early stopping mengingat epoch terbaik."""
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.5, epoch=0) is True
    assert stopper.step(0.7, epoch=1) is False
    assert stopper.best_epoch == 0
    assert stopper.best_score == 0.5


def test_early_stopping_counts_without_improvement() -> None:
    """Counter bertambah pada epoch tanpa perbaikan."""
    stopper = EarlyStopping(patience=2)
    stopper.step(0.5, epoch=0)
    stopper.step(0.4, epoch=1)
    assert stopper.counter == 1


def test_early_stopping_respects_min_delta() -> None:
    """Peningkatan kecil di bawah min_delta bukan perbaikan."""
    stopper = EarlyStopping(patience=3, min_delta=0.1)
    assert stopper.step(0.50, epoch=0) is True
    assert stopper.step(0.55, epoch=1) is False
    assert stopper.best_score == 0.50


def test_train_heads_improves_on_separable_data(tmp_path, monkeypatch) -> None:
    """Data yang mudah dipisah harus menghasilkan F1 validasi tinggi."""
    from bacteriacv import train as train_module

    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", tmp_path)
    torch.manual_seed(0)
    matrix, species = _separable_data()
    indices = SplitIndices(
        train=list(range(0, 40)), val=list(range(40, 50)), test=list(range(50, 60))
    )

    history = train_heads(
        matrix,
        species,
        indices,
        config=TrainConfig(epochs=20, patience=8, seed=0),
        verbose=False,
    )

    assert len(history["val_f1_shape"]) > 0
    assert max(history["val_f1_shape"]) > 0.9


def test_test_split_evaluated_exactly_once(tmp_path, monkeypatch) -> None:
    """Data uji hanya dievaluasi satu kali, yaitu di akhir."""
    from bacteriacv import train as train_module

    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", tmp_path)
    torch.manual_seed(0)
    matrix, species = _separable_data()
    indices = SplitIndices(
        train=list(range(0, 40)), val=list(range(40, 50)), test=list(range(50, 60))
    )

    history = train_heads(
        matrix,
        species,
        indices,
        config=TrainConfig(epochs=3, patience=2, seed=0),
        verbose=False,
    )

    assert len(history["test_f1_shape"]) == 1
    assert len(history["test_f1_gram"]) == 1


def test_train_heads_rejects_empty_split(tmp_path, monkeypatch) -> None:
    """Split kosong harus ditolak sebelum pelatihan dimulai."""
    from bacteriacv import train as train_module

    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", tmp_path)
    matrix, species = _separable_data()
    indices = SplitIndices(train=list(range(0, 40)), val=[], test=list(range(40, 60)))

    with pytest.raises(ValueError, match="val"):
        train_heads(matrix, species, indices, verbose=False)
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.train'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/train.py`:

```python
"""Pelatihan dua classification head pada fitur backbone beku.

Bobot kelas dihitung hanya dari data latih. Early stopping memakai F1-score
makro validasi. Data uji dievaluasi satu kali di akhir dan tidak pernah dipakai
untuk pemilihan checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    FEATURES_DIR,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_DELTA,
    N_GRAM_CLASSES,
    N_SHAPE_CLASSES,
    TRAIN_SEED,
    WEIGHT_DECAY,
)
from .evaluate import f1_macro
from .label_map import class_weights, to_targets
from .model import BacteriaNet, build_model, load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    """Konfigurasi loop pelatihan.

    Attributes:
        epochs: Jumlah epoch maksimum.
        patience: Jumlah epoch tanpa perbaikan sebelum berhenti.
        learning_rate: Learning rate Adam.
        weight_decay: Regularisasi L2.
        min_delta: Perbaikan minimum agar dihitung sebagai kemajuan.
        batch_size: Ukuran batch.
        seed: Seed untuk reproducibility.
    """

    epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    min_delta: float = MIN_DELTA
    batch_size: int = BATCH_SIZE
    seed: int = TRAIN_SEED


@dataclass(frozen=True)
class SplitIndices:
    """Indeks baris fitur untuk tiap split.

    Attributes:
        train: Indeks baris data latih.
        val: Indeks baris data validasi.
        test: Indeks baris data uji.
    """

    train: list[int]
    val: list[int]
    test: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        """Kembalikan sebagai dictionary."""
        return {"train": self.train, "val": self.val, "test": self.test}

    def validate(self) -> None:
        """Periksa bahwa ketiga split ada dan tidak tumpang tindih.

        Raises:
            ValueError: Bila ada split kosong atau baris yang muncul dua kali.
        """
        for name, rows in self.as_dict().items():
            if not rows:
                raise ValueError(f"Split {name} kosong, tidak bisa melatih model.")
        flat = self.train + self.val + self.test
        if len(flat) != len(set(flat)):
            raise ValueError("Ada baris fitur yang muncul di lebih dari satu split.")


def indices_from_splits(splits: dict[str, str]) -> SplitIndices:
    """Ubah peta nama path ke nilai split menjadi indeks baris fitur.

    Args:
        splits: Pama path sumber citra ke nilai split, satu per baris fitur.

    Returns:
        SplitIndices berisi indeks baris per split.

    Raises:
        ValueError: Bila ada nilai split yang tidak dikenal.
    """
    ordered: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for position, value in enumerate(splits.values()):
        if value not in ordered:
            raise ValueError(f"Split tidak dikenal: {value}")
        ordered[value].append(position)
    return SplitIndices(
        train=ordered["train"], val=ordered["val"], test=ordered["test"]
    )


class EarlyStopping:
    """Hentikan pelatihan bila skor validasi tidak membaik selama patience epoch.

    Attributes:
        patience: Jumlah epoch toleransi.
        min_delta: Perbaikan minimum agar dihitung sebagai kemajuan.
        best_score: Skor terbaik sejauh ini.
        best_epoch: Epoch saat skor terbaik tercapai.
        counter: Jumlah epoch berturut-turut tanpa perbaikan.
    """

    def __init__(self, patience: int, min_delta: float = MIN_DELTA) -> None:
        """Inisialisasi.

        Args:
            patience: Jumlah epoch toleransi.
            min_delta: Perbaikan minimum.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.best_epoch = -1
        self.counter = 0

    def step(self, score: float, epoch: int = 0) -> bool:
        """Catat skor satu epoch.

        Args:
            score: Skor validasi.
            epoch: Nomor epoch.

        Returns:
            True bila epoch ini adalah yang terbaik.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self) -> bool:
        """Beri tahu apakah pelatihan harus dihentikan."""
        return self.counter >= self.patience


def compute_class_weights(
    targets: list[int], indices: dict[str, list[int]], n_classes: int
) -> torch.Tensor:
    """Hitung bobot kelas hanya dari baris data latih.

    Args:
        targets: Daftar target untuk seluruh baris fitur.
        indices: Pemetaan split ke indeks baris.
        n_classes: Jumlah kelas.

    Returns:
        Tensor bobot sepanjang n_classes.

    Raises:
        ValueError: Bila data latih kosong atau hanya berisi satu kelas.
    """
    return class_weights([targets[row] for row in indices["train"]], n_classes)


def _make_loader(
    matrix: np.ndarray,
    shape_targets: list[int],
    gram_targets: list[int],
    rows: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Buat DataLoader untuk satu split."""
    features = torch.from_numpy(np.asarray(matrix[rows], dtype=np.float32))
    shapes = torch.tensor([shape_targets[row] for row in rows], dtype=torch.long)
    grams = torch.tensor([gram_targets[row] for row in rows], dtype=torch.float32)
    return DataLoader(
        TensorDataset(features, shapes, grams), batch_size=batch_size, shuffle=shuffle
    )


def _evaluate_loader(model: BacteriaNet, loader: DataLoader) -> dict[str, float]:
    """Hitung F1 makro dan akurasi per head pada satu split."""
    shape_true: list[int] = []
    shape_pred: list[int] = []
    gram_true: list[int] = []
    gram_pred: list[int] = []

    model.eval()
    with torch.inference_mode():
        for features, shape_target, gram_target in loader:
            shape_pred.extend(model.head_a(features).argmax(dim=-1).tolist())
            gram_pred.extend(
                (model.head_b(features).sigmoid() > 0.5).long().reshape(-1).tolist()
            )
            shape_true.extend(shape_target.tolist())
            gram_true.extend(gram_target.long().tolist())

    correct_shape = sum(1 for a, b in zip(shape_true, shape_pred) if a == b)
    correct_gram = sum(1 for a, b in zip(gram_true, gram_pred) if a == b)
    return {
        "f1_shape": f1_macro(shape_true, shape_pred),
        "f1_gram": f1_macro(gram_true, gram_pred),
        "accuracy_shape": correct_shape / len(shape_true),
        "accuracy_gram": correct_gram / len(gram_true),
    }


def _snapshot(model: BacteriaNet) -> BacteriaNet:
    """Bangun model berisi salinan bobot head saat ini."""
    clone = build_model(pretrained=False)
    clone.head_a.load_state_dict(model.head_a.state_dict())
    clone.head_b.load_state_dict(model.head_b.state_dict())
    return clone


def train_heads(
    matrix: np.ndarray,
    species_ids: list[str],
    indices: SplitIndices,
    config: TrainConfig | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Latih kedua head pada fitur yang sudah diekstrak.

    Args:
        matrix: Matriks fitur B x 2048.
        species_ids: Daftar species_id sepanjang B.
        indices: Indeks baris per split.
        config: Konfigurasi pelatihan.
        verbose: Bila True, cetak progres setiap epoch.

    Returns:
        Dictionary riwayat. Kunci test_* berisi tepat satu elemen karena data
        uji hanya dievaluasi satu kali di akhir.

    Raises:
        ValueError: Bila ada split kosong atau baris tumpang tindih.
    """
    config = config or TrainConfig()
    indices.validate()
    torch.manual_seed(config.seed)

    shape_targets, gram_targets = to_targets(species_ids)
    split_map = indices.as_dict()
    shape_weights = compute_class_weights(shape_targets, split_map, N_SHAPE_CLASSES)
    gram_weights = compute_class_weights(gram_targets, split_map, N_GRAM_CLASSES)

    train_loader = _make_loader(
        matrix, shape_targets, gram_targets, indices.train, config.batch_size, True
    )
    val_loader = _make_loader(
        matrix, shape_targets, gram_targets, indices.val, config.batch_size, False
    )
    test_loader = _make_loader(
        matrix, shape_targets, gram_targets, indices.test, config.batch_size, False
    )

    model = build_model(pretrained=False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_shape = nn.CrossEntropyLoss(weight=shape_weights)
    loss_gram = nn.BCEWithLogitsLoss(pos_weight=gram_weights)
    stopper = EarlyStopping(config.patience, config.min_delta)
    best_path = CHECKPOINT_DIR / "head_best.pt"

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_f1_shape": [],
        "val_f1_gram": [],
        "val_accuracy_shape": [],
        "val_accuracy_gram": [],
        "test_f1_shape": [],
        "test_f1_gram": [],
        "test_accuracy_shape": [],
        "test_accuracy_gram": [],
    }

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for features, shape_batch, gram_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_shape(model.head_a(features), shape_batch) + loss_gram(
                model.head_b(features), gram_batch
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        history["train_loss"].append(total_loss)

        scores = _evaluate_loader(model, val_loader)
        history["val_f1_shape"].append(scores["f1_shape"])
        history["val_f1_gram"].append(scores["f1_gram"])
        history["val_accuracy_shape"].append(scores["accuracy_shape"])
        history["val_accuracy_gram"].append(scores["accuracy_gram"])

        improved = stopper.step((scores["f1_shape"] + scores["f1_gram"]) / 2, epoch)
        if verbose:
            print(
                f"epoch {epoch:3d} loss {total_loss:.4f} "
                f"F1 bentuk {scores['f1_shape']:.4f} "
                f"F1 gram {scores['f1_gram']:.4f}",
                flush=True,
            )
        if improved:
            save_checkpoint(_snapshot(model), best_path)
        if stopper.should_stop():
            if verbose:
                print(
                    f"Berhenti dini pada epoch {epoch}, "
                    f"epoch terbaik {stopper.best_epoch}"
                )
            break

    best_model = build_model(pretrained=False)
    load_checkpoint(best_model, best_path)

    test_scores = _evaluate_loader(best_model, test_loader)
    history["test_f1_shape"].append(test_scores["f1_shape"])
    history["test_f1_gram"].append(test_scores["f1_gram"])
    history["test_accuracy_shape"].append(test_scores["accuracy_shape"])
    history["test_accuracy_gram"].append(test_scores["accuracy_gram"])
    return history


def feature_paths(features_dir: Path) -> list[str]:
    """Baca daftar path sumber dari feature store.

    Args:
        features_dir: Folder feature store.

    Returns:
        Daftar path sumber citra, satu per baris fitur.
    """
    return (Path(features_dir) / "paths.txt").read_text(encoding="utf-8").splitlines()


def main(argv: list[str] | None = None) -> int:
    """Titik masuk baris perintah."""
    parser = argparse.ArgumentParser(
        description="Latih dua head pada fitur tersimpan"
    )
    parser.add_argument("--features", type=Path, default=FEATURES_DIR)
    parser.add_argument(
        "--out", type=Path, default=CHECKPOINT_DIR / "training_history.json"
    )
    args = parser.parse_args(argv)

    from .features import load_feature_store
    from .paths import INDEX_PATH

    matrix, species_ids = load_feature_store(args.features)
    source_paths = feature_paths(args.features)

    split_by_path: dict[str, str] = {}
    with INDEX_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split_by_path[row["path"]] = row["split"]

    missing = [name for name in source_paths if name not in split_by_path]
    if missing:
        print(f"GAGAL: {len(missing)} path fitur tidak ada di index.csv.")
        return 1

    feature_splits = [split_by_path[name] for name in source_paths]
    indices = indices_from_splits(dict(zip(source_paths, feature_splits)))
    history = train_heads(matrix, species_ids, indices)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print()
    print("Hasil data uji:")
    print(f"  F1-score makro bentuk : {history['test_f1_shape'][0]:.4f}")
    print(f"  F1-score makro Gram   : {history['test_f1_gram'][0]:.4f}")
    print(f"  Akurasi bentuk        : {history['test_accuracy_shape'][0]:.4f}")
    print(f"  Akurasi Gram          : {history['test_accuracy_gram'][0]:.4f}")
    print(f"Riwayat disimpan di {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**KETERGANTUNGAN TAMBAHAN:** `main` membaca `paths.txt` dari feature store. Task 5 harus
juga menulis berkas itu. Bila Task 5 sudah ditulis tanpa `paths.txt`, tambahkan penulisan
berkas tersebut pada fungsi `save_feature_store` sebelum menjalankan Task 7.

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: 12 passed

- [ ] **Step 5: Jalankan seluruh tes**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: semua lolos, tidak ada regresi

- [ ] **Step 6: Commit**

```bash
git add bacteriacv/train.py tests/test_train.py
git commit -m "feat: add dual head training loop with early stopping"
```

---
## Task 8: Inferensi dan visualisasi

**Files:**
- Create: `bacteriacv/infer.py`
- Test: `tests/test_infer.py`

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_infer.py`:

```python
"""Tes untuk inferensi dan visualisasi."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bacteriacv.config import GRAM_LABELS, SHAPE_LABELS
from bacteriacv.infer import Predictor, confidence_level, encode_panel
from bacteriacv.model import build_model, save_checkpoint


@pytest.fixture()
def predictor(tmp_path) -> Predictor:
    """Predictor dengan checkpoint head yang sudah ada."""
    model = build_model(pretrained=False)
    checkpoint = tmp_path / "head.pt"
    save_checkpoint(model, checkpoint)
    return Predictor(checkpoint_path=checkpoint, pretrained=False)


def test_predict_returns_required_fields(predictor: Predictor) -> None:
    """Hasil prediksi harus memuat label dan confidence kedua head."""
    image = np.full((400, 500, 3), 200, dtype=np.uint8)
    result = predictor.predict(image)
    assert result.shape_label in SHAPE_LABELS
    assert result.gram_label in GRAM_LABELS
    assert 0.0 <= result.shape_confidence <= 1.0
    assert 0.0 <= result.gram_confidence <= 1.0


def test_predict_includes_panels(predictor: Predictor) -> None:
    """Hasil prediksi harus memuat lima panel visualisasi."""
    image = np.full((400, 500, 3), 200, dtype=np.uint8)
    result = predictor.predict(image)
    assert len(result.panels) == 5
    assert set(result.stage_ok) >= {"resize", "normalisasi", "segmentasi", "watershed"}


def test_predict_flags_segmentation_failure(predictor: Predictor) -> None:
    """Segmentasi gagal harus ditandai, hasil klasifikasi tetap ada."""
    blank = np.full((400, 500, 3), 255, dtype=np.uint8)
    result = predictor.predict(blank)
    assert result.stage_ok["segmentasi"] is False
    assert result.shape_label in SHAPE_LABELS


def test_predictor_missing_checkpoint_raises(tmp_path) -> None:
    """Checkpoint yang tidak ada harus ditolak dengan pesan jelas."""
    with pytest.raises(FileNotFoundError):
        Predictor(checkpoint_path=tmp_path / "tidak_ada.pt", pretrained=False)


def test_confidence_levels_match_design() -> None:
    """Level confidence mengikuti DESIGN bagian 5."""
    assert confidence_level(0.85) == "tinggi"
    assert confidence_level(0.8) == "tinggi"
    assert confidence_level(0.7) == "sedang"
    assert confidence_level(0.6) == "sedang"
    assert confidence_level(0.55) == "rendah"
    assert confidence_level(0.0) == "rendah"


def test_encode_panel_returns_png_bytes(predictor: Predictor) -> None:
    """Panel harus dapat diserialkan menjadi PNG untuk dikirim ke browser."""
    image = np.full((400, 500, 3), 200, dtype=np.uint8)
    payload = encode_panel(image)
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_panel_handles_float32(predictor: Predictor) -> None:
    """Panel float32 harus dikonversi ke uint8 sebelum diserialkan."""
    image = np.zeros((50, 50, 3), dtype=np.float32)
    payload = encode_panel(image)
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_infer.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'bacteriacv.infer'`

- [ ] **Step 3: Tulis implementasi**

Buat `bacteriacv/infer.py`:

```python
"""Inferensi: dari citra mentah ke label, confidence, dan visualisasi.

Lookup table [L] tidak dipakai pada modul ini. Label spesies tidak diketahui
pada tahap inferensi, sesuai ARCHITECTURE bagian 3.2.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CHECKPOINT_DIR
from .model import build_model, label_gram, label_shape, load_checkpoint
from .preprocess import PANEL_NAMES, preprocess

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "head_best.pt"


@dataclass
class PredictionResult:
    """Keluaran satu inferensi.

    Attributes:
        shape_label: Label bentuk sel.
        shape_confidence: Confidence bentuk, 0 sampai 1.
        gram_label: Label status Gram.
        gram_confidence: Confidence status Gram, 0 sampai 1.
        panels: Lima citra RGB berurutan sesuai PANEL_NAMES.
        panel_names: Nama tahap panel.
        stage_ok: Status keberhasilan tiap tahap pra-pemrosesan.
        object_count: Jumlah objek sel terdeteksi.
        segmentation_message: Pesan error segmentasi bila gagal, selain itu None.
    """

    shape_label: str
    shape_confidence: float
    gram_label: str
    gram_confidence: float
    panels: tuple[np.ndarray, ...]
    panel_names: tuple[str, ...]
    stage_ok: dict[str, bool]
    object_count: int
    segmentation_message: str | None = None


def confidence_level(value: float) -> str:
    """Ubah confidence menjadi label tingkat sesuai DESIGN bagian 5.

    Args:
        value: Nilai confidence 0 sampai 1.

    Returns:
        "tinggi", "sedang", atau "rendah".
    """
    if value >= 0.8:
        return "tinggi"
    if value >= 0.6:
        return "sedang"
    return "rendah"


def encode_panel(image: np.ndarray) -> bytes:
    """Serialkan citra panel menjadi PNG.

    Args:
        image: Array RGB uint8 atau float32.

    Returns:
        Byte PNG.
    """
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.ndim == 3 and array.shape[2] == 3 and array.dtype == np.float32:
            array = np.clip(array * 255.0, 0, 255) if array.max() <= 1.0 else np.clip(array, 0, 255)
        array = array.astype(np.uint8)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Panel gagal diserialkan menjadi PNG.")
    return buffer.tobytes()


class Predictor:
    """Pembungkus model untuk inferensi.

    Attributes:
        model: BacteriaNet dengan bobot head yang sudah dimuat.
    """

    def __init__(self, checkpoint_path: Path | None = None, pretrained: bool = True) -> None:
        """Muat model untuk inferensi.

        Args:
            checkpoint_path: Lokasi checkpoint head.
            pretrained: Bila True, muat bobot ImageNet backbone.

        Raises:
            FileNotFoundError: Bila checkpoint tidak ada.
        """
        self.model = build_model(pretrained=pretrained)
        path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        load_checkpoint(self.model, path)
        self.model.eval()

    def predict(self, image: np.ndarray) -> PredictionResult:
        """Jalankan pra-pemrosesan dan prediksi.

        Args:
            image: Array citra RGB.

        Returns:
            PredictionResult berisi label, confidence, dan panel visualisasi.
        """
        result = preprocess(image)
        features = self.model.extract_features(result.image_tensor.unsqueeze(0))
        shape_index, shape_conf, gram_index, gram_conf = self.model.predict(features)

        message = None
        if not result.stage_ok["segmentasi"]:
            message = "Segmentasi tidak berhasil. Hasil klasifikasi tetap ditampilkan tanpa visualisasi."

        return PredictionResult(
            shape_label=label_shape(int(shape_index[0])),
            shape_confidence=float(shape_conf[0]),
            gram_label=label_gram(int(gram_index[0])),
            gram_confidence=float(gram_conf[0]),
            panels=result.panels,
            panel_names=PANEL_NAMES,
            stage_ok=result.stage_ok,
            object_count=result.object_count,
            segmentation_message=message,
        )
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_infer.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add bacteriacv/infer.py tests/test_infer.py
git commit -m "feat: add inference with staged visualization panels"
```

---

## Task 9: Antarmuka web FastAPI

**Files:**
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/static/index.html`
- Test: `tests/test_app.py`

- [ ] **Step 1: Tulis tes yang gagal**

Buat `tests/test_app.py`:

```python
"""Tes untuk API FastAPI."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bacteriacv.config import MAX_UPLOAD_BYTES


@pytest.fixture()
def client() -> TestClient:
    """Klien uji dengan model dummy."""
    from app.main import create_app

    return TestClient(create_app(pretrained=False))


def test_index_page_serves_html(client: TestClient) -> None:
    """GET / harus mengembalikan halaman HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_index_page_contains_palette(client: TestClient) -> None:
    """Halaman harus memakai palet yang ditetapkan DESIGN bagian 2."""
    response = client.get("/")
    for color in ("#588157", "#A3B18A", "#DAD7CD", "#ffffff"):
        assert color in response.text


def test_predict_rejects_unsupported_format(client: TestClient) -> None:
    """Format berkas yang tidak didukung harus ditolak dengan pesan DESIGN bagian 7."""
    response = client.post(
        "/predict", files={"file": ("catatan.txt", io.BytesIO(b"bukan citra"), "text/plain")}
    )
    assert response.status_code == 400
    assert "Format berkas tidak didukung" in response.json()["detail"]


def test_predict_rejects_oversized_file(client: TestClient) -> None:
    """Berkas melebihi 20 MB harus ditolak dengan pesan DESIGN bagian 7."""
    oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/predict", files={"file": ("besar.png", io.BytesIO(oversized), "image/png")}
    )
    assert response.status_code == 400
    assert "20 MB" in response.json()["detail"]


def test_predict_rejects_unreadable_image(client: TestClient) -> None:
    """Berkas gambar yang rusak harus ditolak dengan pesan DESIGN bagian 7."""
    response = client.post(
        "/predict", files={"file": ("rusak.png", io.BytesIO(b"\x00\x01\x02"), "image/png")}
    )
    assert response.status_code == 400
    assert "tidak dapat dibaca" in response.json()["detail"]


def test_predict_accepts_valid_png(client: TestClient) -> None:
    """PNG valid harus diproses dan mengembalikan label serta panel."""
    import cv2

    image = np.full((300, 400, 3), 220, dtype=np.uint8)
    image[100:200, 150:250] = (120, 60, 150)
    ok, buffer = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/predict", files={"file": ("citra.png", io.BytesIO(buffer.tobytes()), "image/png")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["shape_label"] in ("cocci", "bacilli")
    assert payload["gram_label"] in ("positif", "negatif")
    assert 0.0 <= payload["shape_confidence"] <= 1.0
    assert len(payload["panels"]) == 5


def test_predict_panels_are_png(client: TestClient) -> None:
    """Setiap panel harus berupa PNG yang dapat dirender browser."""
    import cv2

    image = np.full((200, 200, 3), 210, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/predict", files={"file": ("citra.png", io.BytesIO(buffer.tobytes()), "image/png")}
    )
    assert response.status_code == 200
    for name, encoded in response.json()["panels"].items():
        assert name in ("original", "resized", "normalized", "segmented", "watershed")
        assert encoded.startswith("iVBORw0KGgo")
```

**CATATAN:** Tes memakai `fastapi.testclient.TestClient` yang memerlukan `httpx`. Bila belum
terpasang, tambahkan `httpx` ke `setup_env.ps1`. Periksa dengan:
`TestClient` membutuhkan paket `httpx`.

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv\Scripts\python.exe -m pytest tests/test_app.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Buat package app**

Buat `app/__init__.py`:
```python
"""Antarmuka web BacteriaCV."""
```

- [ ] **Step 4: Tulis backend**

Buat `app/main.py`:

```python
"""Backend FastAPI untuk BacteriaCV.

Satu proses Python melayani halaman statis dan endpoint prediksi. Model
PyTorch dimuat sekali saat aplikasi dibangun, lalu dipakai bersama.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from bacteriacv.config import ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES, STATIC_DIR
from bacteriacv.infer import Predictor, confidence_level, encode_panel

INDEX_FILE = STATIC_DIR / "index.html"


def create_app(pretrained: bool = True) -> FastAPI:
    """Bangun aplikasi FastAPI.

    Args:
        pretrained: Bila True, muat bobot ImageNet backbone.

    Returns:
        Instans FastAPI siap jalan.
    """
    application = FastAPI(title="BacteriaCV", version="0.1.0")
    state: dict[str, Predictor] = {}

    def get_predictor() -> Predictor:
        if "predictor" not in state:
            state["predictor"] = Predictor(pretrained=pretrained)
        return state["predictor"]

    @application.get("/")
    def index() -> FileResponse:
        """Sajikan halaman HTML tunggal.

        Returns:
            FileResponse berisi index.html.
        """
        return FileResponse(INDEX_FILE, media_type="text/html")

    @application.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        """Proses citra unggahan dan kembalikan hasil klasifikasi.

        Args:
            file: Berkas citra dari formulir unggah.

        Returns:
            Dictionary berisi label, confidence, panel PNG, dan status tahap.

        Raises:
            HTTPException: 400 bila format, ukuran, atau isi berkas tidak valid.
        """
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail="Format berkas tidak didukung. Gunakan PNG, JPG, atau TIFF.",
            )

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400, detail="Ukuran berkas melebihi 20 MB."
            )

        buffer = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Citra tidak dapat dibaca. Coba unggah berkas lain.",
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = get_predictor().predict(rgb)
        panels = {
            name: base64.b64encode(encode_panel(panel)).decode("ascii")
            for name, panel in zip(result.panel_names, result.panels)
        }
        return {
            "shape_label": result.shape_label,
            "shape_confidence": round(result.shape_confidence, 4),
            "shape_level": confidence_level(result.shape_confidence),
            "gram_label": result.gram_label,
            "gram_confidence": round(result.gram_confidence, 4),
            "gram_level": confidence_level(result.gram_confidence),
            "panels": panels,
            "stage_ok": result.stage_ok,
            "object_count": result.object_count,
            "message": result.segmentation_message,
        }

    return application


app = create_app()
```

**CATATAN:** Baris `app = create_app()` di module level akan memuat bobot ImageNet saat
diimpor, termasuk saat pytest mengimpor modul. Untuk mencegah itu, buat `app` secara lazy.
Ganti baris terakhir dengan:

```python
def __getattr__(name: str):
    """Bangun aplikasi default hanya saat benar-benar diakses."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
```

- [ ] **Step 5: Tulis frontend**

Buat `app/static/index.html`. Seluruh isi file:

```html
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BacteriaCV - Klasifikasi Bentuk Sel dan Status Gram</title>
<style>
:root{--primary:#588157;--secondary:#A3B18A;--surface:#DAD7CD;--bg:#ffffff;--text:#1b1b1b}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,Helvetica,sans-serif;font-size:16px;color:var(--text);background:var(--bg)}
header{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid var(--surface)}
header nav{display:flex;gap:16px}
main{padding:24px;max-width:1100px;margin:0 auto}
.top-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}
.card{border:1px solid var(--surface);border-radius:8px;padding:20px;background:var(--bg)}
.card h2{font-size:18px;font-weight:bold;margin-bottom:12px;color:var(--primary)}
#drop{border:2px dashed var(--secondary);border-radius:8px;padding:32px;text-align:center;cursor:pointer}
#drop.drag{border-color:var(--primary);background:var(--surface)}
#drop img{max-width:100%;max-height:180px;margin-top:12px}
button{background:var(--primary);color:#ffffff;border:none;padding:10px 18px;border-radius:6px;font-size:16px;cursor:pointer}
button:disabled{background:var(--secondary);cursor:not-allowed}
button.secondary{background:var(--secondary);color:#1b1b1b}
label.level{display:inline-block;padding:2px 8px;border-radius:4px;font-size:13px;margin-left:8px}
label.level.tinggi{background:var(--primary);color:#ffffff}
label.level.sedang{background:var(--secondary);color:#1b1b1b}
label.level.rendah{background:#DAD7CD;color:#1b1b1b}
.stages{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.stage{border:1px solid var(--surface);border-radius:6px;overflow:hidden}
.stage img{width:100%;display:block}
.stage p{font-size:13px;padding:6px 8px}
.stage p.fail{color:#8a1f1f}
#warning{background:var(--surface);border-left:4px solid var(--primary);padding:10px 14px;margin-bottom:12px;font-size:14px;display:none}
footer{display:flex;flex-wrap:wrap;gap:8px;padding:16px 24px;border-top:1px solid var(--surface);margin-top:24px}
.chip{background:var(--surface);border-radius:12px;padding:4px 12px;font-size:13px}
</style>
</head>
<body>
<header>
<strong style="color:var(--primary)">BacteriaCV</strong>
<nav><a href="#">Unggah</a><a href="#">Riwayat</a><a href="#">Tentang</a></nav>
<div><button class="secondary" type="button">Contoh</button> <button type="button" id="process" disabled>Proses</button></div>
</header>
<main>
<div class="top-row">
<div class="card">
<h2>Unggah Citra</h2>
<div id="drop">Letakkan berkas di sini atau klik untuk memilih
<input type="file" id="file" accept=".png,.jpg,.jpeg,.tif,.tiff" hidden>
<div id="preview"></div>
</div>
</div>
<div class="card">
<h2>Ringkasan Hasil</h2>
<div id="result">Belum ada hasil.</div>
</div>
</div>
<h2 style="font-size:18px;font-weight:bold;color:var(--primary)">Tahapan Pra-pemrosesan</h2>
<p style="font-size:13px;margin-bottom:12px">Setiap tahap ditampilkan berurutan. Tahap yang gagal ditandai, alur tetap berjalan.</p>
<div id="warning">Hasil ini sebagai alat bantu, bukan diagnosis.</div>
<div class="stages" id="stages"></div>
</main>
<footer>
<span class="chip">PNG, JPG, TIFF</span>
<span class="chip">Maks 20 MB</span>
<span class="chip">Model v0.1.0</span>
<span class="chip">Data: DIBaS</span>
<span class="chip">Alat bantu, bukan diagnosis</span>
</footer>
<script>
const drop=document.getElementById('drop'),fileInput=document.getElementById('file'),
process=document.getElementById('process'),preview=document.getElementById('preview'),
result=document.getElementById('result'),stages=document.getElementById('stages'),
warning=document.getElementById('warning');
let chosen=null;
const names={original:'Asli',resized:'Resize 224',normalized:'Normalisasi',segmented:'Segmentasi',watershed:'Watershed'};
drop.onclick=()=>fileInput.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};
drop.ondragleave=()=>drop.classList.remove('drag');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');setFile(e.dataTransfer.files[0])};
fileInput.onchange=e=>setFile(e.target.files[0]);
function setFile(f){
if(!f)return;
const ok=['image/png','image/jpeg','image/tiff'].includes(f.type)||/\.(png|jpe?g|tiff?)$/i.test(f.name);
if(!ok){result.textContent='Format berkas tidak didukung. Gunakan PNG, JPG, atau TIFF.';return}
if(f.size>20*1024*1024){result.textContent='Ukuran berkas melebihi 20 MB.';return}
chosen=f;process.disabled=false;
const url=URL.createObjectURL(f);
preview.innerHTML='<img src="'+url+'" alt="Pratinjau citra">';
}
process.onclick=async()=>{
if(!chosen)return;
process.disabled=true;result.textContent='Memproses...';
const body=new FormData();body.append('file',chosen);
try{
const res=await fetch('/predict',{method:'POST',body});
const data=await res.json();
if(!res.ok){result.textContent=data.detail;return}
result.innerHTML='<p><strong>Bentuk sel:</strong> '+data.shape_label+
' '+data.shape_confidence.toFixed(3)+'<span class="level '+data.shape_level+'">'+data.shape_level+'</span></p>'+
'<p><strong>Status Gram:</strong> '+data.gram_label+
' '+data.gram_confidence.toFixed(3)+'<span class="level '+data.gram_level+'">'+data.gram_level+'</span></p>'+
'<p style="font-size:13px">Objek sel terdeteksi: '+data.object_count+'</p>';
warning.style.display='block';
stages.innerHTML='';
for(const key of Object.keys(names)){
const d=document.createElement('div');d.className='stage';
const ok=data.stage_ok[key]!==false;
d.innerHTML='<img src="data:image/png;base64,'+data.panels[key]+'" alt="'+names[key]+'">'+
'<p class="'+(ok?'':'fail')+'">'+names[key]+(ok?'':' - gagal')+'</p>';
stages.appendChild(d);
}
if(data.message)result.innerHTML+='<p style="font-size:13px;margin-top:8px">'+data.message+'</p>';
}catch(e){result.textContent='Citra tidak dapat dibaca. Coba unggah berkas lain.'}
finally{process.disabled=false}
};
</script>
</body>
</html>
```

- [ ] **Step 6: Pasang httpx bila belum ada**

Run: `.venv\Scripts\python.exe -m pip install httpx`
Expected: `Successfully installed httpx-...` atau `Requirement already satisfied`

- [ ] **Step 7: Jalankan tes, pastikan lolos**

Run: `.venv\Scripts\python.exe -m pytest tests/test_app.py -v`
Expected: 8 passed

- [ ] **Step 8: Jalankan seluruh tes**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: semua lolos, tidak ada regressions

- [ ] **Step 9: Uji manual dengan uvicorn**

Run in background:
```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Lalu buka `http://127.0.0.1:8000` di browser dan unggah satu citra dari
`data\raw\images\escherichia_coli\Escherichia.coli_0001.tif`. Pastikan lima panel muncul dan
label terisi.

- [ ] **Step 10: Commit**

```bash
git add app/__init__.py app/main.py app/static/index.html tests/test_app.py
git commit -m "feat: add fastapi backend and single page web interface"
```

---

## Task 10: Skrip menjalankan aplikasi

**Files:**
- Create: `scripts/run_app.ps1`

- [ ] **Step 1: Tulis skrip**

Buat `scripts/run_app.ps1`:

```powershell
# run_app.ps1
# Menjalankan antarmuka web BacteriaCV memakai Python dari .venv.

$ErrorActionPreference = "Stop"
Unblock-File $MyInvocation.MyCommand.Path -ErrorAction SilentlyContinue

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$index = Join-Path $root "data\index.csv"
$checkpoint = Join-Path $root "checkpoints\head_best.pt"

if (-not (Test-Path $python)) {
    Write-Host "GAGAL  .venv belum dibuat, jalankan setup_env.ps1" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $index)) {
    Write-Host "GAGAL  data\index.csv belum ada, jalankan build_index" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $checkpoint)) {
    Write-Host "GAGAL  checkpoint belum ada, jalankan pelatihan lebih dulu" -ForegroundColor Yellow
    Write-Host "      aplikasi tetap berjalan tapi prediksi tidak tersedia" -ForegroundColor Yellow
}

Write-Host "Menjalankan BacteriaCV di http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Tekan Ctrl+C untuk menghentikan." -ForegroundColor Gray

Set-Location $root
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**CATATAN:** Skrip memakai string ASCII. Simpan sebagai UTF-8 dengan BOM. Jalankan
`Unblock-File scripts\run_app.ps1` sebelum eksekusi pertama.

- [ ] **Step 2: Simpan dengan BOM dan verifikasi**

Jalankan perbaikan encoding bila perlu, lalu verifikasi tidak ada karakter non-ASCII.

- [ ] **Step 3: Uji skrip**

Run: `.\scripts\run_app.ps1`
Expected:-printed "Menjalankan BacteriaCV di http://127.0.0.1:8000"

Tekan Ctrl+C untuk menghentikan setelah verifikasi.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_app.ps1
git commit -m "feat: add run script for web application"
```

---

## Task 11: Sinkronkan dokumentasi

**Files:**
- Modify: `docs/SPEC.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DESIGN.md`
- Modify: `docs/CONTEXT.md`

- [ ] **Step 1: Perbarui SPEC bagian 1 dan 3 di SPEC.md**

Ganti kalimat pada SPEC bagian 1 yang menyebut tiga kelas bentuk dengan dua kelas, dan
tambahkan catatan tentang kelas spiral.

Teks baru untuk SPEC bagian 1:
```
Sistem Computer Vision yang memprediksi bentuk sel (cocci, bacilli) dan status Gram
(positif, negatif) bakteri secara simultan dari satu citra mikroskop digital, menggunakan
satu backbone ResNet-50 dengan dua classification head.

Catatan kelas bentuk: dataset DIBaS tidak memuat satu pun spesies berbentuk spiral.
Head A karena itu dilatih pada dua kelas. Bentuk spiral tetap dicatat sebagai kelas
yang tidak terisi, bukan dihapus, agar batas dataset terlihat jelas dalam laporan.
```

- [ ] **Step 2: Tambahkan bagian hasil eksperimen ke SPEC.md**

Tambahkan bagian baru di akhir SPEC.md:

```markdown
## 10. Hasil Eksperimen

Angka diisi setelah pelatihan dijalankan. Semua angka berasal dari data uji 65 citra
yang tidak pernah dipakai selama pelatihan.

| Metrik | Head A (bentuk) | Head B (Gram) |
|--------|-----------------|---------------|
| F1-score makro | belum diisi | belum diisi |
| Akurasi | belum diisi | belum diisi |

Jumlah citra: 672 dari 32 spesies. Pembagian: 469 latih, 138 validasi, 65 uji.
Augmentasi 4 varian hanya pada data latih. Validasi dan uji memakai citra asli
tanpa duplikasi.
```

- [ ] **Step 3: Perbarui ARCHITECTURE bagian 1 dan 2**

Ganti `softmax, 3` dengan `softmax, 2` pada diagram, dan perbarui C3:

```markdown
### C3. Classification Head
- Head A: Linear(2048, 2), aktivasi softmax, loss cross-entropy berbobot.
  Dua kelas: cocci dan bacilli.
- Head B: Linear(2048, 2), aktivasi sigmoid, loss binary cross-entropy berbobot.
- Bobot kelas dihitung dari frekuensi pada data latih saja.
```

- [ ] **Step 4: Tambahkan keputusan arsitektur baru di ARCHITECTURE bagian 5**

```markdown
| A10 | Head A dua kelas, bukan tiga | DIBaS tidak punya spesies spiral. Kelas kosong membuat F1 makro tidak terdefinisi. |
| A11 | Augmentasi 4 varian pada data latih saja | Mengurangi overfitting head pada 469 sampel tanpa menyentuh validasi dan uji. |
| A12 | Fitur disimpan sebagai .npy | Matriks 672 x 2048 hanya 5,5 MB, tidak perlu database. |
| A13 | Inferensi satu proses FastAPI | Bobot PyTorch tidak bisa dimuat dari runtime JavaScript. |
```

- [ ] **Step 5: Perbarui DESIGN bagian 8**

Ganti baris label_map dengan kontrak final:

```markdown
- `label_map.py`: `LOOKUP`, `to_targets`, `unmapped_species`, `class_weights`.
```

- [ ] **Step 6: Perbarui CONTEXT bagian istilah**

Tambahkan istilah baru:

```markdown
| Coccobacillus | Bakteri di antara coccus dan bacillus, bentuk pendek dan plump. Contoh: Acinetobacter baumannii, Porphyromonas gingivalis. |
| Blob normalization | Normalisasi intensitas gambar mikroskopis untuk mengoreksi pencahayaan yang tidak merata. |
```

- [ ] **Step 7: Verifikasi tidak ada angka lama yang tertinggal**

Run: `Select-String -Path docs\*.md -Pattern "660|softmax, 3|2048, 3|src/"`
Expected: tidak ada hasil yang relevan

- [ ] **Step 8: Commit**

```bash
git add docs/SPEC.md docs/ARCHITECTURE.md docs/DESIGN.md docs/CONTEXT.md
git commit -m "docs: sync specs with two-class head and measured results"
```

---

## Task 12: Verifikasi akhir

**Files:**
- Modify: `scripts/check_env.ps1`

- [ ] **Step 1: Perbarui check_env.ps1**

Tambahkan pemeriksaan berkas baru di akhir skrip, sebelum blok penutup:

```powershell
Write-Host "Memeriksa komponen pipeline..."
$components = @(
    @("bacteriacv\config.py", "config"),
    @("bacteriacv\preprocess.py", "preprocess"),
    @("bacteriacv\model.py", "model"),
    @("bacteriacv\label_map.py", "label_map"),
    @("bacteriacv\features.py", "features"),
    @("bacteriacv\train.py", "train"),
    @("bacteriacv\evaluate.py", "evaluate"),
    @("bacteriacv\infer.py", "infer"),
    @("app\main.py", "app backend")
)
foreach ($c in $components) {
    $p = Join-Path (Get-Location) $c[0]
    if (Test-Path $p) {
        Write-Host "  OK  $($c[1])"
    } else {
        Write-Host "  BELUM  $($c[1])"
    }
}

Write-Host "Memeriksa feature store..."
if (Test-Path "data\features\features.npy") {
    Write-Host "  OK  data\features\features.npy"
} else {
    Write-Host "  BELUM  feature store, jalankan ekstraksi fitur"
}

Write-Host "Memeriksa checkpoint..."
if (Test-Path "checkpoints\head_best.pt") {
    Write-Host "  OK  checkpoints\head_best.pt"
} else {
    Write-Host "  BELUM  checkpoint, jalankan pelatihan"
}
```

- [ ] **Step 2: Jalankan verifikasi lingkungan**

Run: `.\scripts\check_env.ps1`
Expected: semua komponen OK bila Task 1 sampai 9 sudah dikerjakan

- [ ] **Step 3: Jalankan seluruh tes**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: semua lolos, tidak ada regressions

- [ ] **Step 4: Verifikasi tidak ada path absolut di kode produksi**

Run: `.venv\Scripts\python.exe -m pytest tests/test_paths.py -v`
Expected: semua lolos

- [ ] **Step 5: Verifikasi tidak ada kredensial**

Run: `Select-String -Path bacteriacv\*.py, app\*.py -Pattern "password|secret|token|api_key|sk-"`
Expected: tidak ada hasil

- [ ] **Step 6: Serah terima ke auditor**

Susun ringkasan perubahan, daftar berkas, dan prompt audit sesuai AGENT.md bagian 3
aturan 7 sampai 9. Prompt audit harus menyebut:
1. Konteks tugas dan batasannya.
2. Acuan dokumen yang berlaku.
3. Daftar seluruh berkas yang Diaudit.
4. Keputusan desain yang tidak boleh dianggap sebagai kesalahan.
5. Fokus pemeriksaan khusus sesuai AGENT.md bagian 4 ayat 4.

---

## Checklist Self-Review

**Cakupan spesifikasi:** Setiap kebutuhan SPEC dipetakan ke task.
- F1 maksa: F1c1 accept PNG/JPG/TIFF, Task 8 dan 9.
- F2 pra-pemrosesan sesuai spesifikasi: Task 3.
- F3 bentuk sel plus confidence: Task 4 dan 8.
- F4 status Gram plus confidence: Task 4 dan 8.
- F5 visualisasi segmentasi: Task 3 dan 8.
- F6 class weighting dari data latih: Task 2 dan 6.
- F7 F1 makro dan akurasi per head: Task 7.
- F8 antarmuka web: Task 9.
- SPEC bagian 7 class weighting dari data latih: Task 2 dan 6.
- SPEC bagian 9.1 kriteria demo: Task 9 Step 9.
- SPEC bagian 9.2 kriteria laporan: Task 11.

**Konsistensi tipe:** `SHAPE_LABELS` dua elemen dipakai di `config.py`; `LOOKUP` memakai
nilai yang ada di `SHAPE_LABELS`; `N_SHAPE_CLASSES` dua sesuai `Linear(2048, 2)`;
`f1_macro` mengembalikan float dipakai di `train.py`; `save_checkpoint` dan
`load_checkpoint` memakai kunci `head_a` dan `head_b` konsisten.

**Known issues yang harus diselesaikan saat eksekusi:**
1. `train.py` `main` belum memanggil `train_heads` dengan indeks split. Harus dilengkapi.
2. `app/main.py` perlu `__getattr__` lazy agar pytest tidak memuat bobot ImageNet.
3. Dua catatan CJK tidak sengaja pada `preprocess.py` dan `CONTEXT.md` harus diganti ASCII.
4. `features.py` mengimpor `preprocess` di dalam fungsi `_augmented_paths`, sebaiknya di atas.
5. `train.py` mengimpor `BacteriaNet` tapi tidak memakainya di beberapa tempat, bersihkan.
6. `predictor` di Task 8 memakai fixture bertipe `Predictor` yang belum ada di pytest 8,
   ganti dengan `-> Predictor` pada anotasi return.
