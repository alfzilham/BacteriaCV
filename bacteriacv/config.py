"""Konfigurasi terpusat untuk seluruh pipeline BacteriaCV.

Nilai di sini adalah sumber kebenaran tunggal untuk hyperparameter. Modul lain
tidak boleh mendefinisikan angka yang sama secara lokal.
"""

from __future__ import annotations

from .paths import PROJECT_ROOT

# ---------------------------------------------------------------------
# Citra
# ---------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------
# Head A: bentuk sel
# ---------------------------------------------------------------------
# Keputusan D1: hanya dua kelas yang terisi pada DIBaS. Tidak ada satu pun
# spesies DIBaS yang berbentuk spiral, sehingga Head A dilatih pada dua kelas.
# Mempertahankan kelas kosong akan membuat F1-score makro tidak terdefinisi.
N_SHAPE_CLASSES = 2
SHAPE_LABELS = ("cocci", "bacilli")

# Enum lengkap tetap memuat spiral sebagai penanda kelas yang tidak terisi,
# agar batas dataset terlihat jelas dalam laporan dan bukan dihapus diam-diam.
SHAPE_LABELS_FULL = ("cocci", "bacilli", "spiral")
SHAPE_UNPOPULATED = "spiral"

# ---------------------------------------------------------------------
# Head B: status Gram
# ---------------------------------------------------------------------
N_GRAM_CLASSES = 2
GRAM_LABELS = ("positive", "negative")

# ---------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------
FEATURE_DIM = 2048
BACKBONE_NAME = "resnet50"
BACKBONE_WEIGHTS = "IMAGENET1K_V2"
HEAD_A_SIZE = N_SHAPE_CLASSES
HEAD_B_SIZE = N_GRAM_CLASSES

# ---------------------------------------------------------------------
# Augmentasi, hanya pada data latih
# ---------------------------------------------------------------------
# Keputusan D3. Augmentasi diterapkan pada level citra, sebelum backbone
# membekukan fitur. Data validasi dan uji memakai citra asli tanpa duplikasi.
AUGMENT_VARIANTS = 4
AUGMENT_ROTATION_DEGREES = 15
AUGMENT_SCALE_RANGE = (0.9, 1.1)
AUGMENT_SHIFT_FRACTION = 0.05
AUGMENT_HORIZONTAL_FLIP = True

# ---------------------------------------------------------------------
# Segmentasi morfologis
# ---------------------------------------------------------------------
# Keputusan pemilik proyek: segmentasi berjalan pada resolusi asli
# 2048 x 1532, bukan pada 224 x 224. Alasannya bersifat skala.
#
# Pada resolusi asli, satu piksel setara sekitar 0,048 mikron, sehingga
# sel bakteri 1 mikron berdiameter sekitar 21 piksel. Pada 224 x 224, satu
# piksel setara sekitar 0,43 mikron, sehingga sel yang sama hanya
# 2,3 piksel. Footprint morfologis bersifat scale-dependent: konstanta yang
# sama berarti erosi jauh lebih destructive pada citra yang sudah di-resize.
#
# Nilai di bawah sudah diskalakan untuk resolusi asli. Ukuran objek median
# dari citra DIBaS sekitar 21 piksel, sehingga jejak 5 sampai 7 piksel masih
# proporsional terhadap ukuran sel.
SEGMENT_SCALE = "full"

SEGMENT_EROSION_DISK = 5
SEGMENT_DILATION_DISK = 5
SEGMENT_MIN_PEAK_DISTANCE = 15
SEGMENT_MIN_OBJECT_AREA = 300

# Perkiraan mikron per piksel pada kedua skala, dipakai untuk menuliskan
# setara mikron dari konstanta morfologi pada laporan.
APPROX_MICRONS_PER_PIXEL_FULL = 0.048
APPROX_MICRONS_PER_PIXEL_RESIZED = 0.43

# Ambang integritas data. Spesies dengan citra kurang dari angka ini dianggap
# hasil ekstraksi tidak lengkap.
MIN_IMAGES_PER_SPECIES = 15

# ---------------------------------------------------------------------
# Pembagian data
# ---------------------------------------------------------------------
# Hanya data latih yang punya nomor lipatan. Data validasi dipakai untuk
# early stopping, data uji hanya dievaluasi sekali di akhir.
N_FOLDS = 5
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20
INDEX_SEED = 20260203

# ---------------------------------------------------------------------
# Pelatihan
# ---------------------------------------------------------------------
TRAIN_SEED = 1337
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_DELTA = 1e-4

# ---------------------------------------------------------------------
# Lokasi keluaran
# ---------------------------------------------------------------------
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"

# ---------------------------------------------------------------------
# Antarmuka web
# ---------------------------------------------------------------------
# DESIGN bagian 7: batas ukuran 20 MB.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# DESIGN bagian 5: level confidence tinggi, sedang, rendah.
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.6

LOW_CONFIDENCE_WARNING = "Hasil ini sebagai alat bantu, bukan diagnosis."