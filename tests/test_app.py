"""Tes untuk antarmuka web BacteriaCV.

Tes memakai TestClient FastAPI dengan checkpoint sementara. Checkpoint sengaja
dibuat dengan bobot head yang ditentukan, bukan acak, supaya hasil prediksi
dapat diassert tanpa bergantung pada inisialisasi.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from bacteriacv.config import MAX_UPLOAD_BYTES
from bacteriacv.model import build_model, save_checkpoint

GOOD_IMAGE = "citra.png"
BAD_SUFFIX = "catatan.txt"

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "app" / "static"


def _png_bytes(dark: bool = True) -> bytes:
    """Buat citra PNG di memori."""
    import cv2

    array = np.full((240, 320, 3), 215, dtype=np.uint8)
    if dark:
        array[80:180, 90:230] = (20, 20, 220)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    """Checkpoint head yang terklasifikasi sebagai cocci positif."""
    model = build_model(pretrained=False)
    with torch.no_grad():
        model.head_a.weight.zero_()
        model.head_a.weight[:, 0] = torch.tensor([4.0, -4.0])
        model.head_a.bias.zero_()
        model.head_b.weight.zero_()
        model.head_b.weight[:, 0] = torch.tensor([4.0, -4.0])
        model.head_b.bias.zero_()
    return save_checkpoint(model, tmp_path / "heads.pt")


@pytest.fixture()
def client(checkpoint: Path):
    """Klien uji dengan checkpoint sementara."""
    import app.main as app_module

    application = app_module.create_app(checkpoint)
    with TestClient(application) as test_client:
        yield test_client


# --- Halaman dan status ---


def test_index_page_is_served(client: TestClient) -> None:
    """Halaman utama harus dapat dibuka."""
    response = client.get("/")

    assert response.status_code == 200
    assert "BacteriaCV" in response.text


def test_health_reports_checkpoint(client: TestClient) -> None:
    """Endpoint status harus menyebut nama checkpoint yang dimuat."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "siap"
    assert response.json()["checkpoint"] == "heads.pt"
    assert response.json()["max_upload_bytes"] == MAX_UPLOAD_BYTES


# --- Prediksi ---


def test_predict_returns_labels_confidence_and_panel(client: TestClient) -> None:
    """Respons harus memuat prediksi, panel, dan catatan."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["shape_label"] == "cocci"
    assert payload["prediction"]["gram_label"] == "positive"
    assert 0.0 <= payload["prediction"]["shape_confidence"] <= 1.0
    assert payload["panel"]
    assert payload["stage_index"] == 0


def test_predict_sends_five_panels_and_five_tabs(client: TestClient) -> None:
    """Tombol alih tahap membutuhkan kelima panel dan judulnya."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    payload = response.json()
    assert len(payload["panels"]) == 5
    assert len(payload["stage_names"]) == 5
    assert len(payload["stage_texts"]) == 5
    assert len(payload["stage_keys"]) == 5
    assert payload["stage_names"][0] == "original"


def test_predict_honours_stage_query_parameter(client: TestClient) -> None:
    """Parameter stage menentukan panel yang dikirim."""
    response = client.post(
        "/api/predict?stage=3",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    payload = response.json()
    assert payload["stage_index"] == 3
    assert payload["panel"] == payload["panels"][3]
    assert payload["stage_name"] == "segment"


def test_predict_always_states_segmentation_not_validated(client: TestClient) -> None:
    """Catatan keterbatasan segmentasi harus ikut pada respons."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    payload = response.json()
    assert payload["notes"]
    assert any("not yet validated" in note for note in payload["notes_text"])
    assert payload["disclaimer"]
    assert payload["prediction"]["segmentation_validated"] is False


def test_predict_keeps_result_when_segmentation_fails(client: TestClient) -> None:
    """Citra tanpa area gelap harus tetap terjawab."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, _png_bytes(dark=False), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["shape_label"] == "cocci"
    assert payload["failed_stages"]
    assert payload["prediction"]["segmentation_ok"] is False


def test_predict_rejects_non_image_suffix(client: TestClient) -> None:
    """Berkas noncitra harus ditolak dengan kode 415."""
    response = client.post(
        "/api/predict",
        files={"file": (BAD_SUFFIX, b"bukan citra", "text/plain")},
    )

    assert response.status_code == 415
    assert "not supported" in response.json()["detail"]


def test_predict_rejects_empty_file(client: TestClient) -> None:
    """Berkas kosong harus ditolak dengan kode 400."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, b"", "image/png")},
    )

    assert response.status_code == 400


def test_predict_rejects_unreadable_image(client: TestClient) -> None:
    """Payload PNG rusak harus ditolak dengan kode 400."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, b"bukan png sama sekali", "image/png")},
    )

    assert response.status_code == 400
    assert "image" in response.json()["detail"]


def test_predict_rejects_oversized_file(client: TestClient) -> None:
    """Berkas melebihi batas harus ditolak dengan kode 413."""
    oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1024)

    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, oversized, "image/png")},
    )

    assert response.status_code == 413


def test_predict_rejects_out_of_range_stage(client: TestClient) -> None:
    """Stage di luar rentang harus ditolak."""
    response = client.post(
        "/api/predict?stage=99",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert "Stage" in response.json()["detail"]


def test_predict_rejects_non_numeric_stage(client: TestClient) -> None:
    """Stage bukan angka harus ditolak."""
    response = client.post(
        "/api/predict?stage=abc",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    assert response.status_code == 400


def test_predict_is_deterministic(client: TestClient) -> None:
    """Citra sama harus menghasilkan prediksi sama."""
    payload = _png_bytes()

    first = client.post(
        "/api/predict", files={"file": (GOOD_IMAGE, payload, "image/png")}
    ).json()["prediction"]
    second = client.post(
        "/api/predict", files={"file": (GOOD_IMAGE, payload, "image/png")}
    ).json()["prediction"]

    assert first["shape_label"] == second["shape_label"]
    assert first["shape_confidence"] == second["shape_confidence"]


# --- Laporan evaluasi ---


def test_report_returns_404_before_evaluation(client: TestClient, monkeypatch) -> None:
    """Laporan belum ada harus memberi 404, bukan error server."""
    import app.main as app_module

    monkeypatch.setattr(app_module, "CHECKPOINT_DIR", REPO_ROOT / "tidak_ada")

    response = client.get("/api/report")

    assert response.status_code == 404


def test_report_returns_evaluation_file(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """Laporan evaluasi harus disajikan apa adanya."""
    import json

    import app.main as app_module

    monkeypatch.setattr(app_module, "CHECKPOINT_DIR", tmp_path)
    (tmp_path / app_module.EVALUATION_NAME).write_text(
        json.dumps({"shape": {"f1_macro": 0.9}}), encoding="utf-8"
    )

    response = client.get("/api/report")

    assert response.status_code == 200
    assert response.json()["shape"]["f1_macro"] == 0.9


# --- Kekhususan proyek ---


def test_index_html_has_one_panel_with_toggle() -> None:
    """DESIGN bagian 3: satu panel dengan tombol alih, bukan lima panel.

    Berkas harus punya tepat satu elemen img untuk panel dan satu wadah tombol
    alih. Lima elemen img berarti desain satu-panel dilanggar.
    """
    page = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")

    # Logo merek juga memakai <img> dan itu bukan panel, jadi yang dihitung
    # adalah img selain logo. Jumlahnya tetap harus tepat satu.
    images = re.findall(r"<img\b[^>]*>", page)
    panels = [tag for tag in images if 'class="brand-logo"' not in tag]
    assert len(panels) == 1, f"harus ada tepat satu img panel, ada {len(panels)}"

    # Tanpa ini, dua img logo ikut tersaring dan hitungan panel tetap satu.
    logos = [tag for tag in images if 'class="brand-logo"' in tag]
    assert len(logos) == 1, f"harus ada tepat satu img logo, ada {len(logos)}"

    assert 'id="tabs"' in page
    assert 'id="panel"' in page


def test_static_css_and_js_are_external_files() -> None:
    """CSS dan JavaScript harus berada di berkas statis sendiri.

    Berkas main.css dan main.js wajib ada dan tidak kosong, index.html harus
    merujuk keduanya lewat path absolut /static, dan tidak boleh ada blok
    <style> atau <script> inline yang tertinggal.

    Path absolut itu wajib: FastAPI memasang static di /static, sehingga
    rujukan relatif seperti href="main.css" akan menghasilkan 404. Halaman
    lalu tampil tanpa gaya dan tanpa interaksi, dan tidak ada tes lama yang
    akan gagal karena itu.
    """
    static_dir = REPO_ROOT / "app" / "static"
    css = static_dir / "main.css"
    js = static_dir / "main.js"

    assert css.is_file(), "app/static/main.css tidak ada"
    assert js.is_file(), "app/static/main.js tidak ada"
    assert css.stat().st_size > 0, "app/static/main.css kosong"
    assert js.stat().st_size > 0, "app/static/main.js kosong"

    page = (static_dir / "index.html").read_text(encoding="utf-8")

    assert '<link rel="stylesheet" href="/static/main.css">' in page
    assert '<script src="/static/main.js" defer></script>' in page

    # Tidak ada blok inline. Satu-satunya tag script adalah rujukan eksternal,
    # sehingga blok <script>...</script> yang berisi kode mustahil ada.
    assert "<style" not in page
    assert page.count("<script") == 1
    assert "</script>" in page and "main.js" in page


# Ambang kontras untuk komponen UI non-teks, bukan teks. Lihat DESIGN.md
# bagian 2.1: scrollbar adalah komponen UI, jadi yang berlaku adalah
# WCAG 1.4.11 Non-text Contrast dengan ambang 3:1, bukan WCAG 1.4.3 yang
# 4,5:1 dan hanya berlaku untuk teks.
SCROLLBAR_MIN_CONTRAST = 3.0


def _srgb_to_linear(channel: int) -> float:
    """Linearisasi satu kanal sRGB 0-255, rumus WCAG 2.1."""
    c = channel / 255
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_colour: str) -> float:
    """Luminositas relatif WCAG dari warna hexRRGGBB."""
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * _srgb_to_linear(r)
            + 0.7152 * _srgb_to_linear(g)
            + 0.0722 * _srgb_to_linear(b))


def _contrast_ratio(first: str, second: str) -> float:
    """Rasio kontras WCAG (hi + 0.05) / (lo + 0.05)."""
    a = _relative_luminance(first)
    b = _relative_luminance(second)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def _design_tokens(css: str) -> dict[str, str]:
    """Kumpulkan setiap definisi --nama: nilai dari blok :root."""
    root = re.search(r":root\s*\{([^}]*)\}", css)
    assert root, "blok :root tidak ditemukan di main.css"
    tokens = dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", root.group(1)))
    assert tokens, "blok :root tidak mendefinisikan custom property"
    return tokens


def _resolve(value: str, tokens: dict[str, str], limit: int = 10) -> str:
    """Ganti setiap var(--nama) dengan nilainya, berulang sampai tidak ada var().

    limit membatasi iterasi supaya token yang saling menunjuk tidak looping.
    """
    for _ in range(limit):
        if "var(" not in value:
            break
        value = re.sub(
            r"var\(\s*(--[a-z0-9-]+)\s*\)",
            lambda m: tokens.get(m.group(1), m.group(0)),
            value,
        )
    return value.strip()


def _declared_value(css: str, selector: str, prop: str) -> str:
    """Baca satu properti dari satu blok CSS, lalu resolve tokennya.

    Lookbehind mencegah properti_background cocok dengan background-color.
    """
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"blok {selector} tidak ditemukan di main.css"
    decl = re.search(rf"(?<![-\w]){re.escape(prop)}\s*:\s*([^;]+);", match.group(1))
    assert decl, f"{selector} tidak mendeklarasikan properti {prop}"
    return _resolve(decl.group(1), _design_tokens(css))


def _declared_background(css: str, selector: str) -> str:
    """Nilai background dari satu blok CSS, setelah token var(--) di-resolve."""
    value = _declared_value(css, selector, "background")
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", value), (
        f"{selector} tidak menghasilkan warna hex, dapat {value!r}"
    )
    return value


# Track #EAE8E3 hanya 1,11:1 terhadap latar halaman #F4F4F0, jadi tanpa
# pembatas dia praktis tidak terlihat. Ambang pembatas memakai standar yang
# sama seperti ambang thumb, yaitu 3:1.
SCROLLBAR_TRACK_MIN_CONTRAST = 3.0


def _colour_in_shorthand(shorthand: str) -> str:
    """Ambil warna hex dari satu shorthand, misalnya "2px solid #050505"."""
    found = re.search(r"#[0-9A-Fa-f]{6}", shorthand)
    assert found, f"shorthand tidak memuat warna hex, dapat {shorthand!r}"
    return found.group(0)


def test_scrollbar_track_has_visible_delimiter() -> None:
    """Track scrollbar wajib punya pembatas yang kontrasnya minimal 3:1.

    Track #EAE8E3 hanya 1,11:1 terhadap latar halaman #F4F4F0. Tanpa
    pembatas, track menyatu dengan halaman dan hilang. Tes ini menuntut
    adanya pembatas dan warnanya harus kontras minimal 3:1 terhadap track.

    Tes ini sengaja tidak mengunci nilai pembatas. Kalau nanti track
    digelapkan sebagai alternatif yang sah, cukup ganti pembatasnya; yang
    tetap wajib adalah track itu punya pembatas yang terlihat.
    """
    css = (REPO_ROOT / "app" / "static" / "main.css").read_text(encoding="utf-8")

    track = _declared_background(css, "::-webkit-scrollbar-track")

    track_block = re.search(
        r"::-webkit-scrollbar-track\s*\{([^}]*)\}", css
    )
    assert track_block, "blok ::-webkit-scrollbar-track tidak ditemukan di main.css"

    border = re.search(r"(?<![-\w])border-left\s*:\s*([^;]+);", track_block.group(1))
    assert border, (
        "track scrollbar tidak punya pembatas, jadi tidak terlihat di atas "
        "latar halaman: hex #EAE8E3 hanya 1,11:1 terhadap #F4F4F0"
    )

    delimiter = _colour_in_shorthand(_resolve(border.group(1), _design_tokens(css)))
    ratio = _contrast_ratio(delimiter, track)

    assert ratio >= SCROLLBAR_TRACK_MIN_CONTRAST, (
        f"pembatas track {delimiter} cuma {ratio:.2f}:1 terhadap track {track}, "
        f"di bawah ambang {SCROLLBAR_TRACK_MIN_CONTRAST}:1, jadi track tidak "
        "terlihat di atas latar halaman"
    )

    print(
        f"pembatas track {delimiter} di track {track} = {ratio:.2f}:1, "
        f"ambang {SCROLLBAR_TRACK_MIN_CONTRAST}:1"
    )


def test_webkit_scrollbar_contrast_is_enforced() -> None:
    """Kontras scrollbar harus tetap di atas ambang 3:1.

    DESIGN.md bagian 2.1 menetapkan ambang 3:1 karena scrollbar adalah
    komponen UI, bukan teks. Tes ini menghitung ulang rasionya dari
    main.css supaya revisi UI berikutnya tidak bisa menurunkannya tanpa
    ketahuan.

    Warna scrollbar ditulis sebagai var(--token), jadi nilainya dibaca lewat
    resolver token, bukan hex mentah. Nilai yang harus terbaca setelah
    di-resolve:
      scrollbar-color #050505 #EAE8E3, track #EAE8E3, thumb #050505,
      hover #D31515.
    """
    css = (REPO_ROOT / "app" / "static" / "main.css").read_text(encoding="utf-8")

    # 1. properti scrollbar-color pada blok html
    html_block = re.search(r"\bhtml\s*\{([^}]*)\}", css)
    assert html_block, "blok html tidak ditemukan di main.css"
    tokens = _design_tokens(css)
    shorthand = re.search(r"scrollbar-color\s*:\s*([^;]+);", html_block.group(1))
    assert shorthand, "properti scrollbar-color tidak ditemukan pada blok html"
    resolved = _resolve(shorthand.group(1), tokens).split()
    assert len(resolved) == 2, f"scrollbar-color harus punya dua warna, dapat {resolved}"
    shorthand_thumb, shorthand_track = resolved
    for colour in (shorthand_thumb, shorthand_track):
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", colour), (
            f"scrollbar-color tidak menghasilkan warna hex, dapat {colour!r}"
        )

    # 2. background track, thumb, dan hover
    track = _declared_background(css, "::-webkit-scrollbar-track")
    thumb = _declared_background(css, "::-webkit-scrollbar-thumb")
    hover = _declared_background(css, "::-webkit-scrollbar-thumb:hover")

    # 3. hitung rasio tiap pasangan
    thumb_vs_track = _contrast_ratio(thumb, track)
    hover_vs_track = _contrast_ratio(hover, track)
    shorthand_vs_track = _contrast_ratio(shorthand_thumb, shorthand_track)

    # 4. tegakkan ambang 3:1
    for label, ratio in (
        (f"thumb {thumb} di track {track}", thumb_vs_track),
        (f"hover {hover} di track {track}", hover_vs_track),
        (f"scrollbar-color {shorthand_thumb} di {shorthand_track}", shorthand_vs_track),
    ):
        assert ratio >= SCROLLBAR_MIN_CONTRAST, (
            f"kontras {label} cuma {ratio:.2f}:1, di bawah ambang "
            f"{SCROLLBAR_MIN_CONTRAST}:1"
        )

    # Firefox dan WebKit harus memakai pasangan warna yang sama, kalau tidak
    # scrollbar akan berubah tampilan antarperamban.
    assert shorthand_thumb.lower() == thumb.lower(), (
        f"scrollbar-color {shorthand_thumb} tidak sama dengan thumb WebKit {thumb}"
    )
    assert shorthand_track.lower() == track.lower(), (
        f"scrollbar-color {shorthand_track} tidak sama dengan track WebKit {track}"
    )

    print(
        f"kontras scrollbar: thumb {thumb_vs_track:.2f}:1, "
        f"hover {hover_vs_track:.2f}:1, "
        f"shorthand {shorthand_vs_track:.2f}:1, "
        f"ambang {SCROLLBAR_MIN_CONTRAST}:1"
    )


def test_hidden_attribute_beats_dialog_backdrop_display() -> None:
    """Atribut hidden harus menang atas display yang ditulis .dialog-backdrop.

    .dialog-backdrop mendeklarasikan display: flex, dan deklarasi display di
    level penulis selalu menang atas [hidden] dari stylesheet bawaan browser.
    Akibatnya atasan hidden mati total, kedua dialog tampil sejak halaman
    dimuat, dan tidak ada yang bisa menutupnya.

    Karena itu main.css wajib punya aturan global [hidden] dengan
    display: none !important, dan aturan itu harus muncul sebelum
    .dialog-backdrop.
    """
    css = (REPO_ROOT / "app" / "static" / "main.css").read_text(encoding="utf-8")

    rule = re.search(r"\[hidden\]\s*\{[^}]*\}", css)
    assert rule, "aturan [hidden] tidak ada di main.css"

    block = rule.group(0)
    assert "display" in block, "aturan [hidden] tidak mengatur display"
    assert "none" in block, "aturan [hidden] tidak memakai display: none"
    assert "!important" in block, "aturan [hidden] tidak memakai !important"

    assert css.index("[hidden]") < css.index(".dialog-backdrop"), \
        "aturan [hidden] harus muncul sebelum .dialog-backdrop"

    # Inilah alasan aturan itu perlu: backdrop benar-benar menulis display.
    backdrop = re.search(r"\.dialog-backdrop\s*\{([^}]*)\}", css)
    assert backdrop, "aturan .dialog-backdrop tidak ditemukan di main.css"
    assert re.search(r"display\s*:\s*(?!none)", backdrop.group(1)), (
        ".dialog-backdrop tidak punya deklarasi display eksplisit, "
        "jadi aturan [hidden] tidak diuji oleh keadaan sebenarnya"
    )


def test_index_html_has_no_hardcoded_metric_claims() -> None:
    """Halaman tidak boleh menjanjikan angka sebelum evaluasi.

    Angka di halaman harus berasal dari respons server, bukan ditulis di
    markup, supaya perubahan model tidak meninggalkan klaim basi.
    """
    page = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")

    for claim in ("95%", "99%", "akurasi 9"):
        assert claim not in page


def test_app_module_does_not_import_label_map() -> None:
    """Label spesies tidak boleh bocor ke antarmuka.

    Pada inferensi spesies tidak diketahui. Kalau app mengimpor lookup table,
    developer bisa menambahkan tebakan spesies pada respons tanpa disengaja.
    """
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert "label_map" not in source
    assert "LOOKUP" not in source

# ===========================================================================
# Dua bahasa dan data kanonik
#
# Sifat yang menopang bagian ini: server tidak pernah mengubah data mengikuti
# bahasa pemanggil. Kalau iya, evaluation.json dan seluruh laporan bisa
# menampilkan angka berbeda tergantung siapa yang membuka, dan itu tidak bisa
# diterima untuk dokumen rujukan laporan studi kasus.
#
# Kamus dibaca dengan pembaca kecil sendiri, bukan mesin JavaScript, supaya tes
# ini tidak butuh dependensi baru dan tidak butuh runtime.
# ===========================================================================

I18N_PATH = STATIC_DIR / "i18n.js"
MAIN_JS_PATH = STATIC_DIR / "main.js"
LANGUAGES = ("en", "id")


def _i18n_object(source: str, name: str) -> dict:
    """Parse `export const <name> = { en: {...}, id: {...} };` into a dict.

    Uses ast.literal_eval on the object literal converted to Python syntax. The
    conversion is mechanical: keys become quoted strings, values stay strings.
    """
    match = re.search(
        rf"export const {re.escape(name)}\s*=\s*(\{{.*?\n\}};)", source, re.S
    )
    assert match, f"{name} tidak ditemukan di i18n.js"
    body = match.group(1).rstrip(";")

    # Komentar JavaScript tidak valid di Python, jadi dibuang dulu. Hanya ada
    # komentar satu baris penuh di kamus, jadi ini tidak pernah menyentuh nilai.
    body = re.sub(r"(?m)^\s*//.*$", "", body)
    # Kunci tanpa tanda kutip juga tidak valid di Python.
    python_literal = re.sub(
        r'(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', body
    )
    python_literal = (
        python_literal.replace("true", "True")
        .replace("false", "False")
        .replace("null", "None")
    )
    parsed = ast.literal_eval(python_literal)
    assert isinstance(parsed, dict), f"{name} bukan objek"
    return parsed


def _dictionaries() -> dict:
    """Kembalikan kedua kamus, diindeks dengan kode bahasa."""
    return _i18n_object(I18N_PATH.read_text(encoding="utf-8"), "I18N")


def _dictionary_keys() -> tuple:
    """Kembalikan (kunci bahasa Inggris, kunci bahasa Indonesia)."""
    dicts = _dictionaries()
    return set(dicts["en"]), set(dicts["id"])


# --- 1. berkas kamus ada dan bisa dibaca -----------------------------------
def test_i18n_file_exists_and_is_readable() -> None:
    """app/static/i18n.js harus ada, tidak kosong, dan terbaca jadi dua kamus."""
    assert I18N_PATH.is_file(), "app/static/i18n.js tidak ada"
    assert I18N_PATH.stat().st_size > 0, "app/static/i18n.js kosong"

    dicts = _dictionaries()

    assert set(dicts) == set(LANGUAGES), f"harus ada dua bahasa, ada {set(dicts)}"
    for lang in LANGUAGES:
        assert dicts[lang], f"kamus {lang} kosong"
        for key, value in dicts[lang].items():
            assert isinstance(value, str), f"{lang}.{key} bukan string"


# --- 2. kedua bahasa punya kumpulan kunci yang identik ----------------------
def test_both_languages_have_identical_keys() -> None:
    """Satu kunci yang hilang di satu bahasa akan lolos ke Inggris diam-diam."""
    english, indonesian = _dictionary_keys()

    assert english, "kamus bahasa Inggris kosong"
    assert indonesian, "kamus bahasa Indonesia kosong"
    assert english == indonesian, (
        "kunci tidak sama di kedua bahasa. "
        f"hanya di en: {sorted(english - indonesian)}; "
        f"hanya di id: {sorted(indonesian - english)}"
    )


# --- 3. tidak ada kunci bernilai kosong ------------------------------------
def test_no_key_has_an_empty_value() -> None:
    """Kunci kosong membuat elemen antarmuka kosong tanpa error terlihat."""
    dicts = _dictionaries()
    empty = []

    for lang in LANGUAGES:
        for key, value in dicts[lang].items():
            if not value.strip():
                empty.append(f"{lang}.{key}")

    assert not empty, f"nilai kunci kosong: {empty}"


# --- 4. setiap kunci dari server ada di kedua bahasa -----------------------
def test_every_server_note_and_stage_key_exists_in_both_languages() -> None:
    """note_keys dan stage_keys dari server harus punya terjemahan."""
    from bacteriacv.config import LOW_CONFIDENCE_WARNING_KEY
    from bacteriacv.infer import FAILED_STAGES_NOTE_KEY
    from bacteriacv.preprocess import SEGMENTATION_FAILURE_MESSAGE_KEY
    from bacteriacv.visualize import SEGMENTATION_LIMITATION_KEY, STAGE_KEYS

    english, indonesian = _dictionary_keys()

    note_keys = {
        SEGMENTATION_LIMITATION_KEY,
        LOW_CONFIDENCE_WARNING_KEY,
        SEGMENTATION_FAILURE_MESSAGE_KEY,
        FAILED_STAGES_NOTE_KEY,
    }
    # Sufiks gagal dipakai sebagai bagian dari kunci bertitik, jadi harus ada
    # sebagai kunci tersendiri juga.
    stage_keys = set(STAGE_KEYS.values()) | {"stage_failed_suffix"}

    for key in sorted(note_keys | stage_keys):
        assert key in english, f"kunci server {key} tidak ada di en"
        assert key in indonesian, f"kunci server {key} tidak ada di id"


def test_stage_keys_cover_every_stage() -> None:
    """Setiap tahap perlu kunci, kalau tidak judulnya tak bisa ditransliterasi."""
    from bacteriacv.preprocess import STAGE_NAMES
    from bacteriacv.visualize import STAGE_KEYS

    assert set(STAGE_NAMES) == set(STAGE_KEYS), (
        "STAGE_KEYS harus menutupi setiap tahap; "
        f"tanpa kunci: {sorted(set(STAGE_NAMES) - set(STAGE_KEYS))}"
    )

    english, _ = _dictionary_keys()
    outside = sorted(set(STAGE_KEYS.values()) - english)
    assert not outside, f"ada kunci tahap di luar kamus: {outside}"


# --- 5. respons predict mengembalikan kunci, bukan teks langsung -----------
def test_predict_returns_note_keys_and_notes_text(client: TestClient) -> None:
    """Server mengembalikan kunci dan teks, bukan teks saja."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )
    payload = response.json()

    for field in ("note_keys", "notes_text", "stage_keys", "stage_texts"):
        assert field in payload, f"{field} tidak ada di respons"

    assert payload["note_keys"], "note_keys kosong"
    assert payload["notes_text"], "notes_text kosong"
    assert len(payload["note_keys"]) == len(payload["notes_text"]), (
        "note_keys dan notes_text harus sejajar indeks per indeks"
    )
    assert len(payload["stage_keys"]) == len(payload["stage_texts"]), (
        "stage_keys dan stage_texts harus sejajar indeks per indeks"
    )

    # Field lama diganti, bukan dipertahankan dua-duanya, supaya tiap butir
    # hanya punya satu sumber teks kanonik.
    assert "stage_titles" not in payload, (
        "stage_titles seharusnya digantikan stage_keys dan stage_texts, bukan "
        "dipertahankan dua-duanya"
    )

    english, indonesian = _dictionary_keys()
    for key in payload["note_keys"]:
        assert key in english and key in indonesian, f"note_key {key} tak ada di kamus"
    for key in payload["stage_keys"]:
        base = key.split(".")[0]
        assert base in english and base in indonesian, f"stage_key {key} tak ada di kamus"


# --- 6. label server tetap Inggris apa pun Accept-Language-nya --------------
@pytest.mark.parametrize(
    "accept_language",
    ["id-ID,id;q=0.9", "en-US,en;q=0.9", "en", "id", "*", "", "de-DE,de;q=0.8"],
)
def test_labels_stay_english_under_any_accept_language(
    client: TestClient, accept_language: str
) -> None:
    """Data dari server tidak boleh pernah ikut berubah bahasa.

    Inilah uji yang memastikan arsitektur data kanonik tidak bocor.
    """
    response = client.post(
        "/api/predict",
        headers={"Accept-Language": accept_language},
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )
    payload = response.json()
    prediction = payload["prediction"]

    assert prediction["shape_label"] in {"cocci", "bacilli"}, (
        f"bentuk sel bukan label kanonik: {prediction['shape_label']!r}"
    )
    assert prediction["gram_label"] in {"positive", "negative"}, (
        f"status Gram bukan label kanonik: {prediction['gram_label']!r}"
    )
    assert prediction["shape_level"] in {"high", "medium", "low"}
    assert prediction["gram_level"] in {"high", "medium", "low"}

    for key in payload["note_keys"]:
        assert re.fullmatch(r"[a-z_]+(\.[a-z_]+)?", key), (
            f"note_key {key!r} bukan kunci ascii sederhana"
        )
    for text in payload["stage_texts"]:
        assert "gagal" not in text, f"stage_texts memuat kata Indonesia: {text!r}"


def test_nothing_in_the_stack_reads_accept_language() -> None:
    """Tidak boleh ada jalur kode yang membaca Accept-Language sama sekali.

    Membuktikannya secara statis lebih kuat daripada sekadar percaya satu respons,
    karena percakapan ini tidak bisa berubah tanpa menambah salah satu pola di
    bawah ini.
    """
    forbidden = re.compile(r"accept[-_]?language", re.I)

    checked = 0
    for path in (
        REPO_ROOT / "app" / "main.py",
        MAIN_JS_PATH,
        REPO_ROOT / "Procfile",
        REPO_ROOT / "requirements.txt",
    ):
        assert path.is_file(), f"{path} tidak ada"
        text = path.read_text(encoding="utf-8")
        found = forbidden.search(text)
        assert found is None, (
            f"{path.name} membaca Accept-Language pada baris "
            f"{text[: found.start()].count(chr(10)) + 1}; server harus selalu "
            "mengembalikan bahasa Inggris"
        )
        checked += 1
    assert checked == 4


# --- 7. nilai respons sama persis apa pun Accept-Language-nya --------------
def test_response_is_identical_under_any_accept_language(client: TestClient) -> None:
    """Dua header berbeda harus menghasilkan respons yang sama persis."""
    png = _png_bytes()

    def call(accept_language: str) -> dict:
        response = client.post(
            "/api/predict",
            headers={"Accept-Language": accept_language},
            files={"file": (GOOD_IMAGE, png, "image/png")},
        )
        assert response.status_code == 200
        return json.loads(response.content)

    with_indonesian = call("id-ID,id;q=0.9")
    with_english = call("en-US,en;q=0.9")
    without_header = call("")

    assert with_indonesian == with_english, (
        "respons berbeda antara header id dan en"
    )
    assert with_indonesian == without_header, (
        "respons berbeda antara header bahasa dan tanpa header"
    )

    # Ditulis satu per satu supaya kegagalan menyebut field yang sebenarnya beda.
    assert (
        with_indonesian["prediction"]["gram_label"]
        == with_english["prediction"]["gram_label"]
    )
    assert with_indonesian["note_keys"] == with_english["note_keys"]
    assert with_indonesian["notes_text"] == with_english["notes_text"]
    assert with_indonesian["stage_keys"] == with_english["stage_keys"]
    assert with_indonesian["stage_texts"] == with_english["stage_texts"]


# --- tombol penganti -------------------------------------------------------
def test_language_toggle_buttons_exist_in_the_header() -> None:
    """Dua tombol kecil di header, dekat tombol navigasi yang sudah ada."""
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="lang-en"' in page, "tombol EN tidak ada"
    assert 'id="lang-id"' in page, "tombol ID tidak ada"
    assert 'data-lang="en"' in page
    assert 'data-lang="id"' in page

    header = page.split("</header>")[0]
    assert 'id="lang-en"' in header, "tombol EN tidak berada di header"
    assert 'id="lang-id"' in header, "tombol ID tidak berada di header"

    assert "lang-toggle" in page, "wadah tombol bahasa tidak ada"
    assert 'role="group"' in page, "wadah tombol bahasa perlu role group"

    js = MAIN_JS_PATH.read_text(encoding="utf-8")
    assert "bacteriacv.lang" in js, "kunci localStorage tidak sesuai yang diminta"
    assert "getItem" in js and "setItem" in js, "pilihan bahasa tidak disimpan"


def test_default_language_is_english() -> None:
    """English adalah bawaan karena README dan dokumentasi sekarang bahasa Inggris."""
    js = MAIN_JS_PATH.read_text(encoding="utf-8")

    assert 'DEFAULT_LANG = "en"' in js, "bahasa bawaan bukan English"
    assert "indexOf(stored) === -1" in js, (
        "nilai localStorage yang tidak dikenal harus jatuh ke English"
    )


def test_language_toggle_follows_the_brutalist_style_rules() -> None:
    """Tanpa border-radius dan tanpa shadow, konsisten dengan header yang ada."""
    css = (STATIC_DIR / "main.css").read_text(encoding="utf-8")

    blocks = re.findall(r"\.lang-btn(?:\.[\w-]+)?\s*\{([^}]*)\}", css)
    assert blocks, "blok .lang-btn tidak ditemukan di main.css"

    for block in blocks:
        assert "border-radius" not in block, "tombol bahasa tidak boleh ada radius"
        assert "box-shadow" not in block, "tombol bahasa tidak boleh ada shadow"

    # Ukuran teks tidak boleh lebih kecil dari nav-link yang sudah ada.
    size = re.search(r"\.lang-btn\s*\{[^}]*font-size:\s*(\d+)px", css, re.S)
    assert size, "ukuran font tombol bahasa tidak ditemukan"
    nav = re.search(r"\.nav-link\s*\{[^}]*font-size:\s*(\d+)px", css, re.S)
    assert nav, "ukuran font nav-link tidak ditemukan"
    assert int(size.group(1)) >= int(nav.group(1)), (
        f"ukuran teks tombol bahasa {size.group(1)}px lebih kecil dari nav-link "
        f"{nav.group(1)}px; DESIGN bagian 3 melarang pengurangan ukuran teks"
    )


def test_confidence_level_selectors_match_the_server_values() -> None:
    """Kelas CSS harus mengikuti nilai server, kalau tidak warnanya hilang."""
    from bacteriacv.visualize import confidence_level

    css = (STATIC_DIR / "main.css").read_text(encoding="utf-8")

    for value in (0.0, 0.7, 0.95):
        level = confidence_level(value)
        assert f".level-{level} {{" in css, f"kelas .level-{level} tidak ada di main.css"


def test_data_labels_are_translated_only_for_display() -> None:
    """label_cocci dan label_bacilli boleh berbeda, tapi data server tidak."""
    dicts = _dictionaries()

    assert dicts["en"]["label_cocci"] == "Cocci"
    assert dicts["id"]["label_cocci"] == "Kokus"
    assert dicts["en"]["label_bacilli"] == "Bacilli"
    assert dicts["id"]["label_bacilli"] == "Batang"

    from bacteriacv.config import GRAM_LABELS, SHAPE_LABELS

    assert SHAPE_LABELS == ("cocci", "bacilli")
    assert GRAM_LABELS == ("positive", "negative")

    js = MAIN_JS_PATH.read_text(encoding="utf-8")
    assert '`label_${' in js, "klien tidak memetakan label untuk tampilan"


def test_every_page_i18n_key_exists_in_the_dictionary() -> None:
    """Setiap data-i18n di index.html harus ada kuncinya di kedua bahasa.

    Inilah yang membuat ganti bahasa benar-benar mengubah semua teks statis,
    bukan hanya sebagian.
    """
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    english, indonesian = _dictionary_keys()

    used = set(re.findall(r'data-i18n(?:-aria-label)?="([^"]+)"', page))
    assert used, "tidak ada data-i18n di index.html; ganti bahasa tidak mengubah apa pun"

    missing = sorted(k for k in used if k not in english or k not in indonesian)
    assert not missing, f"data-i18n di halaman tapi tidak ada di kamus: {missing}"


def test_language_switcher_preserves_the_locked_markup() -> None:
    """Empat hal yang dikunci tes lain tidak boleh berubah oleh tombol bahasa."""
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="tabs"' in page
    assert 'id="panel"' in page
    assert 'id="submit"' in page
    assert 'rel="icon"' in page

    images = re.findall(r"<img\b[^>]*>", page)
    panels = [tag for tag in images if 'class="brand-logo"' not in tag]
    assert len(panels) == 1, f"harus ada tepat satu img panel, ada {len(panels)}"
    logos = [tag for tag in images if 'class="brand-logo"' in tag]
    assert len(logos) == 1, f"harus ada tepat satu img logo, ada {len(logos)}"

    assert page.count("<script") == 1, "jumlah tag script harus tetap satu"
    assert "<style" not in page, "tidak boleh ada blok style inline"
    assert page.count('name="viewport"') == 1, "meta viewport harus tetap satu"

    # Kamus dimuat sebagai modul dari main.js, bukan dengan tag kedua.
    assert 'import("/static/i18n.js")' in MAIN_JS_PATH.read_text(encoding="utf-8"), (
        "i18n.js harus dimuat lewat import dinamis, bukan tag script kedua"
    )