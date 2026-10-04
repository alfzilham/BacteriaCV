"""Tes untuk antarmuka web BacteriaCV.

Tes memakai TestClient FastAPI dengan checkpoint sementara. Checkpoint sengaja
dibuat dengan bobot head yang ditentukan, bukan acak, supaya hasil prediksi
dapat diassert tanpa bergantung pada inisialisasi.
"""

from __future__ import annotations

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
    assert payload["prediction"]["gram_label"] == "positif"
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
    assert len(payload["stage_titles"]) == 5
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
    assert any("belum tervalidasi" in note for note in payload["notes"])
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
    assert "tidak didukung" in response.json()["detail"]


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
    assert "citra" in response.json()["detail"]


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
