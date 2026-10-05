"""Tests for the BacteriaCV web interface.

The tests use the FastAPI TestClient with a temporary checkpoint. The checkpoint is
deliberately built with fixed head weights rather than random ones, so a prediction
result can be asserted without depending on initialisation.
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
    """Build a PNG image in memory."""
    import cv2

    array = np.full((240, 320, 3), 215, dtype=np.uint8)
    if dark:
        array[80:180, 90:230] = (20, 20, 220)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    """A head checkpoint that always classifies as Gram positive cocci."""
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
    """A test client with a temporary checkpoint."""
    import app.main as app_module

    application = app_module.create_app(checkpoint)
    with TestClient(application) as test_client:
        yield test_client


# --- Page and status ---


def test_index_page_is_served(client: TestClient) -> None:
    """The main page must open."""
    response = client.get("/")

    assert response.status_code == 200
    assert "BacteriaCV" in response.text


def test_health_reports_checkpoint(client: TestClient) -> None:
    """The status endpoint must name the checkpoint that is loaded."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "siap"
    assert response.json()["checkpoint"] == "heads.pt"
    assert response.json()["max_upload_bytes"] == MAX_UPLOAD_BYTES


# --- Prediction ---


def test_predict_returns_labels_confidence_and_panel(client: TestClient) -> None:
    """The response must carry the prediction, the panel, and the notes."""
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
    """The stage toggle needs all five panels and their titles."""
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
    """The stage parameter decides which panel is sent."""
    response = client.post(
        "/api/predict?stage=3",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    payload = response.json()
    assert payload["stage_index"] == 3
    assert payload["panel"] == payload["panels"][3]
    assert payload["stage_name"] == "segment"


def test_predict_always_states_segmentation_not_validated(client: TestClient) -> None:
    """The segmentation limitation note must be in the response."""
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
    """An image without a dark area must still get an answer."""
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
    """A non image file must be rejected with code 415."""
    response = client.post(
        "/api/predict",
        files={"file": (BAD_SUFFIX, b"bukan citra", "text/plain")},
    )

    assert response.status_code == 415
    assert "not supported" in response.json()["detail"]


def test_predict_rejects_empty_file(client: TestClient) -> None:
    """An empty file must be rejected with code 400."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, b"", "image/png")},
    )

    assert response.status_code == 400


def test_predict_rejects_unreadable_image(client: TestClient) -> None:
    """A broken PNG payload must be rejected with code 400."""
    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, b"bukan png sama sekali", "image/png")},
    )

    assert response.status_code == 400
    assert "image" in response.json()["detail"]


def test_predict_rejects_oversized_file(client: TestClient) -> None:
    """A file over the limit must be rejected with code 413."""
    oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1024)

    response = client.post(
        "/api/predict",
        files={"file": (GOOD_IMAGE, oversized, "image/png")},
    )

    assert response.status_code == 413


def test_predict_rejects_out_of_range_stage(client: TestClient) -> None:
    """A stage out of range must be rejected."""
    response = client.post(
        "/api/predict?stage=99",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert "Stage" in response.json()["detail"]


def test_predict_rejects_non_numeric_stage(client: TestClient) -> None:
    """A stage that is not a number must be rejected."""
    response = client.post(
        "/api/predict?stage=abc",
        files={"file": (GOOD_IMAGE, _png_bytes(), "image/png")},
    )

    assert response.status_code == 400


def test_predict_is_deterministic(client: TestClient) -> None:
    """The same image must produce the same prediction."""
    payload = _png_bytes()

    first = client.post(
        "/api/predict", files={"file": (GOOD_IMAGE, payload, "image/png")}
    ).json()["prediction"]
    second = client.post(
        "/api/predict", files={"file": (GOOD_IMAGE, payload, "image/png")}
    ).json()["prediction"]

    assert first["shape_label"] == second["shape_label"]
    assert first["shape_confidence"] == second["shape_confidence"]


# --- Evaluation report ---


def test_report_returns_404_before_evaluation(client: TestClient, monkeypatch) -> None:
    """A missing report must give 404, not a server error."""
    import app.main as app_module

    monkeypatch.setattr(app_module, "CHECKPOINT_DIR", REPO_ROOT / "tidak_ada")

    response = client.get("/api/report")

    assert response.status_code == 404


def test_report_returns_evaluation_file(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """The evaluation report must be served as it stands."""
    import json

    import app.main as app_module

    monkeypatch.setattr(app_module, "CHECKPOINT_DIR", tmp_path)
    (tmp_path / app_module.EVALUATION_NAME).write_text(
        json.dumps({"shape": {"f1_macro": 0.9}}), encoding="utf-8"
    )

    response = client.get("/api/report")

    assert response.status_code == 200
    assert response.json()["shape"]["f1_macro"] == 0.9


# --- Project specifics ---


def test_index_html_has_one_panel_with_toggle() -> None:
    """DESIGN section 3: one panel with a toggle, not five panels.

    The file must have exactly one img element for the panel and one toggle
    container. Five img elements would break the one panel design.
    """
    page = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")

    # The brand logo also uses <img> and is not a panel, so what is counted
    # is the img elements other than the logo. The count must still be one.
    images = re.findall(r"<img\b[^>]*>", page)
    panels = [tag for tag in images if 'class="brand-logo"' not in tag]
    assert len(panels) == 1, f"harus ada tepat satu img panel, ada {len(panels)}"

    # Without this, two logo images are filtered out and the panel count stays one.
    logos = [tag for tag in images if 'class="brand-logo"' in tag]
    assert len(logos) == 1, f"harus ada tepat satu img logo, ada {len(logos)}"

    assert 'id="tabs"' in page
    assert 'id="panel"' in page


def test_static_css_and_js_are_external_files() -> None:
    """CSS and JavaScript must live in their own static files.

    The main.css and main.js files must exist and be non empty, index.html must
    reference both through the absolute /static path, and no leftover inline
    <style> or <script> block may remain.

    The absolute path is required: FastAPI mounts static at /static, so a
    relative reference such as href="main.css" returns 404. The page then
    renders with no styling and no interaction, and no existing test would
    fail because of it.
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

    # No inline blocks. The only script tag is the external reference, so a
    # <script>...</script> block holding code cannot exist.
    assert "<style" not in page
    assert page.count("<script") == 1
    assert "</script>" in page and "main.js" in page


# The contrast threshold for non text UI components, not for text. See
# DESIGN.md section 2.1: the scrollbar is a UI component, so what applies
# is WCAG 1.4.11 Non-text Contrast at a 3:1 threshold, not WCAG 1.4.3 which
# is 4.5:1 and applies to text only.
SCROLLBAR_MIN_CONTRAST = 3.0


def _srgb_to_linear(channel: int) -> float:
    """Linearise one sRGB channel 0-255, the WCAG 2.1 formula."""
    c = channel / 255
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_colour: str) -> float:
    """The WCAG relative luminance of a #RRGGBB colour."""
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * _srgb_to_linear(r)
            + 0.7152 * _srgb_to_linear(g)
            + 0.0722 * _srgb_to_linear(b))


def _contrast_ratio(first: str, second: str) -> float:
    """The WCAG contrast ratio (hi + 0.05) / (lo + 0.05)."""
    a = _relative_luminance(first)
    b = _relative_luminance(second)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def _design_tokens(css: str) -> dict[str, str]:
    """Collect every --name: value definition from the :root block."""
    root = re.search(r":root\s*\{([^}]*)\}", css)
    assert root, "blok :root tidak ditemukan di main.css"
    tokens = dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", root.group(1)))
    assert tokens, "blok :root tidak mendefinisikan custom property"
    return tokens


def _resolve(value: str, tokens: dict[str, str], limit: int = 10) -> str:
    """Replace every var(--name) with its value, repeating until no var() is left.

    limit caps the iterations so tokens that point at each other cannot loop.
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
    """Read one property from one CSS block, then resolve its tokens.

    The lookbehind stops property_background from matching background-color.
    """
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"blok {selector} tidak ditemukan di main.css"
    decl = re.search(rf"(?<![-\w]){re.escape(prop)}\s*:\s*([^;]+);", match.group(1))
    assert decl, f"{selector} tidak mendeklarasikan properti {prop}"
    return _resolve(decl.group(1), _design_tokens(css))


def _declared_background(css: str, selector: str) -> str:
    """The background value of one CSS block, after var(--) resolution."""
    value = _declared_value(css, selector, "background")
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", value), (
        f"{selector} tidak menghasilkan warna hex, dapat {value!r}"
    )
    return value


# The #EAE8E3 track is only 1.11:1 against the #F4F4F0 page background, so
# without a delimiter it is practically invisible. The delimiter threshold
# uses the same standard as the thumb threshold, namely 3:1.
SCROLLBAR_TRACK_MIN_CONTRAST = 3.0


def _colour_in_shorthand(shorthand: str) -> str:
    """Take the hex colour out of one shorthand, for example "2px solid #050505"."""
    found = re.search(r"#[0-9A-Fa-f]{6}", shorthand)
    assert found, f"shorthand tidak memuat warna hex, dapat {shorthand!r}"
    return found.group(0)


def test_scrollbar_track_has_visible_delimiter() -> None:
    """The scrollbar track must have a delimiter of at least 3:1 contrast.

    The #EAE8E3 track is only 1.11:1 against the #F4F4F0 page background.
    Without a delimiter the track merges into the page and disappears. This
    test demands the delimiter exists and that its colour contrasts at least
    3:1 against the track.
    This test deliberately does not lock the delimiter value. If the track is
    darkened later as a legitimate alternative, only the delimiter changes;
    what stays required is that the track has a visible delimiter.
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
    """The scrollbar contrast must stay above the 3:1 threshold.

    DESIGN.md section 2.1 sets the 3:1 threshold because the scrollbar is a
    UI component, not text. This test recomputes the ratio from
    main.css so the next UI revision cannot lower it unnoticed.
    ketahuan.

    The scrollbar colours are written as var(--token), so the values are read
    through the token resolver rather than as raw hex. The values that must
    be readable after resolution:
      scrollbar-color #050505 #EAE8E3, track #EAE8E3, thumb #050505,
      hover #D31515.
    """
    css = (REPO_ROOT / "app" / "static" / "main.css").read_text(encoding="utf-8")

    # 1. the scrollbar-color property on the html block
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

    # 2. the background of track, thumb, and hover
    track = _declared_background(css, "::-webkit-scrollbar-track")
    thumb = _declared_background(css, "::-webkit-scrollbar-thumb")
    hover = _declared_background(css, "::-webkit-scrollbar-thumb:hover")

    # 3. compute the ratio for each pair
    thumb_vs_track = _contrast_ratio(thumb, track)
    hover_vs_track = _contrast_ratio(hover, track)
    shorthand_vs_track = _contrast_ratio(shorthand_thumb, shorthand_track)

    # 4. enforce the 3:1 threshold
    for label, ratio in (
        (f"thumb {thumb} di track {track}", thumb_vs_track),
        (f"hover {hover} di track {track}", hover_vs_track),
        (f"scrollbar-color {shorthand_thumb} di {shorthand_track}", shorthand_vs_track),
    ):
        assert ratio >= SCROLLBAR_MIN_CONTRAST, (
            f"kontras {label} cuma {ratio:.2f}:1, di bawah ambang "
            f"{SCROLLBAR_MIN_CONTRAST}:1"
        )

    # Firefox and WebKit must use the same colour pair, otherwise the
    # scrollbar would look different across browsers.
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
    """The hidden attribute must win over a display written by .dialog-backdrop.

    .dialog-backdrop declares display: flex, and an author level display
    declaration always beats the browser default [hidden] rule.
    That kills the hidden instruction completely, both dialogs show from
    page load, and nothing can close them.

    So main.css must carry a global [hidden] rule with
    display: none !important, and that rule must appear before
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

    # This is why the rule is needed: the backdrop really does write display.
    backdrop = re.search(r"\.dialog-backdrop\s*\{([^}]*)\}", css)
    assert backdrop, "aturan .dialog-backdrop tidak ditemukan di main.css"
    assert re.search(r"display\s*:\s*(?!none)", backdrop.group(1)), (
        ".dialog-backdrop tidak punya deklarasi display eksplisit, "
        "jadi aturan [hidden] tidak diuji oleh keadaan sebenarnya"
    )


def test_index_html_has_no_hardcoded_metric_claims() -> None:
    """The page must not promise figures before the evaluation.

    The figures on the page must come from the server response, not be written
    into the markup, so a model change cannot leave a stale claim behind.
    """
    page = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")

    for claim in ("95%", "99%", "akurasi 9"):
        assert claim not in page


def test_app_module_does_not_import_label_map() -> None:
    """Species labels must not leak into the interface.

    At inference the species is not known. If the app imported the lookup
    table, a developer could add a species guess to the response by accident.
    """
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

    assert "label_map" not in source
    assert "LOOKUP" not in source

# ===========================================================================
# Two languages and canonical data
#
# The property that holds this section up: the server never changes its data
# following the caller's language. If it did, evaluation.json and the whole
# report could show different numbers depending on who opened it, and that
# is unacceptable for a document that serves as a case study reference.
#
# The dictionary is read by a small reader of its own, not a JavaScript
# engine, so these tests need no new dependency and no runtime.
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

    # JavaScript comments are not valid Python, so they are dropped first. Only
    # whole line comments occur in the dictionary, so this cannot touch a value.
    body = re.sub(r"(?m)^\s*//.*$", "", body)
    # Unquoted keys are not valid Python either.
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
    """Return both dictionaries, indexed by language code."""
    return _i18n_object(I18N_PATH.read_text(encoding="utf-8"), "I18N")


def _dictionary_keys() -> tuple:
    """Return (english keys, indonesian keys)."""
    dicts = _dictionaries()
    return set(dicts["en"]), set(dicts["id"])


# --- 1. the dictionary file exists and is readable -----------------------------
def test_i18n_file_exists_and_is_readable() -> None:
    """app/static/i18n.js must exist, be non empty, and read as two dictionaries."""
    assert I18N_PATH.is_file(), "app/static/i18n.js tidak ada"
    assert I18N_PATH.stat().st_size > 0, "app/static/i18n.js kosong"

    dicts = _dictionaries()

    assert set(dicts) == set(LANGUAGES), f"harus ada dua bahasa, ada {set(dicts)}"
    for lang in LANGUAGES:
        assert dicts[lang], f"kamus {lang} kosong"
        for key, value in dicts[lang].items():
            assert isinstance(value, str), f"{lang}.{key} bukan string"


# --- 2. both languages have an identical key set -----------------------------
def test_both_languages_have_identical_keys() -> None:
    """A key missing in one language would silently fall through to English."""
    english, indonesian = _dictionary_keys()

    assert english, "kamus bahasa Inggris kosong"
    assert indonesian, "kamus bahasa Indonesia kosong"
    assert english == indonesian, (
        "kunci tidak sama di kedua bahasa. "
        f"hanya di en: {sorted(english - indonesian)}; "
        f"hanya di id: {sorted(indonesian - english)}"
    )


# --- 3. no key has an empty value -------------------------------------------
def test_no_key_has_an_empty_value() -> None:
    """An empty key leaves a UI element blank with no visible error."""
    dicts = _dictionaries()
    empty = []

    for lang in LANGUAGES:
        for key, value in dicts[lang].items():
            if not value.strip():
                empty.append(f"{lang}.{key}")

    assert not empty, f"nilai kunci kosong: {empty}"


# --- 4. every server key exists in both languages -----------------------------
def test_every_server_note_and_stage_key_exists_in_both_languages() -> None:
    """note_keys and stage_keys from the server must be translatable."""
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
    # The failure suffix is addressed by name inside a dotted key, so it must
    # exist as a key in its own right too.
    stage_keys = set(STAGE_KEYS.values()) | {"stage_failed_suffix"}

    for key in sorted(note_keys | stage_keys):
        assert key in english, f"kunci server {key} tidak ada di en"
        assert key in indonesian, f"kunci server {key} tidak ada di id"


def test_stage_keys_cover_every_stage() -> None:
    """Every stage needs a key, otherwise its title cannot be translated."""
    from bacteriacv.preprocess import STAGE_NAMES
    from bacteriacv.visualize import STAGE_KEYS

    assert set(STAGE_NAMES) == set(STAGE_KEYS), (
        "STAGE_KEYS harus menutupi setiap tahap; "
        f"tanpa kunci: {sorted(set(STAGE_NAMES) - set(STAGE_KEYS))}"
    )

    english, _ = _dictionary_keys()
    outside = sorted(set(STAGE_KEYS.values()) - english)
    assert not outside, f"ada kunci tahap di luar kamus: {outside}"


# --- 5. the predict response returns keys, not bare translated text -----------
def test_predict_returns_note_keys_and_notes_text(client: TestClient) -> None:
    """The server returns keys and text, not text alone."""
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

    # The old field is replaced, not kept alongside, so each item has exactly
    # one canonical source of text.
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


# --- 6. server labels stay English whatever the Accept-Language header says ---
@pytest.mark.parametrize(
    "accept_language",
    ["id-ID,id;q=0.9", "en-US,en;q=0.9", "en", "id", "*", "", "de-DE,de;q=0.8"],
)
def test_labels_stay_english_under_any_accept_language(
    client: TestClient, accept_language: str
) -> None:
    """Data from the server must never change language.

    This is the test that proves the canonical data architecture does not leak.
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
    """No code path may read Accept-Language at all.

    Proving it statically is stronger than trusting one response, because this
    conversation cannot change without adding one of the patterns below.
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


# --- 7. the response value is identical whatever the Accept-Language header ---
def test_response_is_identical_under_any_accept_language(client: TestClient) -> None:
    """Two different headers must produce exactly the same response."""
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

    # Written one by one so a failure names the field that actually differs.
    assert (
        with_indonesian["prediction"]["gram_label"]
        == with_english["prediction"]["gram_label"]
    )
    assert with_indonesian["note_keys"] == with_english["note_keys"]
    assert with_indonesian["notes_text"] == with_english["notes_text"]
    assert with_indonesian["stage_keys"] == with_english["stage_keys"]
    assert with_indonesian["stage_texts"] == with_english["stage_texts"]


# --- language switcher -------------------------------------------------------
def test_language_toggle_buttons_exist_in_the_header() -> None:
    """Two small buttons in the header, next to the existing navigation buttons."""
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
    """English is the default because the README and docs are English now."""
    js = MAIN_JS_PATH.read_text(encoding="utf-8")

    assert 'DEFAULT_LANG = "en"' in js, "bahasa bawaan bukan English"
    assert "indexOf(stored) === -1" in js, (
        "nilai localStorage yang tidak dikenal harus jatuh ke English"
    )


def test_language_toggle_follows_the_brutalist_style_rules() -> None:
    """No border radius and no shadow, consistent with the existing header."""
    css = (STATIC_DIR / "main.css").read_text(encoding="utf-8")

    blocks = re.findall(r"\.lang-btn(?:\.[\w-]+)?\s*\{([^}]*)\}", css)
    assert blocks, "blok .lang-btn tidak ditemukan di main.css"

    for block in blocks:
        assert "border-radius" not in block, "tombol bahasa tidak boleh ada radius"
        assert "box-shadow" not in block, "tombol bahasa tidak boleh ada shadow"

    # The text size must not be smaller than the existing nav-link.
    size = re.search(r"\.lang-btn\s*\{[^}]*font-size:\s*(\d+)px", css, re.S)
    assert size, "ukuran font tombol bahasa tidak ditemukan"
    nav = re.search(r"\.nav-link\s*\{[^}]*font-size:\s*(\d+)px", css, re.S)
    assert nav, "ukuran font nav-link tidak ditemukan"
    assert int(size.group(1)) >= int(nav.group(1)), (
        f"ukuran teks tombol bahasa {size.group(1)}px lebih kecil dari nav-link "
        f"{nav.group(1)}px; DESIGN bagian 3 melarang pengurangan ukuran teks"
    )


def test_confidence_level_selectors_match_the_server_values() -> None:
    """The CSS classes must follow the server values, or the colour is lost."""
    from bacteriacv.visualize import confidence_level

    css = (STATIC_DIR / "main.css").read_text(encoding="utf-8")

    for value in (0.0, 0.7, 0.95):
        level = confidence_level(value)
        assert f".level-{level} {{" in css, f"kelas .level-{level} tidak ada di main.css"


def test_data_labels_are_translated_only_for_display() -> None:
    """label_cocci and label_bacilli may differ, but the server data must not."""
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
    """Every data-i18n in index.html must have its key in both languages.

    This is what makes a language change actually alter all static text,
    not just part of it.
    """
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    english, indonesian = _dictionary_keys()

    used = set(re.findall(r'data-i18n(?:-aria-label)?="([^"]+)"', page))
    assert used, "tidak ada data-i18n di index.html; ganti bahasa tidak mengubah apa pun"

    missing = sorted(k for k in used if k not in english or k not in indonesian)
    assert not missing, f"data-i18n di halaman tapi tidak ada di kamus: {missing}"


def test_language_switcher_preserves_the_locked_markup() -> None:
    """The four things other tests lock must not change because of the switcher."""
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

    # The dictionary is loaded as a module from main.js, not with a second tag.
    assert 'import("/static/i18n.js")' in MAIN_JS_PATH.read_text(encoding="utf-8"), (
        "i18n.js harus dimuat lewat import dinamis, bukan tag script kedua"
    )