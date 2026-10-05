// Required DOM element references
const tabsEl = document.getElementById("tabs");
const panelEl = document.getElementById("panel");
const panelPlaceholderEl = document.getElementById("panel-placeholder");
const stageTitleEl = document.getElementById("stage-title");
const fileEl = document.getElementById("file");
const fileNameEl = document.getElementById("file-name");
const submitEl = document.getElementById("submit");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const notesEl = document.getElementById("notes");
const shapeEl = document.getElementById("shape");
const gramEl = document.getElementById("gram");
const shapeMetaEl = document.getElementById("shape-meta");
const gramMetaEl = document.getElementById("gram-meta");
const lowConfidenceWarningEl = document.getElementById("low-confidence-warning");
const dropzoneEl = document.getElementById("dropzone");
const langBtnEn = document.getElementById("lang-en");
const langBtnId = document.getElementById("lang-id");

const state = {
  panels: [],
  titles: [],
  keys: [],
  names: [],
  stage: 0,
  prediction: null,
  noteKeys: [],
  notesText: [],
  failedStages: [],
  lang: "en",
  // Set while a request is in flight so the language can change underneath it
  // and the response still renders in whatever language is active when it lands.
  analyzing: false,
  // Remembered by key so a language switch re-renders the message.
  errorKey: null,
  errorText: ""
};

// The dictionary lives in its own module. It is imported dynamically rather than
// with a second <script> tag, because index.html must keep exactly one script tag.
let I18N = null;
let STORAGE_KEY = "bacteriacv.lang";
const DEFAULT_LANG = "en";
const LANGUAGES = ["en", "id"];

function translate(key, params) {
  if (!I18N) return key;
  const dict = I18N[state.lang] || I18N[DEFAULT_LANG];
  let value = dict[key];
  if (value === undefined && key.indexOf(".") !== -1) {
    const parts = key.split(".");
    const base = dict[parts[0]];
    const suffix = dict[parts[1]];
    if (base !== undefined && suffix !== undefined) value = base + suffix;
  }
  if (value === undefined) value = I18N[DEFAULT_LANG][key];
  if (value === undefined) return key;
  if (params) {
    Object.keys(params).forEach(name => {
      value = value.split("{" + name + "}").join(params[name]);
    });
  }
  return value;
}

// Display label for a data label that the server sent in canonical English.
// Only the display text is translated; the value itself is untouched.
function labelText(label) {
  const key = `label_${String(label).toLowerCase()}`;
  const value = translate(key);
  return value === key ? label : value;
}

function levelText(level) {
  const key = `level_${String(level).toLowerCase()}`;
  const value = translate(key);
  return value === key ? level : value;
}

// A note from the server: translate by key, fall back to the English text the
// server sent when this dictionary has no entry for that key.
function noteText(key, fallback, params) {
  const value = translate(key, params);
  if (value === key) return fallback === undefined ? key : fallback;
  return value;
}

// Apply the dictionary to every element carrying a data-i18n attribute.
function applyStaticText() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    el.textContent = translate(key);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach(el => {
    const key = el.getAttribute("data-i18n-aria-label");
    el.setAttribute("aria-label", translate(key));
  });
  document.title = translate("ui_page_title");
  document.documentElement.lang = state.lang;
}

// --- language persistence ---------------------------------------------------
function readStoredLang() {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return LANGUAGES.indexOf(stored) === -1 ? DEFAULT_LANG : stored;
  } catch (error) {
    return DEFAULT_LANG;
  }
}

function storeLang(lang) {
  try {
    window.localStorage.setItem(STORAGE_KEY, lang);
  } catch (error) {
    // A blocked localStorage must not break the switcher; the choice simply
    // does not survive a reload.
  }
}

// --- switching --------------------------------------------------------------
// Every switch re-renders from the stored result, so the classification, the
// panels and the notes survive the language change.
function setLanguage(lang) {
  if (LANGUAGES.indexOf(lang) === -1) lang = DEFAULT_LANG;
  state.lang = lang;
  storeLang(lang);
  syncLangButtons();
  applyStaticText();
  redraw();
}

function syncLangButtons() {
  const active = state.lang;
  [langBtnEn, langBtnId].forEach(btn => {
    const isActive = btn.dataset.lang === active;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-pressed", String(isActive));
  });
}

langBtnEn.addEventListener("click", () => setLanguage("en"));
langBtnId.addEventListener("click", () => setLanguage("id"));

// --- fetching the upload limit ----------------------------------------------
fetch("/api/health")
  .then(r => r.json())
  .then(d => {
    if (d.max_upload_bytes) {
      const mb = Math.round(d.max_upload_bytes / (1024 * 1024));
      const limitEl = document.getElementById("limit");
      if (limitEl) limitEl.textContent = mb;
    }
  })
  .catch(() => {});

// Validate the file format
function isAllowedFile(fileName) {
  const allowed = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];
  const lower = fileName.toLowerCase();
  return allowed.some(ext => lower.endsWith(ext));
}

// change event on the file input
fileEl.addEventListener("change", () => {
  const file = fileEl.files[0];
  showError("");

  if (!file) {
    fileNameEl.textContent = translate("ui_no_file");
    submitEl.disabled = true;
    return;
  }

  // Validate the file format exactly per DESIGN.md Section 7
  if (!isAllowedFile(file.name)) {
    showError(translate("error_format"), "error_format");
    submitEl.disabled = true;
    fileNameEl.textContent = file.name;
    return;
  }

  // Validate the file size exactly per DESIGN.md Section 7 (20 MB = 20 * 1024 * 1024)
  if (file.size > 20 * 1024 * 1024) {
    showError(translate("error_too_large"), "error_too_large");
    submitEl.disabled = true;
    fileNameEl.textContent = file.name;
    return;
  }

  fileNameEl.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  submitEl.disabled = false;
});

// Drag & Drop handling
dropzoneEl.addEventListener("dragover", e => {
  e.preventDefault();
  dropzoneEl.classList.add("dragover");
});

dropzoneEl.addEventListener("dragleave", () => {
  dropzoneEl.classList.remove("dragover");
});

dropzoneEl.addEventListener("drop", e => {
  e.preventDefault();
  dropzoneEl.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    fileEl.files = e.dataTransfer.files;
    fileEl.dispatchEvent(new Event("change"));
  }
});

// Submit / Analyse
submitEl.addEventListener("click", async () => {
  const file = fileEl.files[0];
  if (!file) return;

  showError("");
  submitEl.disabled = true;
  state.analyzing = true;
  submitEl.textContent = translate("ui_analyzing");

  try {
    const body = new FormData();
    body.append("file", file);

    const response = await fetch(`/api/predict?stage=${state.stage}`, {
      method: "POST",
      body,
    });

    const data = await response.json();
    if (!response.ok) {
      let msg = data.detail || translate("error_request_failed");
      // Error message mapping exactly per DESIGN.md Section 7. The server text
      // is English in every mode, so the match is on the English wording only.
      if (response.status === 415 || msg.indexOf("not supported") !== -1) {
        msg = translate("error_format");
      } else if (response.status === 413 || msg.indexOf("exceeds the") !== -1) {
        msg = translate("error_too_large");
      } else if (msg.indexOf("cannot be read") !== -1) {
        msg = translate("error_unreadable");
      }
      throw new Error(msg);
    }

    render(data);
  } catch (error) {
    // A message the dictionary owns is re-rendered on a language switch; one
    // that came straight from the server is stored as plain text.
    const owned = ["error_format", "error_too_large", "error_unreadable",
                   "error_request_failed", "error_segmentation"];
    const match = owned.find(key => error.message === translate(key));
    showError(error.message, match || null);
  } finally {
    state.analyzing = false;
    submitEl.disabled = false;
    submitEl.textContent = translate("ui_analyze");
  }
});

function formatConfidence(confidence, level) {
  const pct = (confidence * 100).toFixed(1);
  // The level arrives from the server as English data and is translated here.
  // The CSS class keeps the English key so the palette does not change.
  return `<span class="level-${level}">${pct}% (${levelText(level)})</span>`;
}

function render(data) {
  state.panels = data.panels;
  state.names = data.stage_names;
  state.keys = data.stage_keys;
  state.titles = data.stage_texts;
  state.prediction = data.prediction;
  state.stage = data.stage_index;
  state.noteKeys = data.note_keys || [];
  state.notesText = data.notes_text || data.notes || [];
  state.failedStages = data.failed_stages || [];

  redraw();

  // Show a warning if confidence is low on either head
  const p = state.prediction;
  if (p.shape_level === "low" || p.gram_level === "low") {
    lowConfidenceWarningEl.hidden = false;
  } else {
    lowConfidenceWarningEl.hidden = true;
  }

  // Show a warning if segmentation failed exactly per DESIGN.md Section 7
  if (p.segmentation_ok === false || state.failedStages.length > 0) {
    showError(translate("error_segmentation"), "error_segmentation");
  }

  resultEl.hidden = false;
}

// Re-render everything that depends on the language from the stored result.
// Called on every switch, so nothing is lost when the language changes.
function redraw() {
  syncLangButtons();
  redrawError();

  const p = state.prediction;
  if (!p) {
    shapeMetaEl.textContent = translate("ui_waiting");
    gramMetaEl.textContent = translate("ui_waiting");
    stageTitleEl.textContent = translate("ui_no_image_analyzed");
    return;
  }

  shapeEl.textContent = labelText(p.shape_label);
  gramEl.textContent = labelText(p.gram_label);
  shapeMetaEl.innerHTML = formatConfidence(p.shape_confidence, p.shape_level);
  gramMetaEl.innerHTML = formatConfidence(p.gram_confidence, p.gram_level);

  lowConfidenceWarningEl.textContent = translate("low_confidence_warning");
  lowConfidenceWarningEl.hidden = !(p.shape_level === "low" || p.gram_level === "low");

  buildTabs(state.names, state.failedStages);
  showStage(state.stage);
  renderNotes();
}

function renderNotes() {
  notesEl.innerHTML = "";
  const stages = state.failedStages.join(", ");
  const seen = [];
  state.noteKeys.forEach((key, index) => {
    const fallback = state.notesText[index];
    const params = key === "failed_stages_note" ? { stages } : null;
    const text = noteText(key, fallback, params);
    if (seen.indexOf(text) !== -1) return;
    seen.push(text);
    const li = document.createElement("li");
    li.textContent = text;
    notesEl.appendChild(li);
  });
}

function buildTabs(names, failed) {
  tabsEl.innerHTML = "";
  names.forEach((name, index) => {
    const button = document.createElement("button");
    button.textContent = translate(state.keys[index] || name);
    button.setAttribute("aria-pressed", String(index === state.stage));
    if (failed && failed.indexOf(name) !== -1) {
      button.classList.add("failed");
    }
    button.addEventListener("click", () => showStage(index));
    tabsEl.appendChild(button);
  });
}

function showStage(index) {
  state.stage = index;
  const key = state.keys[index];
  const title = translate(key || state.names[index]);
  panelEl.src = `data:image/png;base64,${state.panels[index]}`;
  panelEl.alt = `${translate("ui_panel_alt")} ${title}`;
  panelEl.style.display = "block";
  if (panelPlaceholderEl) panelPlaceholderEl.style.display = "none";
  stageTitleEl.textContent = title;

  Array.from(tabsEl.children).forEach((btn, i) => {
    btn.setAttribute("aria-pressed", String(i === index));
  });
}

// The active error is remembered by its i18n key, not by its rendered text, so
// switching the language re-renders it instead of leaving it stale.
function showError(message, key) {
  state.errorKey = key || null;
  state.errorText = key ? null : message || "";
  if (!message && !key) {
    errorEl.hidden = true;
    errorEl.textContent = "";
    return;
  }
  errorEl.hidden = false;
  errorEl.textContent = key ? translate(key) : message;
}

function redrawError() {
  if (state.errorKey) {
    errorEl.hidden = false;
    errorEl.textContent = translate(state.errorKey);
  } else if (state.errorText) {
    errorEl.hidden = false;
    errorEl.textContent = state.errorText;
  }
}

// Report & About dialogs
const reportDialog = document.getElementById("dialog-report");
const aboutDialog = document.getElementById("dialog-about");
const btnReport = document.getElementById("btn-report");
const btnAbout = document.getElementById("btn-about");
const btnCloseReport = document.getElementById("btn-close-report");
const btnCloseAbout = document.getElementById("btn-close-about");
const reportContent = document.getElementById("report-content");

btnReport.addEventListener("click", async () => {
  reportDialog.hidden = false;
  reportContent.textContent = translate("ui_report_loading");
  try {
    const res = await fetch("/api/report");
    if (!res.ok) throw new Error(translate("ui_report_missing"));
    const json = await res.json();
    reportContent.innerHTML = `<pre style="margin:0; overflow-x:auto;">${JSON.stringify(json, null, 2)}</pre>`;
  } catch (err) {
    reportContent.textContent = err.message;
  }
});

btnCloseReport.addEventListener("click", () => {
  reportDialog.hidden = true;
});

btnAbout.addEventListener("click", (e) => {
  e.preventDefault();
  aboutDialog.hidden = false;
});

btnCloseAbout.addEventListener("click", () => {
  aboutDialog.hidden = true;
});

// --- start ------------------------------------------------------------------
// Load the dictionary first, then paint. Nothing is rendered before I18N exists,
// so the page never flashes English into an Indonesian session.
import("/static/i18n.js")
  .then(module => {
    I18N = module.I18N;
    STORAGE_KEY = module.STORAGE_KEY;
    setLanguage(readStoredLang());
  })
  .catch(() => {
    // No dictionary: fall back to English, which is the canonical language and
    // also what the server sent.
    setLanguage(DEFAULT_LANG);
  });