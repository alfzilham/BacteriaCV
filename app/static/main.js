
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

let state = {
  panels: [],
  titles: [],
  names: [],
  stage: 0,
  prediction: null
};

// Fetch the upload limit from /api/health
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
    fileNameEl.textContent = "Belum ada berkas yang dipilih";
    submitEl.disabled = true;
    return;
  }

  // Validate the file format exactly per DESIGN.md Section 7
  if (!isAllowedFile(file.name)) {
    showError("Format berkas tidak didukung. Gunakan PNG, JPG, atau TIFF.");
    submitEl.disabled = true;
    fileNameEl.textContent = file.name;
    return;
  }

  // Validate the file size exactly per DESIGN.md Section 7 (20 MB = 20 * 1024 * 1024)
  if (file.size > 20 * 1024 * 1024) {
    showError("Ukuran berkas melebihi 20 MB.");
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
  submitEl.textContent = "MENGANALISIS...";

  try {
    const body = new FormData();
    body.append("file", file);

    const response = await fetch(`/api/predict?stage=${state.stage}`, {
      method: "POST",
      body,
    });

    const data = await response.json();
    if (!response.ok) {
      let msg = data.detail || "Permintaan gagal";
      // Error message mapping exactly per DESIGN.md Section 7
      if (response.status === 415 || msg.includes("tidak didukung")) {
        msg = "Format berkas tidak didukung. Gunakan PNG, JPG, atau TIFF.";
      } else if (response.status === 413 || msg.includes("melebihi batas")) {
        msg = "Ukuran berkas melebihi 20 MB.";
      } else if (msg.includes("tidak dapat dibaca")) {
        msg = "Citra tidak dapat dibaca. Coba unggah berkas lain.";
      }
      throw new Error(msg);
    }

    render(data);
  } catch (error) {
    showError(error.message);
  } finally {
    submitEl.disabled = false;
    submitEl.textContent = "Analisis Citra";
  }
});

function formatConfidence(confidence, level) {
  const pct = (confidence * 100).toFixed(1);
  // Make sure the High, Medium, Low text is explicitly capitalised (DESIGN 2.1)
  const capLevel = level.charAt(0).toUpperCase() + level.slice(1);
  return `<span class="level-${level}">${pct}% (${capLevel})</span>`;
}

function render(data) {
  state.panels = data.panels;
  state.names = data.stage_names;
  state.titles = data.stage_titles;
  state.prediction = data.prediction;
  state.stage = data.stage_index;

  buildTabs(state.names, data.failed_stages);
  showStage(state.stage);

  const p = data.prediction;
  shapeEl.textContent = p.shape_label;
  gramEl.textContent = p.gram_label;
  shapeMetaEl.innerHTML = formatConfidence(p.shape_confidence, p.shape_level);
  gramMetaEl.innerHTML = formatConfidence(p.gram_confidence, p.gram_level);

  // Show a warning if confidence is low on either head
  if (p.shape_level === "rendah" || p.gram_level === "rendah") {
    lowConfidenceWarningEl.hidden = false;
  } else {
    lowConfidenceWarningEl.hidden = true;
  }

  // Show a warning if segmentation failed exactly per DESIGN.md Section 7
  if (p.segmentation_ok === false || (data.failed_stages && data.failed_stages.length > 0)) {
    showError("Segmentasi tidak berhasil. Hasil klasifikasi tetap ditampilkan tanpa visualisasi.");
  }

  // Display the pipeline notes
  notesEl.innerHTML = "";
  const notes = [...data.notes];
  if (data.disclaimer && !notes.includes(data.disclaimer)) {
    notes.push(data.disclaimer);
  }
  notes.forEach(note => {
    const li = document.createElement("li");
    li.textContent = note;
    notesEl.appendChild(li);
  });

  resultEl.hidden = false;
}

function buildTabs(names, failed) {
  tabsEl.innerHTML = "";
  names.forEach((name, index) => {
    const button = document.createElement("button");
    button.textContent = name;
    button.setAttribute("aria-pressed", String(index === state.stage));
    if (failed && failed.includes(name)) {
      button.classList.add("failed");
    }
    button.addEventListener("click", () => showStage(index));
    tabsEl.appendChild(button);
  });
}

function showStage(index) {
  state.stage = index;
  const title = state.titles[index] || state.names[index];
  panelEl.src = `data:image/png;base64,${state.panels[index]}`;
  panelEl.alt = `Panel visualisasi ${title}`;
  panelEl.style.display = "block";
  if (panelPlaceholderEl) panelPlaceholderEl.style.display = "none";
  stageTitleEl.textContent = title;

  Array.from(tabsEl.children).forEach((btn, i) => {
    btn.setAttribute("aria-pressed", String(i === index));
  });
}

function showError(message) {
  if (!message) {
    errorEl.hidden = true;
    errorEl.textContent = "";
    return;
  }
  errorEl.hidden = false;
  errorEl.textContent = message;
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
  reportContent.textContent = "Memuat laporan dari /api/report...";
  try {
    const res = await fetch("/api/report");
    if (!res.ok) throw new Error("Laporan evaluasi belum tersedia pada server.");
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
