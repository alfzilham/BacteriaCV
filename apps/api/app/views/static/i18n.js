// Two-language dictionary for the BacteriaCV interface.
//
// The server never knows about languages: it always returns English data plus
// keys. This file is the only place a translation exists.
//
// Loaded as an ES module by main.js through a dynamic import, not through a
// second <script> tag. index.html must keep exactly one script tag, so a second
// one would break a locked test.
//
// Rules for the keys:
//   - Both dictionaries have exactly the same key set. A missing key would make
//     one language fall through to raw English, which is a silent bug, so the
//     test suite asserts the two key sets are identical.
//   - No key may have an empty value in either language, for the same reason.
//   - Data keys (label_*, level_*, stage_*, and the note keys) are display-only.
//     The server data itself is never translated: this dictionary changes what
//     is shown, not what the model returned.
//
// label_cocci and label_bacilli are the one place data labels are translated
// for display. Indonesian speakers say "kokus" and "batang"; the server still
// sends "cocci" and "bacilli" in both modes.

export const I18N = {
  en: {
    // --- document ---
    ui_page_title: "BacteriaCV // Bacterial Morphology & Gram Classification",

    // --- header navigation ---
    ui_nav_upload: "Upload",
    ui_nav_history: "History",
    ui_nav_about: "About",
    ui_report: "Report",
    ui_close: "Close",

    // --- upload card ---
    ui_upload_title: "Upload Microscope Image",
    ui_dropzone_label: "Choose or Drop Image Here",
    ui_dropzone_format_prefix: "Format PNG, JPG, or TIFF (Max.",
    ui_dropzone_format_suffix: "MB)",
    ui_no_file: "No file selected yet",
    ui_analyze: "Analyze Image",
    ui_analyzing: "ANALYZING...",

    // --- result summary card ---
    ui_summary_title: "Classification Summary",
    ui_shape_label: "Cell Shape (Head A)",
    ui_gram_label: "Gram Status (Head B)",
    ui_waiting: "Waiting for analysis",

    // --- main area ---
    ui_analysis_heading: "Morphology & Gram Status Analysis",
    ui_analysis_desc:
      "Simultaneous classification of cell shape (cocci, bacilli) and Gram status (positive, negative) from a DIBaS microscope image using a dual ResNet-50 backbone.",
    ui_notes_title: "Pipeline Notes & Limitations",

    // --- visualisation panel ---
    ui_visualization_title: "Segmentation Visualisation",
    ui_panel_alt: "Segmentation visualisation panel for the image",
    ui_panel_placeholder:
      '[ Image Visualisation Area // Upload a microscope image and press "Analyze Image" ]',
    ui_control_label: "Control Panel: Display & Contour Options",
    ui_no_image_analyzed: "No image analyzed yet.",

    // --- segmentation limitation box ---
    ui_limit_title: "Segmentation Limitation (SPEC 8.5):",
    ui_limit_body:
      "Morphological segmentation separates bacterial groups but not individual cells, because cells touch each other in 100x Gram images. The intraspecies range 1.03 to 4.19 exceeds the between-group difference of 0.068, so shape classification cannot be validated from the segmented objects.",

    // --- footer chips ---
    ui_chip_format: "Format:",
    ui_chip_limit: "Limit:",
    ui_chip_model: "Model:",
    ui_chip_dataset: "Dataset:",
    ui_chip_disclaimer: "Disclaimer:",

    // --- dialogs ---
    ui_report_title: "Model Evaluation Report",
    ui_report_loading: "Loading report from /api/report...",
    ui_report_missing: "The evaluation report is not available on the server.",
    ui_about_title: "About BacteriaCV",
    ui_about_body_1:
      "is a Computer Vision system that predicts bacterial cell shape (cocci, bacilli) and Gram status (positive, negative) simultaneously from a single digital microscope image using a ResNet-50 backbone and two classification heads.",
    ui_about_body_2:
      "The system is trained on the DIBaS dataset (Digital Images of Bacterial Species), which covers 32 bacterial species. It is a research and educational diagnostic aid, not a final medical diagnosis.",

    // --- language switcher ---
    ui_language_label: "Language",
    ui_lang_en: "EN",
    ui_lang_id: "ID",

    // --- errors ---
    error_request_failed: "Request failed",
    error_format: "File format not supported. Use PNG, JPG, or TIFF.",
    error_too_large: "File size exceeds 20 MB.",
    error_unreadable: "Image cannot be read. Try uploading another file.",
    error_segmentation:
      "Segmentation did not succeed. The classification result is still displayed without visualization.",

    // --- confidence levels, English on the wire ---
    level_high: "High",
    level_medium: "Medium",
    level_low: "Low",

    // --- data labels, display only ---
    label_cocci: "Cocci",
    label_bacilli: "Bacilli",
    label_positive: "Positive",
    label_negative: "Negative",

    // --- note keys sent by the server ---
    low_confidence_warning: "This result is an aid, not a diagnosis.",
    shape_not_validated:
      "Cell shape is not yet validated from segmentation: median elongation 1.30-1.43 for cocci and 1.64-1.76 for bacilli, but the intraspecies range 1.03-4.19 exceeds the between-group difference of 0.068.",
    segmentation_failure:
      "Segmentation works at the cell group level, not the individual cell; cells touch each other in 100x Gram images.",
    failed_stages_note:
      "Failed stages: {stages}. The related panels are empty, the classification is still computed.",

    // --- stage keys sent by the server ---
    stage_original: "Original image",
    stage_resized: "Resize 224 x 224",
    stage_normalized: "ImageNet normalization",
    stage_segment: "Segmentation mask",
    stage_watershed: "Object boundary",
    stage_failed_suffix: " (failed)"
  },

  id: {
    // --- document ---
    ui_page_title: "BacteriaCV // Sistem Klasifikasi Morfologi & Gram Bakteri",

    // --- header navigation ---
    ui_nav_upload: "Unggah",
    ui_nav_history: "Riwayat",
    ui_nav_about: "Tentang",
    ui_report: "Laporan",
    ui_close: "Tutup",

    // --- upload card ---
    ui_upload_title: "Unggah Citra Mikroskop",
    ui_dropzone_label: "Pilih atau Jatuhkan Citra di Sini",
    ui_dropzone_format_prefix: "Format PNG, JPG, atau TIFF (Maks.",
    ui_dropzone_format_suffix: "MB)",
    ui_no_file: "Belum ada berkas yang dipilih",
    ui_analyze: "Analisis Citra",
    ui_analyzing: "MENGANALISIS...",

    // --- result summary card ---
    ui_summary_title: "Ringkasan Klasifikasi",
    ui_shape_label: "Bentuk Sel (Head A)",
    ui_gram_label: "Status Gram (Head B)",
    ui_waiting: "Menunggu analisis",

    // --- main area ---
    ui_analysis_heading: "Analisis Morfologi & Status Gram",
    ui_analysis_desc:
      "Klasifikasi simultan bentuk sel (kokus, batang) dan status Gram (positif, negatif) dari citra mikroskop DIBaS menggunakan backbone ResNet-50 ganda.",
    ui_notes_title: "Catatan & Keterbatasan Pipeline",

    // --- visualisation panel ---
    ui_visualization_title: "Visualisasi Segmentasi",
    ui_panel_alt: "Panel visualisasi segmentasi citra",
    ui_panel_placeholder:
      '[ Area Visualisasi Citra // Unggah citra mikroskop dan tekan "Analisis Citra" ]',
    ui_control_label: "Panel Kontrol: Opsi Tampilan & Kontur",
    ui_no_image_analyzed: "Belum ada citra yang dianalisis.",

    // --- segmentation limitation box ---
    ui_limit_title: "Batasan Segmentasi (SPEC 8.5):",
    ui_limit_body:
      "Segmentasi morfologis memisahkan kelompok bakteri tetapi bukan sel individual, karena sel bersentuhan pada citra Gram 100x. Rentang intraspesies 1,03 sampai 4,19 melampaui selisih antargrup 0,068, sehingga klasifikasi bentuk tidak dapat divalidasi dari objek hasil segmentasi.",

    // --- footer chips ---
    ui_chip_format: "Format:",
    ui_chip_limit: "Batas:",
    ui_chip_model: "Model:",
    ui_chip_dataset: "Dataset:",
    ui_chip_disclaimer: "Disclaimer:",

    // --- dialogs ---
    ui_report_title: "Laporan Evaluasi Model",
    ui_report_loading: "Memuat laporan dari /api/report...",
    ui_report_missing: "Laporan evaluasi belum tersedia pada server.",
    ui_about_title: "Tentang BacteriaCV",
    ui_about_body_1:
      "adalah sistem Computer Vision yang memprediksi bentuk sel (kokus, batang) dan status Gram (positif, negatif) bakteri secara simultan dari satu citra mikroskop digital dengan backbone ResNet-50 dan dua classification head.",
    ui_about_body_2:
      "Sistem dilatih menggunakan dataset DIBaS (Digital Images of Bacterial Species) yang mencakup 32 spesies bakteri. Sistem ini merupakan instrumen alat bantu diagnostik penelitian dan pendidikan, bukan diagnosis medis final.",

    // --- language switcher ---
    ui_language_label: "Bahasa",
    ui_lang_en: "EN",
    ui_lang_id: "ID",

    // --- errors ---
    error_request_failed: "Permintaan gagal",
    error_format: "Format berkas tidak didukung. Gunakan PNG, JPG, atau TIFF.",
    error_too_large: "Ukuran berkas melebihi 20 MB.",
    error_unreadable: "Citra tidak dapat dibaca. Coba unggah berkas lain.",
    error_segmentation:
      "Segmentasi tidak berhasil. Hasil klasifikasi tetap ditampilkan tanpa visualisasi.",

    // --- confidence levels, English on the wire ---
    level_high: "Tinggi",
    level_medium: "Sedang",
    level_low: "Rendah",

    // --- data labels, display only ---
    label_cocci: "Kokus",
    label_bacilli: "Batang",
    label_positive: "Positif",
    label_negative: "Negatif",

    // --- note keys sent by the server ---
    low_confidence_warning: "Hasil ini sebagai alat bantu, bukan diagnosis.",
    shape_not_validated:
      "Bentuk sel belum tervalidasi dari segmentasi: elongasi median kokus 1.30-1.43 dan batang 1.64-1.76, tetapi rentang intraspesies 1.03-4.19 melampaui selisih antargrup 0.068.",
    segmentation_failure:
      "Segmentasi pada level kelompok sel, bukan sel individual; sel bersentuhan pada citra Gram 100x.",
    failed_stages_note:
      "Tahap gagal: {stages}. Panel terkait dikosongkan, klasifikasi tetap dihitung.",

    // --- stage keys sent by the server ---
    stage_original: "Citra asal",
    stage_resized: "Resize 224 x 224",
    stage_normalized: "Normalisasi ImageNet",
    stage_segment: "Mask segmentasi",
    stage_watershed: "Batas objek",
    stage_failed_suffix: " (gagal)"
  }
};

export const STORAGE_KEY = "bacteriacv.lang";
export const DEFAULT_LANG = "en";
export const LANGUAGES = ["en", "id"];

// Resolve a key for a language. A dotted key means the server flagged the item
// as failed, so the language's own failure suffix is appended to its base title.
export function translate(lang, key, params) {
  const dict = I18N[lang] || I18N[DEFAULT_LANG];
  let value = dict[key];
  if (value === undefined && key.indexOf(".") !== -1) {
    const parts = key.split(".");
    const baseValue = dict[parts[0]];
    const suffixValue = dict[parts[1]];
    if (baseValue !== undefined && suffixValue !== undefined) {
      value = baseValue + suffixValue;
    }
  }
  if (value === undefined) {
    value = I18N[DEFAULT_LANG][key];
  }
  if (value === undefined) {
    return key;
  }
  if (params) {
    Object.keys(params).forEach(name => {
      value = value.split("{" + name + "}").join(params[name]);
    });
  }
  return value;
}

// Same as translate, but takes the server text as the fallback. Used for notes:
// a client that ships no dictionary still renders the English the server sent.
export function translateWithFallback(lang, key, text, params) {
  const value = translate(lang, key, params);
  return value === key && text !== undefined ? text : value;
}