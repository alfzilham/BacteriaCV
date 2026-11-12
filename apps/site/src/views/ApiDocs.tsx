"use client";

import { usePrefs } from "@/controllers/prefs";
import { Code } from "./DocBits";
import { DocsShell } from "./DocsShell";

const ENDPOINTS: [string, string, string][] = [
  ["GET", "/api/health", "status, checkpoint, max_upload_bytes"],
  ["POST", "/api/predict?stage=0", "multipart field file (PNG, JPG, TIFF, max 20 MB). stage 0 to 4."],
  ["GET", "/api/report", "evaluation.json, or 404 when no evaluation has been run"],
];

const FIELDS: [string, string][] = [
  ["prediction", "Shape and Gram labels with confidence values"],
  ["stage_index, stage_name, stage_title", "The selected stage"],
  ["stage_names", "The five stage keys in order"],
  ["panel", "The selected stage as a base64 PNG"],
  ["panels", "All five stages as base64 PNGs"],
  ["notes_text", "English notes about the result"],
  ["disclaimer", "The fixed low-confidence warning"],
  ["failed_stages", "Stages that could not be produced"],
];

const CURL = `curl -X POST "$API/api/predict?stage=3" \\
  -F "file=@sample.png"`;

const PYTHON = `import requests

with open("sample.png", "rb") as f:
    r = requests.post("http://localhost:8000/api/predict", params={"stage": 3}, files={"file": f})
r.raise_for_status()
data = r.json()
print(data["prediction"]["shape_label"], data["prediction"]["gram_label"])`;

const JS = `const form = new FormData();
form.append("file", fileInput.files[0]);
const res = await fetch("http://localhost:8000/api/predict?stage=3", { method: "POST", body: form });
const data = await res.json();`;

export function ApiDocs() {
  const { t } = usePrefs();
  return (
    <DocsShell page="api">
      <h1>{t("d_api_h")}</h1>
      <p>{t("d_api_d")}</p>
      <table>
        <thead><tr><th>Method</th><th>Path</th><th>Notes</th></tr></thead>
        <tbody>{ENDPOINTS.map(([m, p, n]) => <tr key={p}><td>{m}</td><td><code className="ic">{p}</code></td><td>{n}</td></tr>)}</tbody>
      </table>
      <h2>{t("da_f_h")}</h2>
      <table>
        <thead><tr><th>Field</th><th>Description</th></tr></thead>
        <tbody>{FIELDS.map(([f, d]) => <tr key={f}><td><code className="ic">{f}</code></td><td>{d}</td></tr>)}</tbody>
      </table>
      <h2>{t("da_ex_h")}</h2>
      <Code>{CURL}</Code>
      <Code>{PYTHON}</Code>
      <Code>{JS}</Code>
      <p>{t("da_cors")}</p>
      <h2>{t("d_err_h")}</h2>
      <p>{t("d_err_d")}</p>
    </DocsShell>
  );
}
