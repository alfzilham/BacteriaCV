"use client";

import { usePrefs } from "@/controllers/prefs";
import { DocsShell } from "./DocsShell";

const ENDPOINTS: [string, string, string][] = [
  ["GET", "/api/health", "status, checkpoint, max_upload_bytes"],
  ["POST", "/api/predict?stage=0", "multipart field file (PNG, JPG, TIFF, max 20 MB). stage 0 to 4."],
  ["GET", "/api/report", "evaluation.json"],
];
const SAMPLE = `curl -X POST "$API/api/predict?stage=0" \\
  -F "file=@sample.png"`;

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
      <pre><code>{SAMPLE}</code></pre>
      <h2>{t("d_err_h")}</h2>
      <p>{t("d_err_d")}</p>
    </DocsShell>
  );
}
