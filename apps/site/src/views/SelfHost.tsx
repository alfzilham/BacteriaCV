"use client";

import { usePrefs } from "@/controllers/prefs";
import { Code, DocList, Ic } from "./DocBits";
import { DocsShell } from "./DocsShell";

const API_CMD = `cd apps/api
python -m venv .venv
# Windows (PowerShell): .venv\\Scripts\\Activate.ps1
# macOS or Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
# Windows (PowerShell): $env:CORS_ORIGINS = "http://localhost:3000"
# macOS or Linux: export CORS_ORIGINS=http://localhost:3000
uvicorn app.main:app --port 8000`;

const WEB_CMD = `cd apps/site
npm install
npm run dev`;

const ENV_ROWS: [string, string, "sh_e1" | "sh_e2" | "sh_e3" | "sh_e4" | "sh_e5" | "sh_e6" | "sh_e7" | "sh_e8"][] = [
  ["API", "CORS_ORIGINS", "sh_e1"],
  ["API", "PORT", "sh_e2"],
  ["API", "RATE_LIMIT_PER_MINUTE", "sh_e7"],
  ["API", "RATE_LIMIT_TRUST_FORWARDED", "sh_e8"],
  ["Site", "NEXT_PUBLIC_API_URL", "sh_e3"],
  ["Site", "GITHUB_REPO", "sh_e4"],
  ["Site", "GITHUB_WEBHOOK_SECRET", "sh_e5"],
  ["Site", "GITHUB_TOKEN", "sh_e6"],
];

export function SelfHost() {
  const { t } = usePrefs();
  return (
    <DocsShell page="self-host">
      <h1>{t("sh_h")}</h1>
      <p>{t("sh_d")}</p>
      <h2>{t("sh_pre_h")}</h2>
      <DocList items={[t("sh_pre1"), t("sh_pre2"), t("sh_pre3")]} />
      <h2>{t("sh_api_h")}</h2>
      <Code>{API_CMD}</Code>
      <p>{t("sh_api_d")}</p>
      <p>{t("sh_cors_d")}</p>
      <h2>{t("sh_web_h")}</h2>
      <Code>{WEB_CMD}</Code>
      <p>{t("sh_web_d")}</p>
      <h2>{t("sh_dep_h")}</h2>
      <h3 style={{ margin: 0, fontSize: 20, fontWeight: 500 }}>{t("sh_v_h")}</h3>
      <DocList ordered items={[t("sh_v1"), t("sh_v2"), t("sh_v3"), t("sh_v4")]} />
      <h3 style={{ margin: 0, fontSize: 20, fontWeight: 500 }}>{t("sh_r_h")}</h3>
      <DocList ordered items={[t("sh_r1"), t("sh_r2"), t("sh_r3")]} />
      <h3 style={{ margin: 0, fontSize: 20, fontWeight: 500 }}>{t("sh_w_h")}</h3>
      <p>{t("sh_w_d")}</p>
      <h2>{t("sh_e_h")}</h2>
      <table>
        <thead><tr><th>{t("th_app")}</th><th>{t("th_var")}</th><th>{t("th_pur")}</th></tr></thead>
        <tbody>{ENV_ROWS.map(([app, name, desc]) => <tr key={name}><td>{app}</td><td><Ic>{name}</Ic></td><td>{t(desc)}</td></tr>)}</tbody>
      </table>
      <h2>{t("sh_demo_h")}</h2>
      <p>{t("sh_demo_d")}</p>
    </DocsShell>
  );
}
