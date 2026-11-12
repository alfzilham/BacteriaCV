"use client";

import Link from "next/link";
import { usePrefs } from "@/controllers/prefs";
import { DocsShell } from "./DocsShell";

export function QuickStart() {
  const { t } = usePrefs();
  return (
    <DocsShell page="quick-start">
      <h1>{t("d_quick_h")}</h1>
      <ol style={{ margin: 0, paddingLeft: 20, display: "flex", flexDirection: "column", gap: 12, color: "var(--t2)" }}>
        <li>{t("d_quick_1")}</li><li>{t("d_quick_2")}</li><li>{t("d_quick_3")}</li>
      </ol>
      <h2>{t("d_conf_h")}</h2>
      <ul style={{ margin: 0, paddingLeft: 20, color: "var(--t2)" }}>
        <li>{t("d_conf_hi")}</li><li>{t("d_conf_me")}</li><li>{t("d_conf_lo")}</li>
      </ul>
      <p>{t("qs_more")} <Link href="/docs/self-host">{t("s_host")}</Link></p>
      <div className="warn"><strong style={{ fontWeight: 600 }}>{t("disc")}</strong> {t("lim_seg")}</div>
    </DocsShell>
  );
}
