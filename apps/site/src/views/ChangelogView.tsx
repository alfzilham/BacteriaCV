"use client";

import { useState } from "react";
import { usePrefs } from "@/controllers/prefs";
import type { Platform, Release } from "@/models/changelog";
import { DocsShell } from "./DocsShell";

export function ChangelogView({ releases }: { releases: Release[] }) {
  const { t } = usePrefs();
  const [f, setF] = useState<"all" | Platform>("all");
  const shown = releases.filter((r) => f === "all" || r.platform === f);
  const filters: ["all" | Platform, "f_all" | "f_web" | "f_and"][] = [["all", "f_all"], ["web", "f_web"], ["android", "f_and"]];
  return (
    <DocsShell page="changelog">
      <h1>{t("d_cl_h")}</h1>
      <p>{t("d_cl_d")}</p>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }} role="group" aria-label={t("d_cl_h")}>
        {filters.map(([k, label]) => <button key={k} type="button" className="chip" aria-pressed={f === k} onClick={() => setF(k)}>{t(label)}</button>)}
      </div>
      {shown.length === 0 && <p>{t("cl_empty")}</p>}
      {shown.map((r) => (
        <article key={r.tag} className="tile">
          <div className="cap"><code className="ic">{r.tag}</code> · {r.platform === "web" ? t("f_web") : t("f_and")} · {r.date}</div>
          <div style={{ fontSize: 22, letterSpacing: "-.02em", margin: "8px 0" }}>{r.title}</div>
          <div style={{ whiteSpace: "pre-wrap", color: "var(--t2)", fontSize: 15 }}>{r.notes}</div>
        </article>
      ))}
    </DocsShell>
  );
}
