"use client";

import Link from "next/link";
import { usePrefs } from "@/controllers/prefs";

/** Tampil saat API tidak aktif. Mengarahkan pengguna ke cara menjalankan API sendiri. */
export function ApiBanner() {
  const { t } = usePrefs();
  return (
    <div role="status" className="warn" style={{ display: "flex", flexWrap: "wrap", gap: "12px 20px", alignItems: "center", justifyContent: "space-between" }}>
      <div style={{ flex: "1 1 320px" }}><strong style={{ fontWeight: 600 }}>{t("bn_t")}</strong> {t("bn_d")}</div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Link className="pill o" href="/docs/self-host">{t("bn_doc")}</Link>
        <a className="pill o" href="https://github.com/alfzilham/BacteriaCV#readme" target="_blank" rel="noreferrer">{t("bn_gh")}</a>
      </div>
    </div>
  );
}
