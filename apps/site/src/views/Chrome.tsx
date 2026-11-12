"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePrefs } from "@/controllers/prefs";
import { ThemeToggle } from "./ThemeToggle";

export function Controls() {
  const { lang, setLang, t } = usePrefs();
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
      <div className="seg" role="group" aria-label={t("lang")}>
        {(["en", "id"] as const).map((l) => (
          <button key={l} type="button" className="segb" aria-pressed={lang === l} onClick={() => setLang(l)}>{l.toUpperCase()}</button>
        ))}
      </div>
      <ThemeToggle />
    </div>
  );
}

export type Section = "home" | "docs" | "changelog" | "app";

export function Header({ current }: { current: Section }) {
  const { t } = usePrefs();
  const [open, setOpen] = useState(false);
  useEffect(() => { if (!open) return; const f = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false); addEventListener("keydown", f); return () => removeEventListener("keydown", f); }, [open]);
  const cur = (s: Section) => (current === s ? { "aria-current": "page" as const } : {});
  return (
    <header className="hdr">
      <Link href="/" style={{ fontSize: 22, fontWeight: 600, letterSpacing: "-.03em", textDecoration: "none" }}>BacteriaCV</Link>
      <button type="button" className="chip menu-btn" aria-expanded={open} aria-controls="mainnav" onClick={() => setOpen(!open)}>{open ? t("close") : t("menu")}</button>
      <nav id="mainnav" className="nav" data-open={open} aria-label="Main">
        <Link href="/#features" onClick={() => setOpen(false)}>{t("nav_feat")}</Link>
        <Link href="/#how" onClick={() => setOpen(false)}>{t("nav_how")}</Link>
        <Link href="/#contact" onClick={() => setOpen(false)}>{t("nav_ct")}</Link>
        <Link href="/docs" {...cur("docs")}>{t("nav_docs")}</Link>
        <Link href="/docs/changelog" {...cur("changelog")}>{t("nav_cl")}</Link>
      </nav>
      <div className="right" style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <Controls />
        <Link className="pill k" href="/app" {...cur("app")}>{t("launch")}</Link>
      </div>
    </header>
  );
}

export function Footer() {
  const { t } = usePrefs();
  const cols: [Parameters<typeof t>[0], [Parameters<typeof t>[0] | "GitHub", string][]][] = [
    ["ft_c1", [["nav_feat", "/#features"], ["nav_how", "/#how"], ["ft_web", "/app"], ["ft_and", "/#platforms"]]],
    ["ft_c2", [["s_quick", "/docs/quick-start"], ["s_api", "/docs/api"], ["nav_cl", "/docs/changelog"], ["s_limits", "/#limits"]]],
    ["ft_c3", [["GitHub", "https://github.com/alfzilham/BacteriaCV"], ["ft_ds", "https://github.com/alfzilham/BacteriaCV/tree/main/apps/api/data"], ["ft_smp", "/samples/contoh-1-bacilli-positif.png"], ["ft_lic", "https://github.com/alfzilham/BacteriaCV/blob/main/LICENSE"]]],
  ];
  return (
    <footer className="foot">
      <div style={{ display: "flex", flexWrap: "wrap", gap: 48 }}>
        <div style={{ flex: "1 1 300px", maxWidth: 380 }}>
          <div style={{ fontSize: 26, fontWeight: 600, letterSpacing: "-.03em" }}>BacteriaCV</div>
          <p style={{ margin: "14px 0 0", fontSize: 15, lineHeight: 1.6, color: "#B4B9C0" }}>{t("ft_desc")}</p>
        </div>
        <div style={{ flex: "2 1 420px", display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: "16px 24px" }}>
          {cols.map(([head, links]) => (
            <div key={head}>
              <div className="col">{t(head)}</div>
              <nav aria-label={t(head)} style={{ display: "flex", flexDirection: "column" }}>
                {links.map(([label, href]) => (
                  <a key={href} href={href}>{label === "GitHub" ? "GitHub" : t(label)}</a>
                ))}
              </nav>
            </div>
          ))}
        </div>
      </div>
      <div style={{ marginTop: 56, paddingTop: 20, borderTop: "1px solid #23262B", display: "flex", flexWrap: "wrap", gap: "8px 24px", justifyContent: "space-between", fontSize: 13, color: "var(--edge)" }}>
        <span>{t("ft_copy")}</span><span>{t("ft_cred")}</span>
        <span style={{ color: "#B4B9C0", flex: "1 1 100%", maxWidth: 640 }}>{t("disc")} {t("lim_seg")}</span>
      </div>
      <div aria-hidden="true" style={{ containerType: "inline-size", overflow: "hidden", marginTop: 64, userSelect: "none" }}><div className="big">BACTERIACV</div></div>
    </footer>
  );
}
