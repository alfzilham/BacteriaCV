"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { usePrefs } from "@/controllers/prefs";
import { Footer, Header } from "./Chrome";

export type DocPage = "quick-start" | "api" | "changelog";

export function DocsShell({ page, children }: { page: DocPage; children: ReactNode }) {
  const { t } = usePrefs();
  const items: [DocPage, "s_quick" | "s_api" | "nav_cl", string][] = [
    ["quick-start", "s_quick", "/docs/quick-start"], ["api", "s_api", "/docs/api"], ["changelog", "nav_cl", "/docs/changelog"],
  ];
  return (
    <div className="wrap">
      <Header current={page === "changelog" ? "changelog" : "docs"} />
      <div className="doc">
        <nav className="side" aria-label={t("d_title")}>
          {items.map(([k, label, href]) => (
            <Link key={k} href={href} {...(page === k ? { "aria-current": "page" as const } : {})}>{t(label)}</Link>
          ))}
        </nav>
        <main className="body">{children}</main>
      </div>
      <Footer />
    </div>
  );
}
