"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { usePrefs } from "@/controllers/prefs";
import type { CopyKey } from "@/models/copy";
import { Footer, Header } from "./Chrome";

export type DocPage = "quick-start" | "how-it-works" | "self-host" | "api" | "limits" | "faq" | "changelog";

export function DocsShell({ page, children }: { page: DocPage; children: ReactNode }) {
  const { t } = usePrefs();
  const items: [DocPage, CopyKey, string][] = [
    ["quick-start", "s_quick", "/docs/quick-start"], ["how-it-works", "s_how", "/docs/how-it-works"],
    ["self-host", "s_host", "/docs/self-host"], ["api", "s_api", "/docs/api"],
    ["limits", "s_limits", "/docs/limits"], ["faq", "s_faq", "/docs/faq"], ["changelog", "nav_cl", "/docs/changelog"],
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
