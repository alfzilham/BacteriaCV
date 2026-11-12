"use client";

import { ThemeToggler } from "@animate-ui/react/primitives/effects/theme-toggler";
import { useTheme } from "next-themes";
import { useEffect, useState, type ComponentProps, type ReactNode } from "react";
import { usePrefs } from "@/controllers/prefs";

type TogglerProps = ComponentProps<typeof ThemeToggler>;

const Icon = ({ d }: { d: ReactNode }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--ink)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{d}</svg>
);
const SUN = <Icon d={<><circle cx="12" cy="12" r="4" /><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4" /></>} />;
const MOON = <Icon d={<path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z" />} />;
const SCREEN = <Icon d={<><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>} />;

/** Toggle terang atau gelap dengan efek wipe, ditambah tombol Sistem. Kunci tersimpan: bcv-theme. */
export function ThemeToggle() {
  const { theme, resolvedTheme, setTheme } = useTheme();
  const { t } = usePrefs();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  if (!mounted) return <div className="seg" aria-hidden="true" style={{ width: 98, height: 44 }} />;

  return (
    <div className="seg" role="group" aria-label={t("th")}>
      <ThemeToggler
        theme={theme as TogglerProps["theme"]}
        resolvedTheme={resolvedTheme as TogglerProps["resolvedTheme"]}
        setTheme={setTheme as TogglerProps["setTheme"]}
      >
        {({ effective, toggleTheme }) => (
          <button
            type="button"
            className="segb"
            style={{ width: 44, padding: 0 }}
            aria-label={t(effective === "dark" ? "th_l" : "th_d")}
            aria-pressed={theme !== "system"}
            onClick={() => toggleTheme(effective === "dark" ? "light" : "dark")}
          >
            {effective === "dark" ? MOON : SUN}
          </button>
        )}
      </ThemeToggler>
      <button
        type="button"
        className="segb"
        style={{ width: 44, padding: 0 }}
        aria-label={t("th_s")}
        aria-pressed={theme === "system"}
        onClick={() => setTheme("system")}
      >
        {SCREEN}
      </button>
    </div>
  );
}
