"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ReactNode } from "react";
import { THEME_KEY } from "./prefs";

/**
 * Tema memakai next-themes. Atribut data-theme dipakai oleh tokens.css, kelas dark
 * dipakai oleh efek wipe Animate UI. Nilai tersimpan tetap light, dark, atau system.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  return (
    <NextThemesProvider attribute={["class", "data-theme"]} defaultTheme="system" enableSystem storageKey={THEME_KEY}>
      {children}
    </NextThemesProvider>
  );
}
