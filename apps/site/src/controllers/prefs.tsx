"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { translate, type CopyKey, type Lang } from "@/models/copy";

export type ThemePref = "light" | "dark" | "system";
/** Kunci penyimpanan tema. Dipakai ThemeProvider (next-themes), nilainya tidak boleh berubah. */
export const THEME_KEY = "bcv-theme";
export const LANG_KEY = "bcv-lang";

type Ctx = { lang: Lang; setLang: (l: Lang) => void; t: (k: CopyKey) => string };
const PrefsContext = createContext<Ctx | null>(null);

const safeGet = (k: string): string | null => {
  try { return localStorage.getItem(k); } catch { return null; }
};
const safeSet = (k: string, v: string) => {
  try { localStorage.setItem(k, v); } catch { /* penyimpanan dapat diblokir */ }
};

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>("en");

  useEffect(() => {
    const lg = safeGet(LANG_KEY);
    if (lg === "en" || lg === "id") setLangState(lg);
  }, []);

  useEffect(() => { document.documentElement.lang = lang; }, [lang]);

  const setLang = useCallback((l: Lang) => { setLangState(l); safeSet(LANG_KEY, l); }, []);
  const t = useCallback((k: CopyKey) => translate(k, lang), [lang]);
  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);
  return <PrefsContext.Provider value={value}>{children}</PrefsContext.Provider>;
}

export const usePrefs = (): Ctx => {
  const c = useContext(PrefsContext);
  if (!c) throw new Error("usePrefs harus dipakai di dalam PrefsProvider");
  return c;
};
