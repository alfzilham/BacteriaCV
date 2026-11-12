"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { translate, type CopyKey, type Lang } from "@/models/copy";

export type ThemePref = "light" | "dark" | "system";
export const THEME_KEY = "bcv-theme";
export const LANG_KEY = "bcv-lang";

type Ctx = { theme: ThemePref; setTheme: (t: ThemePref) => void; lang: Lang; setLang: (l: Lang) => void; t: (k: CopyKey) => string };
const PrefsContext = createContext<Ctx | null>(null);

const safeGet = (k: string): string | null => {
  try { return localStorage.getItem(k); } catch { return null; }
};
const safeSet = (k: string, v: string) => {
  try { localStorage.setItem(k, v); } catch { /* penyimpanan dapat diblokir */ }
};

export const resolveTheme = (pref: ThemePref, systemDark: boolean): "light" | "dark" =>
  pref === "system" ? (systemDark ? "dark" : "light") : pref;

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemePref>("system");
  const [lang, setLangState] = useState<Lang>("en");

  useEffect(() => {
    const th = safeGet(THEME_KEY);
    if (th === "light" || th === "dark" || th === "system") setThemeState(th);
    const lg = safeGet(LANG_KEY);
    if (lg === "en" || lg === "id") setLangState(lg);
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => { document.documentElement.dataset.theme = resolveTheme(theme, mq.matches); };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [theme]);

  useEffect(() => { document.documentElement.lang = lang; }, [lang]);

  const setTheme = useCallback((t: ThemePref) => { setThemeState(t); safeSet(THEME_KEY, t); }, []);
  const setLang = useCallback((l: Lang) => { setLangState(l); safeSet(LANG_KEY, l); }, []);
  const t = useCallback((k: CopyKey) => translate(k, lang), [lang]);
  const value = useMemo(() => ({ theme, setTheme, lang, setLang, t }), [theme, setTheme, lang, setLang, t]);
  return <PrefsContext.Provider value={value}>{children}</PrefsContext.Provider>;
}

export const usePrefs = (): Ctx => {
  const c = useContext(PrefsContext);
  if (!c) throw new Error("usePrefs harus dipakai di dalam PrefsProvider");
  return c;
};

/** Skrip inline agar tema benar sebelum cat pertama. Harus memakai kunci yang sama. */
export const THEME_BOOT = `try{var p=localStorage.getItem("${THEME_KEY}")||"system";var d=p==="dark"||(p==="system"&&matchMedia("(prefers-color-scheme: dark)").matches);document.documentElement.dataset.theme=d?"dark":"light"}catch(e){}`;
