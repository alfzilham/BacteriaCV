"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";

export const SPLASH_KEY = "bcv-splash";
const SHOW_MS = 1200;

const seen = (): boolean => {
  try { return sessionStorage.getItem(SPLASH_KEY) === "1"; } catch { return false; }
};
const markSeen = () => {
  try { sessionStorage.setItem(SPLASH_KEY, "1"); } catch { /* penyimpanan dapat diblokir */ }
};

/** Layar pembuka sekali per sesi. Hanya dipasang di landing, dilewati bila gerak dikurangi. */
export function Splash() {
  const reduce = useReducedMotion();
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (reduce || seen()) return;
    markSeen();
    setShow(true);
    const id = window.setTimeout(() => setShow(false), SHOW_MS);
    return () => window.clearTimeout(id);
  }, [reduce]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          aria-hidden="true"
          style={{ position: "fixed", inset: 0, zIndex: 100, background: "var(--paper)", display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, y: -12, transition: { duration: 0.15, ease: "easeIn" } }}
        >
          <motion.div
            style={{ fontSize: "clamp(40px,8vw,88px)", fontWeight: 600, letterSpacing: "-.05em", color: "var(--ink)" }}
            initial={{ opacity: 0, y: 12, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          >
            BacteriaCV
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
