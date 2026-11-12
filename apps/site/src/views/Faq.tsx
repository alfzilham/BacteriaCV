"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useState } from "react";
import { usePrefs } from "@/controllers/prefs";

const ITEMS = [
  ["qa1_q", "qa1_a"], ["qa2_q", "qa2_a"], ["qa3_q", "qa3_a"],
  ["qa4_q", "qa4_a"], ["qa5_q", "qa5_a"], ["qa6_q", "qa6_a"],
] as const;

/** Akordeon FAQ. Satu jawaban terbuka pada satu waktu, dengan animasi masuk dan keluar. */
export function Faq() {
  const { t } = usePrefs();
  const reduce = useReducedMotion();
  const [open, setOpen] = useState<number | null>(null);
  const enter = reduce ? 0 : 0.25;
  const leave = reduce ? 0 : 0.15;
  return (
    <div className="faq">
      {ITEMS.map(([q, a], i) => {
        const isOpen = open === i;
        return (
          <div key={q} className="faq-i">
            <h3 style={{ margin: 0, fontWeight: 400 }}>
              <button type="button" className="faq-q" id={`faq-q-${i}`} aria-expanded={isOpen} aria-controls={`faq-a-${i}`} onClick={() => setOpen(isOpen ? null : i)}>
                <span>{t(q)}</span>
                <span className="faq-ic" aria-hidden="true" data-open={isOpen}>+</span>
              </button>
            </h3>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div
                  id={`faq-a-${i}`}
                  role="region"
                  aria-labelledby={`faq-q-${i}`}
                  style={{ overflow: "hidden" }}
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1, transition: { duration: enter, ease: [0.2, 0, 0, 1] } }}
                  exit={{ height: 0, opacity: 0, transition: { duration: leave, ease: "easeIn" } }}
                >
                  <p className="faq-a">{t(a)}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}
