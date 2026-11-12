"use client";

import { motion, useReducedMotion } from "motion/react";
import type { CSSProperties, ReactNode } from "react";

type Props = { children: ReactNode; delay?: number; className?: string; style?: CSSProperties };

/** Muncul saat masuk layar, sekali saja. Dilewati bila pengguna meminta gerak berkurang. */
export function Reveal({ children, delay = 0, className, style }: Props) {
  const reduce = useReducedMotion();
  if (reduce) return <div className={className} style={style}>{children}</div>;
  return (
    <motion.div
      className={className}
      style={style}
      initial={{ opacity: 0, y: 12, filter: "blur(4px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.4, delay, ease: [0.2, 0, 0, 1] }}
    >
      {children}
    </motion.div>
  );
}
