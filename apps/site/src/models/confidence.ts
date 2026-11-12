export type Level = "High" | "Medium" | "Low";

/** Ambang dari desain: High >= 0.8, Medium 0.6 sampai 0.79, Low < 0.6. */
export const levelOf = (confidence: number): Level =>
  confidence >= 0.8 ? "High" : confidence >= 0.6 ? "Medium" : "Low";

export const dotsOf = (level: Level): number => (level === "High" ? 3 : level === "Medium" ? 2 : 1);
