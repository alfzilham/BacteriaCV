import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import type { ReactNode } from "react";
import { PrefsProvider } from "@/controllers/prefs";
import { ThemeProvider } from "@/controllers/ThemeProvider";
import { SmoothScroll } from "@/views/SmoothScroll";
import "./globals.css";

export const metadata: Metadata = {
  title: "BacteriaCV",
  description: "Cell shape and Gram status from one microscope image.",
  manifest: "/manifest.webmanifest",
  icons: { icon: [{ url: "/favicon.svg", type: "image/svg+xml" }, { url: "/favicon.ico" }] },
};
const hanken = localFont({
  src: "./fonts/hanken-grotesk-latin-wght-normal.woff2",
  weight: "100 900",
  style: "normal",
  display: "swap",
  variable: "--font-hanken",
});

export const viewport: Viewport = { themeColor: "#4A3AA8" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={hanken.variable} suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <PrefsProvider>
            <SmoothScroll />
            {children}
          </PrefsProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
