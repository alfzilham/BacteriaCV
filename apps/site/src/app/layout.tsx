import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { PrefsProvider, THEME_BOOT } from "@/controllers/prefs";
import "./globals.css";

export const metadata: Metadata = {
  title: "BacteriaCV",
  description: "Cell shape and Gram status from one microscope image.",
  manifest: "/manifest.webmanifest",
  icons: { icon: [{ url: "/favicon.svg", type: "image/svg+xml" }, { url: "/favicon.ico" }] },
};
export const viewport: Viewport = { themeColor: "#4A3AA8" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet" />
      </head>
      <body>
        <PrefsProvider>{children}</PrefsProvider>
      </body>
    </html>
  );
}
