import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "BacteriaCV",
    short_name: "BacteriaCV",
    description: "Cell shape and Gram status from one microscope image.",
    start_url: "/app",
    display: "standalone",
    background_color: "#FFFFFF",
    theme_color: "#4A3AA8",
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
