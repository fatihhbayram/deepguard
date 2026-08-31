import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";

// Self-hosted at build time by `next/font`, not linked from Google at runtime. Geist Mono
// carries every figure on this product — ids, logits, window bounds, timestamps — so the
// pairing is load-bearing rather than decorative.
const sans = Geist({ subsets: ["latin"], variable: "--font-geist-sans" });
const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: "InspectRoot",
  description: "Media forensics dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    // `dark` is set here rather than left to `prefers-color-scheme`: this product has one
    // theme, and the risk palette's `dark:` variants must resolve on every machine, not
    // only on the ones set to a dark system theme.
    <html lang="en" className={`dark h-full antialiased ${sans.variable} ${mono.variable}`}>
      <body className="flex min-h-full flex-col">{children}</body>
    </html>
  );
}
