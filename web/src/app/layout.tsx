import type { Metadata } from "next";
import { Archivo } from "next/font/google";
import { Shell } from "@/components/shell";
import { Providers } from "./providers";
import "./globals.css";

// One family; its width axis (wide titles, condensed data) carries the hierarchy.
const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  axes: ["wdth"],
});

export const metadata: Metadata = {
  title: "RetailSense",
  description:
    "When should forecasting agents that disagree call in an LLM? Results of a disagreement-triggered forecasting study, and a restocking tool for shops.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${archivo.variable} h-full antialiased`}>
      <body className="min-h-full bg-background text-foreground">
        <Providers>
          <Shell>{children}</Shell>
        </Providers>
      </body>
    </html>
  );
}
