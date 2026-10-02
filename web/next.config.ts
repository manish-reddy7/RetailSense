import type { NextConfig } from "next";

// Static site: every page reads JSON from public/data/ (written by `python -m retailsense.web_export`).
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
