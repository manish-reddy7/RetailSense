import type { NextConfig } from "next";

// Static site: research pages read JSON from public/data/ (written by `python -m retailsense.web_export`);
// retailer pages (/retailer) call the API of `uvicorn retailsense.service.app:app`, which also serves out/.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true, // /retailer/ -> retailer/index.html, served as is by the API server
  images: { unoptimized: true },
};

export default nextConfig;
