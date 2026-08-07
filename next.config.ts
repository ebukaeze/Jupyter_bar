import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  turbopack: {
    // Pin the workspace root to this project. Without this, Next.js walks
    // up looking for a lockfile and picks up C:\Users\ebuka\package-lock.json,
    // which makes Turbopack treat the entire home directory as the project
    // root and try to watch/cache it — causing runaway CPU/disk usage.
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
