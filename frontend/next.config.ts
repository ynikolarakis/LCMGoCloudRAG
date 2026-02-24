import createNextIntlPlugin from "next-intl/plugin";
import type { NextConfig } from "next";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  output: "standalone",
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              `connect-src 'self' ${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"} ${process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000"} ${process.env.NEXT_PUBLIC_KEYCLOAK_URL || "http://localhost:8080"}`,
              "style-src 'self' 'unsafe-inline'",
              process.env.NODE_ENV === "development"
                ? "script-src 'self' 'unsafe-eval' 'unsafe-inline'"
                : "script-src 'self' 'unsafe-eval'",
              "img-src 'self' data:",
              "font-src 'self'",
              "frame-src 'self' " + (process.env.NEXT_PUBLIC_KEYCLOAK_URL || "http://localhost:8080"),
            ].join("; "),
          },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
        ],
      },
    ];
  },
};

export default withNextIntl(nextConfig);
