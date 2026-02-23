import React from "react";
import { vi } from "vitest";

// Mock next-intl
vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en",
  NextIntlClientProvider: ({ children }: { children: React.ReactNode }) => children,
}));

// Mock AuthProvider so components that call useAuth() work in isolation
vi.mock("@/components/AuthProvider", () => ({
  useAuth: () => ({
    user: { id: "test-user", email: "test@test.com", name: "Test User", roles: ["user"] },
    token: "mock-token",
    isAuthenticated: true,
    isLoading: false,
    logout: vi.fn(),
  }),
  AuthProvider: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("@/i18n/routing", () => ({
  Link: ({
    children,
    href,
    ...props
  }: {
    children: React.ReactNode;
    href: string;
    [key: string]: unknown;
  }) => {
    const { locale: _locale, ...rest } = props;
    return <a href={typeof href === "string" ? href : "/"} {...rest}>{children}</a>;
  },
  usePathname: () => "/chat",
  useRouter: () => ({ push: vi.fn() }),
  redirect: vi.fn(),
  routing: { locales: ["en", "el"], defaultLocale: "en" },
}));
