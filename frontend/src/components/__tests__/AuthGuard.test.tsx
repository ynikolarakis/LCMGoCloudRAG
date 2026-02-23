import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// Do NOT import @/test/mocks — we control all mocks directly below.

// Mock next-intl (AuthGuard uses useTranslations)
vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en",
}));

vi.mock("@/components/AuthProvider", () => ({
  useAuth: vi.fn(),
}));

vi.mock("@/lib/auth", () => ({
  hasRole: (user: { roles: string[] } | null, role: string) =>
    user?.roles.includes(role) ?? false,
}));

import { useAuth } from "@/components/AuthProvider";
import { AuthGuard } from "../AuthGuard";

const mockUseAuth = vi.mocked(useAuth);

describe("AuthGuard", () => {
  it("renders children when user has required role", () => {
    mockUseAuth.mockReturnValue({
      user: { id: "1", email: "a@b.com", name: "Admin", roles: ["admin"] },
      token: "tok",
      isAuthenticated: true,
      isLoading: false,
      logout: vi.fn(),
    });

    render(
      <AuthGuard requiredRole="admin">
        <div data-testid="protected">Secret</div>
      </AuthGuard>,
    );
    expect(screen.getByTestId("protected")).toBeInTheDocument();
  });

  it("shows access denied when user lacks role", () => {
    mockUseAuth.mockReturnValue({
      user: { id: "2", email: "u@b.com", name: "User", roles: ["user"] },
      token: "tok",
      isAuthenticated: true,
      isLoading: false,
      logout: vi.fn(),
    });

    render(
      <AuthGuard requiredRole="admin">
        <div>Secret</div>
      </AuthGuard>,
    );
    expect(screen.getByTestId("access-denied")).toBeInTheDocument();
  });
});
