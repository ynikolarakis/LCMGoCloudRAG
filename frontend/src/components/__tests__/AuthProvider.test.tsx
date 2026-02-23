import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";

// Do NOT import @/test/mocks here — this file tests AuthProvider itself,
// so we mock its dependencies directly below.

const mockKeycloak = {
  init: vi.fn(),
  authenticated: true,
  token: "mock-token",
  tokenParsed: {
    sub: "user-123",
    email: "test@test.com",
    name: "Test User",
    realm_access: { roles: ["user"] },
  },
  onTokenExpired: null as (() => void) | null,
  updateToken: vi.fn().mockResolvedValue(true),
  logout: vi.fn(),
};

vi.mock("@/lib/auth", () => ({
  getKeycloak: () => mockKeycloak,
  parseUser: () => ({
    id: "user-123",
    email: "test@test.com",
    name: "Test User",
    roles: ["user"],
  }),
  hasRole: (user: { roles: string[] } | null, role: string) =>
    user?.roles.includes(role) ?? false,
}));

vi.mock("@/lib/api", () => ({
  setAuthToken: vi.fn(),
  getAuthToken: vi.fn(),
}));

// Mock next-intl (AuthProvider doesn't use it, but keycloak-js import may pull in deps)
vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en",
}));

import { AuthProvider, useAuth } from "../AuthProvider";

function TestConsumer() {
  const { user, isAuthenticated } = useAuth();
  return (
    <div>
      <span data-testid="auth-status">{isAuthenticated ? "yes" : "no"}</span>
      <span data-testid="user-email">{user?.email}</span>
    </div>
  );
}

describe("AuthProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders children after authentication", async () => {
    mockKeycloak.init.mockResolvedValue(true);
    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>,
    );
    expect(await screen.findByTestId("auth-status")).toHaveTextContent("yes");
    expect(screen.getByTestId("user-email")).toHaveTextContent("test@test.com");
  });

  it("shows loading state before authentication", async () => {
    // init returns a promise that never resolves — component stays in loading state
    let resolveInit!: (value: boolean) => void;
    const pendingPromise = new Promise<boolean>((resolve) => {
      resolveInit = resolve;
    });
    mockKeycloak.init.mockReturnValue(pendingPromise);

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>,
    );

    // Before init resolves, the loading screen must be visible
    expect(screen.getByTestId("auth-loading")).toBeInTheDocument();

    // Cleanup: resolve the promise so React can finish updating without warnings
    await act(async () => {
      resolveInit(false);
      await pendingPromise;
    });
  });
});
