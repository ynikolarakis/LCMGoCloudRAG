# Phase 5: Production Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make LCM DocIntel deployable and secure with real Keycloak authentication, production Docker containers, GitHub Actions CI/CD, and Playwright E2E tests.

**Architecture:** Auth-first approach — integrate keycloak-js in the frontend, then containerize the auth-integrated app with multi-stage Dockerfiles, automate with GitHub Actions CI/CD (ECR push + ECS deploy), and validate everything with Playwright E2E tests.

**Tech Stack:** keycloak-js, @playwright/test, GitHub Actions, AWS ECR/ECS, Docker multi-stage builds

---

## Dependency Graph

```
Task 1 (Keycloak public client) ──→ Task 2 (auth.ts) ──→ Task 3 (AuthProvider) ──→ Task 4 (API client auth) ──→ Task 5 (Sidebar auth) ──→ Task 6 (AuthGuard + admin) ──→ Task 7 (CORS env var)
                                                                                                                                                                                    │
Task 8 (next.config standalone) ←──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    │
    ├──→ Task 9 (Dockerfile.backend) ──→ Task 11 (docker-compose.prod.yml)
    ├──→ Task 10 (Dockerfile.frontend) ─┘
    │
    └──→ Task 12 (Dockerfile.worker update)
                                                    │
Task 13 (CI workflow) ←─────────────────────────────┘
    │
    └──→ Task 14 (CD workflow)

Task 15 (Playwright setup) ──→ Task 16 (auth setup) ──→ Task 17 (login E2E) ──→ Task 18 (upload E2E) ──→ Task 19 (chat E2E) ──→ Task 20 (admin E2E) ──→ Task 21 (role-based E2E) ──→ Task 22 (final verification)
```

---

## Workstream 1: Frontend Keycloak Authentication

### Task 1: Add public Keycloak client to realm export

**Files:**
- Modify: `docker/keycloak/realm-export.json`

**Context:** The existing `docintel-api` client is confidential (has a secret) — browser apps can't use it. We need a public client that uses PKCE for the Authorization Code flow.

**Step 1: Update the realm export**

Add the `docintel-frontend` public client to the `clients` array in `docker/keycloak/realm-export.json`. The file currently has one client (`docintel-api`). Add the second client after it:

```json
{
  "realm": "docintel",
  "enabled": true,
  "roles": {
    "realm": [
      { "name": "admin", "description": "Full system access" },
      { "name": "manager", "description": "Manage documents and users" },
      { "name": "user", "description": "Upload and query documents" },
      { "name": "viewer", "description": "Read-only access" }
    ]
  },
  "clients": [
    {
      "clientId": "docintel-api",
      "enabled": true,
      "protocol": "openid-connect",
      "publicClient": false,
      "secret": "docintel-api-secret",
      "directAccessGrantsEnabled": true,
      "serviceAccountsEnabled": true,
      "standardFlowEnabled": true,
      "redirectUris": ["http://localhost:3000/*"],
      "webOrigins": ["http://localhost:3000"]
    },
    {
      "clientId": "docintel-frontend",
      "enabled": true,
      "protocol": "openid-connect",
      "publicClient": true,
      "standardFlowEnabled": true,
      "directAccessGrantsEnabled": false,
      "serviceAccountsEnabled": false,
      "redirectUris": ["http://localhost:3000/*"],
      "webOrigins": ["http://localhost:3000"],
      "attributes": {
        "pkce.code.challenge.method": "S256"
      }
    }
  ],
  "users": [
    {
      "username": "admin@test.com",
      "email": "admin@test.com",
      "firstName": "Admin",
      "lastName": "User",
      "enabled": true,
      "credentials": [
        {
          "type": "password",
          "value": "admin123",
          "temporary": false
        }
      ],
      "realmRoles": ["admin"]
    },
    {
      "username": "user@test.com",
      "email": "user@test.com",
      "firstName": "Test",
      "lastName": "User",
      "enabled": true,
      "credentials": [
        {
          "type": "password",
          "value": "user123",
          "temporary": false
        }
      ],
      "realmRoles": ["user"]
    }
  ]
}
```

**Step 2: Commit**

```bash
git add docker/keycloak/realm-export.json
git commit -m "feat: add docintel-frontend public Keycloak client with PKCE"
```

---

### Task 2: Install keycloak-js and create auth singleton

**Files:**
- Modify: `frontend/package.json` (via npm install)
- Create: `frontend/src/lib/auth.ts`

**Step 1: Install keycloak-js**

```bash
cd frontend && npm install keycloak-js
```

**Step 2: Create the auth singleton**

Create `frontend/src/lib/auth.ts`:

```typescript
import Keycloak from "keycloak-js";

const keycloakConfig = {
  url: process.env.NEXT_PUBLIC_KEYCLOAK_URL || "http://localhost:8080",
  realm: process.env.NEXT_PUBLIC_KEYCLOAK_REALM || "docintel",
  clientId: process.env.NEXT_PUBLIC_KEYCLOAK_CLIENT_ID || "docintel-frontend",
};

let keycloakInstance: Keycloak | null = null;

export function getKeycloak(): Keycloak {
  if (!keycloakInstance) {
    keycloakInstance = new Keycloak(keycloakConfig);
  }
  return keycloakInstance;
}

export interface AuthUser {
  id: string;
  email: string;
  name: string;
  roles: string[];
}

export function parseUser(keycloak: Keycloak): AuthUser | null {
  if (!keycloak.authenticated || !keycloak.tokenParsed) return null;

  const token = keycloak.tokenParsed;
  return {
    id: token.sub || "",
    email: (token.email as string) || "",
    name: (token.name as string) || token.preferred_username || "",
    roles: (token.realm_access?.roles as string[]) || [],
  };
}

export function hasRole(user: AuthUser | null, role: string): boolean {
  if (!user) return false;
  return user.roles.includes(role);
}
```

**Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/lib/auth.ts
git commit -m "feat: add keycloak-js adapter and auth helpers"
```

---

### Task 3: Create AuthProvider context and integrate into layout

**Files:**
- Create: `frontend/src/components/AuthProvider.tsx`
- Modify: `frontend/src/app/[locale]/layout.tsx`

**Step 1: Write the AuthProvider test**

Create `frontend/src/components/__tests__/AuthProvider.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

// Mock keycloak-js before importing AuthProvider
const mockKeycloak = {
  init: vi.fn().mockResolvedValue(true),
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
    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>,
    );
    expect(await screen.findByTestId("auth-status")).toHaveTextContent("yes");
    expect(screen.getByTestId("user-email")).toHaveTextContent("test@test.com");
  });

  it("shows loading state before authentication", () => {
    mockKeycloak.init.mockReturnValue(new Promise(() => {})); // never resolves
    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>,
    );
    expect(screen.getByTestId("auth-loading")).toBeInTheDocument();
  });
});
```

**Step 2: Run test to verify it fails**

```bash
cd frontend && npx vitest run src/components/__tests__/AuthProvider.test.tsx
```

Expected: FAIL — `AuthProvider` module not found.

**Step 3: Create AuthProvider**

Create `frontend/src/components/AuthProvider.tsx`:

```tsx
"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import type Keycloak from "keycloak-js";
import { getKeycloak, parseUser, type AuthUser } from "@/lib/auth";

interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
  logout: () => {},
});

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [keycloak, setKeycloak] = useState<Keycloak | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const kc = getKeycloak();

    kc.init({ onLoad: "login-required", pkceMethod: "S256" })
      .then((authenticated) => {
        if (authenticated) {
          setKeycloak(kc);
          setUser(parseUser(kc));
          setToken(kc.token || null);
        }
        setIsLoading(false);
      })
      .catch(() => {
        setIsLoading(false);
      });

    kc.onTokenExpired = () => {
      kc.updateToken(30)
        .then(() => {
          setToken(kc.token || null);
        })
        .catch(() => {
          kc.logout();
        });
    };
  }, []);

  const logout = useCallback(() => {
    keycloak?.logout({ redirectUri: window.location.origin });
  }, [keycloak]);

  if (isLoading) {
    return (
      <div
        className="flex min-h-screen items-center justify-center"
        data-testid="auth-loading"
      >
        <p className="text-muted-foreground">Authenticating...</p>
      </div>
    );
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isLoading,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
```

**Step 4: Update the locale layout to wrap with AuthProvider**

Modify `frontend/src/app/[locale]/layout.tsx`. The current layout wraps children with `NextIntlClientProvider > Providers > Sidebar > main`. Insert `AuthProvider` inside `Providers` (it needs to be a client component, and it wraps the rest of the UI):

```tsx
import { NextIntlClientProvider } from "next-intl";
import { getMessages } from "next-intl/server";
import { notFound } from "next/navigation";
import { routing } from "@/i18n/routing";
import { Providers } from "@/components/Providers";
import { AuthProvider } from "@/components/AuthProvider";
import { Sidebar } from "@/components/Sidebar";
import "../globals.css";

interface LocaleLayoutProps {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

export default async function LocaleLayout({ children, params }: LocaleLayoutProps) {
  const { locale } = await params;

  if (!routing.locales.includes(locale as "en" | "el")) {
    notFound();
  }

  const messages = await getMessages();

  return (
    <html lang={locale}>
      <body className="min-h-screen bg-background font-sans antialiased">
        <NextIntlClientProvider messages={messages}>
          <Providers>
            <AuthProvider>
              <Sidebar />
              <main className="md:ml-64 min-h-screen" data-testid="main-content">
                {children}
              </main>
            </AuthProvider>
          </Providers>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
```

**Step 5: Run test to verify it passes**

```bash
cd frontend && npx vitest run src/components/__tests__/AuthProvider.test.tsx
```

Expected: PASS

**Step 6: Run all existing tests to verify no regressions**

```bash
cd frontend && npx vitest run
```

Expected: All 24+ tests pass.

**Step 7: Commit**

```bash
git add frontend/src/components/AuthProvider.tsx frontend/src/components/__tests__/AuthProvider.test.tsx frontend/src/app/\[locale\]/layout.tsx
git commit -m "feat: add AuthProvider with keycloak-js and integrate into layout"
```

---

### Task 4: Inject auth token into API client and WebSocket

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/websocket.ts`
- Modify: `frontend/src/components/chat/ChatInterface.tsx`

**Context:** The API client currently makes unauthenticated requests. We need to inject the Bearer token from the auth context. The approach: export a `setAuthToken` function that the AuthProvider calls whenever the token changes. The `apiFetch` wrapper reads it from a module-level variable. For WebSocket, pass the token as a query parameter since WebSocket doesn't support custom headers.

**Step 1: Update api.ts to support auth token injection**

Replace the full content of `frontend/src/lib/api.ts`. The key change is adding `setAuthToken()` and `getAuthToken()` functions, and using the token in `apiFetch` and `uploadDocument` and `deleteDocument`:

```typescript
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// --- Auth token management ---

let _authToken: string | null = null;

export function setAuthToken(token: string | null): void {
  _authToken = token;
}

export function getAuthToken(): string | null {
  return _authToken;
}

// --- Types matching backend Pydantic schemas ---

export interface ServiceStatus {
  status: string;
  detail?: string | null;
}

export interface HealthResponse {
  status: string;
  database: ServiceStatus;
  qdrant: ServiceStatus;
  redis: ServiceStatus;
  llm: ServiceStatus;
}

export interface Citation {
  source: string;
  page: number | null;
  content_preview: string;
}

export type DocumentStatus = "queued" | "processing" | "completed" | "failed";

export interface DocumentResponse {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  content_type: string;
  status: DocumentStatus;
  language: string | null;
  page_count: number | null;
  chunk_count: number | null;
  client_id: string;
  created_at: string;
}

export interface PaginatedDocumentsResponse {
  items: DocumentResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface AuditLogEntry {
  id: string;
  user_id: string | null;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  details: Record<string, unknown> | null;
  ip_address: string | null;
  client_id: string;
  created_at: string;
}

export interface PaginatedAuditLogsResponse {
  items: AuditLogEntry[];
  total: number;
  page: number;
  page_size: number;
}

// --- Internal fetch wrapper ---

function authHeaders(): Record<string, string> {
  if (_authToken) {
    return { Authorization: `Bearer ${_authToken}` };
  }
  return {};
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      (body as { detail?: string }).detail || `API error: ${res.status}`,
    );
  }

  return res.json() as Promise<T>;
}

// --- API Functions ---

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

export async function fetchDocuments(
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedDocumentsResponse> {
  return apiFetch<PaginatedDocumentsResponse>(
    `/documents?page=${page}&page_size=${pageSize}`,
  );
}

export async function deleteDocument(docId: string): Promise<void> {
  await fetch(`${API_BASE}/documents/${docId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
}

export async function uploadDocument(file: File): Promise<{ id: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      (body as { detail?: string }).detail || `Upload failed: ${res.status}`,
    );
  }

  return res.json() as Promise<{ id: string }>;
}

export async function fetchAuditLogs(
  page: number = 1,
  pageSize: number = 20,
  action?: string,
): Promise<PaginatedAuditLogsResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (action) params.set("action", action);

  return apiFetch<PaginatedAuditLogsResponse>(`/admin/audit-logs?${params}`);
}
```

**Step 2: Update WebSocket to pass token as query parameter**

Modify `frontend/src/lib/websocket.ts`. Change the `connect` method to accept an optional `token` parameter and append it to the WebSocket URL:

In the `connect` method (line 30), change the signature and pass token:

```typescript
  connect(callbacks: WebSocketCallbacks, token?: string | null): void {
    this.callbacks = callbacks;
    this.retryCount = 0;
    this._token = token || null;
    this._connect();
  }
```

Add a private `_token` field to the class (after line 28):

```typescript
  private _token: string | null = null;
```

In the `_connect` method (around line 40), build the URL with token:

```typescript
  private _connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      const url = this._token ? `${WS_BASE}?token=${this._token}` : WS_BASE;
      this.ws = new WebSocket(url);
    } catch {
      this._scheduleReconnect();
      return;
    }
```

**Step 3: Update AuthProvider to sync token to api.ts**

In `frontend/src/components/AuthProvider.tsx`, add a `useEffect` that calls `setAuthToken` whenever the token changes. Add this import and effect:

```typescript
import { setAuthToken } from "@/lib/api";
```

Add after the existing `useEffect` for keycloak init (around line 65):

```typescript
  useEffect(() => {
    setAuthToken(token);
  }, [token]);
```

**Step 4: Update ChatInterface to pass token to WebSocket**

In `frontend/src/components/chat/ChatInterface.tsx`, import `useAuth` and pass the token to `ws.connect()`:

Add import:
```typescript
import { useAuth } from "@/components/AuthProvider";
```

Inside the component, get the token:
```typescript
  const { token } = useAuth();
```

Update the useEffect dependency and connect call (line 30):
```typescript
  useEffect(() => {
    const ws = new WebSocketManager();
    wsRef.current = ws;

    ws.connect({
      onToken: (content: string) => { /* ... same ... */ },
      // ... same callbacks ...
    }, token);

    return () => {
      ws.disconnect();
    };
  }, [token]);
```

**Step 5: Run all tests**

```bash
cd frontend && npx vitest run
```

Expected: All tests pass. Existing tests use mocks that don't depend on auth.

**Step 6: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/lib/websocket.ts frontend/src/components/AuthProvider.tsx frontend/src/components/chat/ChatInterface.tsx
git commit -m "feat: inject auth token into API client and WebSocket"
```

---

### Task 5: Add logout button and role-based nav to Sidebar

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`
- Modify: `frontend/messages/en.json`
- Modify: `frontend/messages/el.json`

**Step 1: Add i18n keys for auth**

Add to `frontend/messages/en.json` at the top level (after the `common` section):

```json
  "auth": {
    "logout": "Logout",
    "loading": "Authenticating..."
  },
```

Add to `frontend/messages/el.json`:

```json
  "auth": {
    "logout": "Αποσύνδεση",
    "loading": "Πιστοποίηση..."
  },
```

**Step 2: Update Sidebar to hide admin for non-admins and add logout**

Replace the full `frontend/src/components/Sidebar.tsx`:

```tsx
"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { FileText, MessageSquare, Settings, Menu, X, Globe, LogOut } from "lucide-react";
import { Link, usePathname } from "@/i18n/routing";
import { useLocale } from "next-intl";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/components/AuthProvider";
import { hasRole } from "@/lib/auth";

const navItems = [
  { href: "/chat" as const, icon: MessageSquare, labelKey: "nav.chat" as const, requiredRole: null },
  { href: "/documents" as const, icon: FileText, labelKey: "nav.documents" as const, requiredRole: null },
  { href: "/admin" as const, icon: Settings, labelKey: "nav.admin" as const, requiredRole: "admin" as const },
];

export function Sidebar() {
  const t = useTranslations();
  const pathname = usePathname();
  const locale = useLocale();
  const { user, logout } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const otherLocale = locale === "en" ? "el" : "en";

  const visibleItems = navItems.filter(
    (item) => !item.requiredRole || hasRole(user, item.requiredRole),
  );

  return (
    <>
      {/* Mobile hamburger */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 md:hidden"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
        data-testid="sidebar-toggle"
      >
        {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r bg-background transition-transform md:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
        data-testid="sidebar"
      >
        <div className="flex h-16 items-center px-6">
          <h1 className="text-lg font-semibold" data-testid="app-title">
            {t("common.appName")}
          </h1>
        </div>

        <Separator />

        <nav className="flex-1 space-y-1 px-3 py-4" data-testid="sidebar-nav">
          {visibleItems.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )}
                onClick={() => setMobileOpen(false)}
                data-testid={`nav-${item.href.slice(1)}`}
              >
                <item.icon className="h-4 w-4" />
                {t(item.labelKey)}
              </Link>
            );
          })}
        </nav>

        <Separator />

        {/* User info + locale + logout */}
        <div className="space-y-1 p-3">
          {user && (
            <div className="px-3 py-1 text-xs text-muted-foreground truncate" data-testid="user-email">
              {user.email}
            </div>
          )}
          <Link
            href={pathname || "/chat"}
            locale={otherLocale}
            className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            data-testid="locale-switcher"
          >
            <Globe className="h-4 w-4" />
            {otherLocale === "el" ? "Ελληνικά" : "English"}
          </Link>
          <button
            onClick={logout}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-destructive hover:text-destructive-foreground"
            data-testid="logout-button"
          >
            <LogOut className="h-4 w-4" />
            {t("auth.logout")}
          </button>
        </div>
      </aside>
    </>
  );
}
```

**Step 3: Run all tests**

```bash
cd frontend && npx vitest run
```

Expected: All tests pass.

**Step 4: Commit**

```bash
git add frontend/src/components/Sidebar.tsx frontend/messages/en.json frontend/messages/el.json
git commit -m "feat: add logout button and role-based admin nav visibility"
```

---

### Task 6: Create AuthGuard and protect admin route

**Files:**
- Create: `frontend/src/components/AuthGuard.tsx`
- Modify: `frontend/src/app/[locale]/admin/page.tsx`

**Step 1: Write AuthGuard test**

Create `frontend/src/components/__tests__/AuthGuard.test.tsx`:

```typescript
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

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
```

**Step 2: Run test to verify it fails**

```bash
cd frontend && npx vitest run src/components/__tests__/AuthGuard.test.tsx
```

Expected: FAIL — `AuthGuard` module not found.

**Step 3: Create AuthGuard**

Create `frontend/src/components/AuthGuard.tsx`:

```tsx
"use client";

import { useTranslations } from "next-intl";
import { ShieldAlert } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { hasRole } from "@/lib/auth";

interface AuthGuardProps {
  children: React.ReactNode;
  requiredRole: string;
}

export function AuthGuard({ children, requiredRole }: AuthGuardProps) {
  const { user } = useAuth();
  const t = useTranslations("common");

  if (!hasRole(user, requiredRole)) {
    return (
      <div
        className="flex min-h-[50vh] flex-col items-center justify-center gap-4"
        data-testid="access-denied"
      >
        <ShieldAlert className="h-12 w-12 text-destructive" />
        <h2 className="text-xl font-semibold">{t("accessDenied")}</h2>
        <p className="text-muted-foreground">{t("accessDeniedDescription")}</p>
      </div>
    );
  }

  return <>{children}</>;
}
```

**Step 4: Add i18n keys for access denied**

In `frontend/messages/en.json`, add to the `common` section:

```json
    "accessDenied": "Access Denied",
    "accessDeniedDescription": "You do not have permission to view this page."
```

In `frontend/messages/el.json`, add to the `common` section:

```json
    "accessDenied": "Πρόσβαση Απορρίφθηκε",
    "accessDeniedDescription": "Δεν έχετε δικαίωμα να δείτε αυτή τη σελίδα."
```

**Step 5: Protect admin route**

Replace `frontend/src/app/[locale]/admin/page.tsx`:

```tsx
import { AdminPage } from "@/components/admin/AdminPage";
import { AuthGuard } from "@/components/AuthGuard";

export default function AdminPageRoute() {
  return (
    <AuthGuard requiredRole="admin">
      <AdminPage />
    </AuthGuard>
  );
}
```

**Step 6: Run tests**

```bash
cd frontend && npx vitest run
```

Expected: All tests pass.

**Step 7: Commit**

```bash
git add frontend/src/components/AuthGuard.tsx frontend/src/components/__tests__/AuthGuard.test.tsx frontend/src/app/\[locale\]/admin/page.tsx frontend/messages/en.json frontend/messages/el.json
git commit -m "feat: add AuthGuard component and protect admin route"
```

---

### Task 7: Make CORS origins configurable via env var

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/main.py`

**Step 1: Add CORS_ORIGINS to settings**

In `backend/app/config.py`, add after the `CLIENT_ID` line (line 21):

```python
    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"
```

**Step 2: Update main.py to use settings**

In `backend/app/main.py`, change the CORS middleware (lines 52-58) from:

```python
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

To:

```python
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

**Step 3: Run backend tests to verify no regressions**

```bash
cd backend && PYTHONPATH=. python -m pytest ../tests/ -x -v
```

Expected: All 71 tests pass.

**Step 4: Commit**

```bash
git add backend/app/config.py backend/app/main.py
git commit -m "feat: make CORS origins configurable via CORS_ORIGINS env var"
```

---

## Workstream 2: Production Docker Builds

### Task 8: Add standalone output to Next.js config

**Files:**
- Modify: `frontend/next.config.ts`

**Step 1: Update next.config.ts**

Replace `frontend/next.config.ts`:

```typescript
import createNextIntlPlugin from "next-intl/plugin";
import type { NextConfig } from "next";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  output: "standalone",
};

export default withNextIntl(nextConfig);
```

**Step 2: Verify build still works**

```bash
cd frontend && npm run build
```

Expected: Build succeeds. Output should mention standalone mode.

**Step 3: Commit**

```bash
git add frontend/next.config.ts
git commit -m "feat: enable Next.js standalone output for Docker builds"
```

---

### Task 9: Create production Dockerfile for backend

**Files:**
- Create: `docker/Dockerfile.backend`

**Step 1: Create the multi-stage Dockerfile**

Create `docker/Dockerfile.backend`:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 appuser && useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Step 2: Verify it builds**

```bash
docker build -f docker/Dockerfile.backend -t docintel-backend:test backend/
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docker/Dockerfile.backend
git commit -m "feat: add multi-stage production Dockerfile for backend"
```

---

### Task 10: Create production Dockerfile for frontend

**Files:**
- Create: `docker/Dockerfile.frontend`

**Step 1: Create the multi-stage Dockerfile**

Create `docker/Dockerfile.frontend`:

```dockerfile
# Stage 1: Install dependencies
FROM node:20-alpine AS deps

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Stage 2: Build
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# Stage 3: Runtime
FROM node:20-alpine

RUN addgroup --system --gid 1001 appuser && adduser --system --uid 1001 appuser

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=builder /app/public ./public
COPY --from=builder --chown=appuser:appuser /app/.next/standalone ./
COPY --from=builder --chown=appuser:appuser /app/.next/static ./.next/static

USER appuser

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
```

**Step 2: Verify it builds**

```bash
docker build -f docker/Dockerfile.frontend -t docintel-frontend:test frontend/
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docker/Dockerfile.frontend
git commit -m "feat: add multi-stage production Dockerfile for frontend"
```

---

### Task 11: Create production docker-compose

**Files:**
- Create: `docker/docker-compose.prod.yml`

**Step 1: Create the production compose file**

Create `docker/docker-compose.prod.yml`:

```yaml
services:
  backend:
    image: ${ECR_REGISTRY:-docintel}/backend:${TAG:-latest}
    build:
      context: ../backend
      dockerfile: ../docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-ragadmin}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-docintel}
      REDIS_URL: redis://redis:6379/0
      QDRANT_URL: http://qdrant:6333
      LLM_BASE_URL: ${LLM_BASE_URL}
      EMBEDDING_BASE_URL: ${EMBEDDING_BASE_URL}
      ENVIRONMENT: production
      CORS_ORIGINS: ${CORS_ORIGINS:-http://localhost:3000}
      KEYCLOAK_URL: http://keycloak:8080
      KEYCLOAK_REALM: docintel
      KEYCLOAK_CLIENT_ID: docintel-api
      KEYCLOAK_CLIENT_SECRET: ${KEYCLOAK_CLIENT_SECRET}
      SECRET_KEY: ${SECRET_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    restart: unless-stopped

  frontend:
    image: ${ECR_REGISTRY:-docintel}/frontend:${TAG:-latest}
    build:
      context: ../frontend
      dockerfile: ../docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-http://localhost:8000/api/v1}
      NEXT_PUBLIC_WS_URL: ${NEXT_PUBLIC_WS_URL:-ws://localhost:8000/api/v1/ws}
      NEXT_PUBLIC_KEYCLOAK_URL: ${NEXT_PUBLIC_KEYCLOAK_URL:-http://localhost:8080}
      NEXT_PUBLIC_KEYCLOAK_REALM: docintel
      NEXT_PUBLIC_KEYCLOAK_CLIENT_ID: docintel-frontend
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped

  celery-worker:
    image: ${ECR_REGISTRY:-docintel}/worker:${TAG:-latest}
    build:
      context: ../backend
      dockerfile: ../docker/Dockerfile.worker
    command: celery -A app.celery_app worker --loglevel=info --concurrency=2
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-ragadmin}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-docintel}
      REDIS_URL: redis://redis:6379/0
      QDRANT_URL: http://qdrant:6333
      LLM_BASE_URL: ${LLM_BASE_URL}
      EMBEDDING_BASE_URL: ${EMBEDDING_BASE_URL}
      ENVIRONMENT: production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-docintel}
      POSTGRES_USER: ${POSTGRES_USER:-ragadmin}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-ragadmin} -d ${POSTGRES_DB:-docintel}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  keycloak:
    image: quay.io/keycloak/keycloak:latest
    ports:
      - "8080:8080"
    environment:
      KC_BOOTSTRAP_ADMIN_USERNAME: ${KEYCLOAK_ADMIN:-admin}
      KC_BOOTSTRAP_ADMIN_PASSWORD: ${KEYCLOAK_ADMIN_PASSWORD}
      KC_DB: postgres
      KC_DB_URL: jdbc:postgresql://postgres:5432/${POSTGRES_DB:-docintel}
      KC_DB_USERNAME: ${POSTGRES_USER:-ragadmin}
      KC_DB_PASSWORD: ${POSTGRES_PASSWORD}
      KC_HOSTNAME: ${KEYCLOAK_HOSTNAME:-localhost}
    command: start --optimized
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

volumes:
  qdrant_data:
  postgres_data:
  redis_data:
```

**Step 2: Commit**

```bash
git add docker/docker-compose.prod.yml
git commit -m "feat: add production docker-compose with all services"
```

---

### Task 12: Update Dockerfile.worker to multi-stage

**Files:**
- Modify: `docker/Dockerfile.worker`

**Step 1: Replace with multi-stage build**

Replace `docker/Dockerfile.worker`:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

RUN groupadd --gid 1001 appuser && useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

RUN mkdir -p /app/uploads && chown -R appuser:appuser /app
USER appuser

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
```

**Step 2: Verify docker-compose.dev.yml still works with the updated Dockerfile**

```bash
docker compose -f docker/docker-compose.dev.yml build celery-worker
```

Expected: Build succeeds.

**Step 3: Commit**

```bash
git add docker/Dockerfile.worker
git commit -m "feat: upgrade Celery worker Dockerfile to multi-stage with non-root user"
```

---

## Workstream 3: CI/CD Pipeline

### Task 13: Create CI workflow (GitHub Actions)

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: ["*"]
  pull_request:
    branches: [master]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  backend-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check backend/
      - run: ruff format --check backend/

  backend-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: docintel_test
          POSTGRES_USER: ragadmin
          POSTGRES_PASSWORD: testpassword
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U ragadmin -d docintel_test"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r backend/requirements.txt
      - run: python -m pytest tests/ -x -v --tb=short
        env:
          PYTHONPATH: backend
          DATABASE_URL: postgresql+asyncpg://ragadmin:testpassword@localhost:5432/docintel_test
          REDIS_URL: redis://localhost:6379/0
          QDRANT_URL: http://localhost:6333
          ENVIRONMENT: dev

  frontend-lint:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npx next lint
      - run: npx tsc --noEmit

  frontend-test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npx vitest run

  frontend-build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run build

  secrets-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "feat: add GitHub Actions CI workflow with lint, test, and build jobs"
```

---

### Task 14: Create CD workflow (GitHub Actions)

**Files:**
- Create: `.github/workflows/cd.yml`

**Step 1: Create the CD workflow**

Create `.github/workflows/cd.yml`:

```yaml
name: CD

on:
  push:
    branches: [master]

concurrency:
  group: cd-${{ github.ref }}
  cancel-in-progress: false

permissions:
  id-token: write
  contents: read

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2

      - name: Set image tags
        run: |
          echo "TAG=${{ github.sha }}" >> $GITHUB_ENV
          echo "ECR_REGISTRY=${{ steps.ecr-login.outputs.registry }}" >> $GITHUB_ENV

      - name: Build and push backend image
        run: |
          docker build -f docker/Dockerfile.backend -t $ECR_REGISTRY/docintel-backend:$TAG -t $ECR_REGISTRY/docintel-backend:latest backend/
          docker push $ECR_REGISTRY/docintel-backend:$TAG
          docker push $ECR_REGISTRY/docintel-backend:latest

      - name: Build and push frontend image
        run: |
          docker build -f docker/Dockerfile.frontend -t $ECR_REGISTRY/docintel-frontend:$TAG -t $ECR_REGISTRY/docintel-frontend:latest frontend/
          docker push $ECR_REGISTRY/docintel-frontend:$TAG
          docker push $ECR_REGISTRY/docintel-frontend:latest

      - name: Build and push worker image
        run: |
          docker build -f docker/Dockerfile.worker -t $ECR_REGISTRY/docintel-worker:$TAG -t $ECR_REGISTRY/docintel-worker:latest backend/
          docker push $ECR_REGISTRY/docintel-worker:$TAG
          docker push $ECR_REGISTRY/docintel-worker:latest

      - name: Deploy backend to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ secrets.ECS_BACKEND_TASK_DEF }}
          service: ${{ secrets.ECS_BACKEND_SERVICE }}
          cluster: ${{ secrets.ECS_CLUSTER }}
          wait-for-service-stability: true

      - name: Deploy frontend to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ secrets.ECS_FRONTEND_TASK_DEF }}
          service: ${{ secrets.ECS_FRONTEND_SERVICE }}
          cluster: ${{ secrets.ECS_CLUSTER }}
          wait-for-service-stability: true

      - name: Deploy worker to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ secrets.ECS_WORKER_TASK_DEF }}
          service: ${{ secrets.ECS_WORKER_SERVICE }}
          cluster: ${{ secrets.ECS_CLUSTER }}
          wait-for-service-stability: true
```

**Step 2: Commit**

```bash
git add .github/workflows/cd.yml
git commit -m "feat: add GitHub Actions CD workflow for ECR push and ECS deploy"
```

---

## Workstream 4: Playwright E2E Tests

### Task 15: Install Playwright and create config

**Files:**
- Modify: `frontend/package.json` (via npm install)
- Create: `frontend/playwright.config.ts`

**Step 1: Install Playwright**

```bash
cd frontend && npm install --save-dev @playwright/test && npx playwright install chromium
```

**Step 2: Create Playwright config**

Create `frontend/playwright.config.ts`:

```typescript
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? "github" : "html",
  timeout: 30000,
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "setup", testMatch: /auth\.setup\.ts/ },
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        storageState: "e2e/.auth/user.json",
      },
      dependencies: ["setup"],
    },
  ],
});
```

**Step 3: Add E2E test script to package.json**

Add to the `scripts` section of `frontend/package.json`:

```json
    "test:e2e": "playwright test"
```

**Step 4: Create the auth storage directory placeholder**

```bash
mkdir -p frontend/e2e/.auth
echo '{}' > frontend/e2e/.auth/user.json
```

Add `frontend/e2e/.auth/` to `.gitignore` (auth state should not be committed). If a `.gitignore` exists in `frontend/`, add `e2e/.auth/` to it. If not, create one:

```bash
echo "e2e/.auth/" >> frontend/.gitignore
```

**Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/playwright.config.ts frontend/.gitignore
git commit -m "feat: install Playwright and add E2E config"
```

---

### Task 16: Create Playwright auth setup and test fixture

**Files:**
- Create: `frontend/e2e/auth.setup.ts`
- Create: `frontend/e2e/fixtures/sample.pdf`

**Step 1: Create the auth setup**

Create `frontend/e2e/auth.setup.ts`:

```typescript
import { test as setup, expect } from "@playwright/test";

const ADMIN_USER = {
  username: process.env.E2E_ADMIN_USER || "admin@test.com",
  password: process.env.E2E_ADMIN_PASSWORD || "admin123",
};

setup("authenticate as admin", async ({ page }) => {
  // Navigate to the app — Keycloak should redirect to login
  await page.goto("/en/chat");

  // Wait for Keycloak login form
  await page.waitForSelector("#username", { timeout: 15000 });

  // Fill in credentials
  await page.fill("#username", ADMIN_USER.username);
  await page.fill("#password", ADMIN_USER.password);
  await page.click("#kc-login");

  // Wait for redirect back to app
  await page.waitForURL("**/en/chat", { timeout: 15000 });

  // Verify we're authenticated — sidebar should be visible
  await expect(page.getByTestId("sidebar")).toBeVisible();

  // Save auth state
  await page.context().storageState({ path: "e2e/.auth/user.json" });
});
```

**Step 2: Create a minimal test PDF fixture**

We need a small PDF for upload testing. Create it programmatically:

```bash
cd frontend && mkdir -p e2e/fixtures
```

Create a simple text file that will serve as a test document. For E2E upload testing, a small text file works since the backend accepts TXT:

Create `frontend/e2e/fixtures/sample.txt` with content:

```
LCM DocIntel Test Document

This is a sample document used for end-to-end testing.
It contains text about contract terms and conditions.

Section 1: Agreement Terms
The agreement shall remain in force for a period of twenty-four (24) months
from the date of execution by both parties.

Section 2: Payment Terms
Payment terms are net 30 days from invoice date.
```

**Step 3: Commit**

```bash
git add frontend/e2e/auth.setup.ts frontend/e2e/fixtures/sample.txt
git commit -m "feat: add Playwright auth setup and test fixture"
```

---

### Task 17: E2E test — Login flow

**Files:**
- Create: `frontend/e2e/login.spec.ts`

**Step 1: Create login test**

Create `frontend/e2e/login.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

test.describe("Login flow", () => {
  test("authenticated user lands on chat page with sidebar visible", async ({ page }) => {
    await page.goto("/en/chat");

    // Should be on the chat page (auth setup already logged in)
    await expect(page).toHaveURL(/\/en\/chat/);

    // Sidebar should be visible
    await expect(page.getByTestId("sidebar")).toBeVisible();

    // App title should be present
    await expect(page.getByTestId("app-title")).toHaveText("LCM DocIntel");

    // Nav items should be visible
    await expect(page.getByTestId("nav-chat")).toBeVisible();
    await expect(page.getByTestId("nav-documents")).toBeVisible();

    // User email should be visible (admin@test.com)
    await expect(page.getByTestId("user-email")).toBeVisible();

    // Logout button should be visible
    await expect(page.getByTestId("logout-button")).toBeVisible();
  });

  test("admin user can see admin nav link", async ({ page }) => {
    await page.goto("/en/chat");
    await expect(page.getByTestId("nav-admin")).toBeVisible();
  });
});
```

**Step 2: Commit**

```bash
git add frontend/e2e/login.spec.ts
git commit -m "test: add E2E login flow test"
```

---

### Task 18: E2E test — Upload document

**Files:**
- Create: `frontend/e2e/upload.spec.ts`

**Step 1: Create upload test**

Create `frontend/e2e/upload.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";
import path from "path";

test.describe("Document upload", () => {
  test("upload a file and see it in the document list", async ({ page }) => {
    await page.goto("/en/documents");

    // Upload form should be visible
    await expect(page.getByTestId("upload-form")).toBeVisible();

    // Upload the test fixture
    const filePath = path.join(__dirname, "fixtures", "sample.txt");
    const fileInput = page.getByTestId("file-input");
    await fileInput.setInputFiles(filePath);

    // Click upload button
    await page.getByTestId("upload-button").click();

    // Wait for the document to appear in the list
    await expect(page.getByText("sample.txt")).toBeVisible({ timeout: 10000 });

    // Status badge should be visible (queued or processing)
    await expect(
      page.getByTestId("document-status").first(),
    ).toBeVisible();
  });
});
```

**Step 2: Commit**

```bash
git add frontend/e2e/upload.spec.ts
git commit -m "test: add E2E document upload test"
```

---

### Task 19: E2E test — Chat query with streaming

**Files:**
- Create: `frontend/e2e/chat.spec.ts`

**Step 1: Create chat test**

Create `frontend/e2e/chat.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

test.describe("Chat query", () => {
  test("send a question and receive a streamed response", async ({ page }) => {
    await page.goto("/en/chat");

    // Chat interface should be visible
    await expect(page.getByTestId("chat-interface")).toBeVisible();

    // Connection indicator should eventually show connected
    await expect(page.getByTestId("connection-status")).toHaveAttribute(
      "data-status",
      "connected",
      { timeout: 10000 },
    );

    // Type a question
    const input = page.getByTestId("chat-input");
    await input.fill("What are the payment terms?");

    // Send the question
    await page.getByTestId("chat-send-button").click();

    // User message should appear
    await expect(page.getByText("What are the payment terms?")).toBeVisible();

    // Wait for assistant response to start streaming (content appears)
    await expect(
      page.locator("[data-testid='message-assistant']").first(),
    ).toBeVisible({ timeout: 30000 });

    // Wait for streaming to complete (send button re-enabled)
    await expect(page.getByTestId("chat-send-button")).toBeEnabled({
      timeout: 60000,
    });
  });
});
```

**Step 2: Commit**

```bash
git add frontend/e2e/chat.spec.ts
git commit -m "test: add E2E chat streaming test"
```

---

### Task 20: E2E test — Admin dashboard

**Files:**
- Create: `frontend/e2e/admin.spec.ts`

**Step 1: Create admin test**

Create `frontend/e2e/admin.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

test.describe("Admin dashboard", () => {
  test("health panel shows service status cards", async ({ page }) => {
    await page.goto("/en/admin");

    // Admin page should load
    await expect(page.getByTestId("admin-page")).toBeVisible();

    // Health tab should be active by default
    await expect(page.getByTestId("health-panel")).toBeVisible();

    // All 4 service cards should be visible
    await expect(page.getByTestId("health-database")).toBeVisible();
    await expect(page.getByTestId("health-qdrant")).toBeVisible();
    await expect(page.getByTestId("health-redis")).toBeVisible();
    await expect(page.getByTestId("health-llm")).toBeVisible();
  });

  test("audit log table loads entries", async ({ page }) => {
    await page.goto("/en/admin");

    // Switch to audit log tab
    await page.getByTestId("audit-tab").click();

    // Audit log table should be visible
    await expect(page.getByTestId("audit-log-table")).toBeVisible();
  });
});
```

**Step 2: Commit**

```bash
git add frontend/e2e/admin.spec.ts
git commit -m "test: add E2E admin dashboard test"
```

---

### Task 21: E2E test — Role-based access

**Files:**
- Create: `frontend/e2e/roles.spec.ts`

**Context:** This test needs to login as a non-admin user. We create a separate auth setup for the regular user.

**Step 1: Create role-based access test**

Create `frontend/e2e/roles.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

test.describe("Role-based access", () => {
  // This test uses a fresh browser context (no stored auth) to login as regular user
  test("non-admin user cannot access admin page", async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();

    // Navigate — will redirect to Keycloak
    await page.goto("/en/chat");

    // Login as regular user
    await page.waitForSelector("#username", { timeout: 15000 });
    await page.fill("#username", "user@test.com");
    await page.fill("#password", "user123");
    await page.click("#kc-login");

    // Wait for redirect
    await page.waitForURL("**/en/chat", { timeout: 15000 });

    // Admin nav should NOT be visible
    await expect(page.getByTestId("nav-admin")).not.toBeVisible();

    // Navigate directly to admin — should show access denied
    await page.goto("/en/admin");
    await expect(page.getByTestId("access-denied")).toBeVisible();

    await context.close();
  });
});
```

**Step 2: Commit**

```bash
git add frontend/e2e/roles.spec.ts
git commit -m "test: add E2E role-based access test"
```

---

### Task 22: Final verification pass

**Step 1: Run backend lint**

```bash
cd backend && ruff check . && ruff format --check .
```

Expected: Clean.

**Step 2: Run backend tests**

```bash
cd backend && PYTHONPATH=. python -m pytest ../tests/ -x -v
```

Expected: All 71+ tests pass.

**Step 3: Run frontend unit tests**

```bash
cd frontend && npx vitest run
```

Expected: All 26+ tests pass (24 existing + 2 new AuthProvider + AuthGuard tests).

**Step 4: Run TypeScript check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: Clean.

**Step 5: Run frontend build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with standalone output.

**Step 6: Verify Docker builds**

```bash
docker build -f docker/Dockerfile.backend -t docintel-backend:verify backend/
docker build -f docker/Dockerfile.frontend -t docintel-frontend:verify frontend/
docker build -f docker/Dockerfile.worker -t docintel-worker:verify backend/
```

Expected: All three build successfully.

**Step 7: Commit any final fixes if needed, then tag**

```bash
git add -A && git status
```

If clean, Phase 5 is complete.
