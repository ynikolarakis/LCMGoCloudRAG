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
import { setAuthToken } from "@/lib/api";

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
    setAuthToken(token);
  }, [token]);

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
