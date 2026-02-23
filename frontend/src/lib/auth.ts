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
