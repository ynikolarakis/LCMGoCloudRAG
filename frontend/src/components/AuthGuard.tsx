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
