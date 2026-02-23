"use client";

import { useTranslations } from "next-intl";
import { Database, Box, Server, Cpu } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useHealth } from "@/hooks/use-health";
import type { ServiceStatus } from "@/lib/api";

const serviceIcons = {
  database: Database,
  qdrant: Box,
  redis: Server,
  llm: Cpu,
};

function ServiceCard({
  name,
  status,
  label,
}: {
  name: keyof typeof serviceIcons;
  status: ServiceStatus;
  label: string;
}) {
  const Icon = serviceIcons[name];
  const isHealthy = status.status === "healthy";

  return (
    <Card data-testid={`health-card-${name}`}>
      <CardContent className="flex items-center gap-4 p-4">
        <div
          className={cn(
            "flex h-10 w-10 items-center justify-center rounded-full",
            isHealthy ? "bg-green-100" : "bg-red-100",
          )}
        >
          <Icon
            className={cn("h-5 w-5", isHealthy ? "text-green-600" : "text-red-600")}
          />
        </div>
        <div>
          <p className="font-medium text-sm">{label}</p>
          <p className={cn("text-xs", isHealthy ? "text-green-600" : "text-red-600")}>
            {status.status}
          </p>
          {status.detail && (
            <p className="text-xs text-muted-foreground">{status.detail}</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export function HealthPanel() {
  const t = useTranslations("admin");
  const { data, isLoading, isError } = useHealth();

  if (isLoading) {
    return (
      <p className="text-sm text-muted-foreground">{t("healthStatus")}...</p>
    );
  }

  if (isError || !data) {
    return (
      <p className="text-sm text-destructive" data-testid="health-error">
        {t("healthStatus")} — error
      </p>
    );
  }

  return (
    <div data-testid="health-panel">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ServiceCard
          name="database"
          status={data.database}
          label={t("services.database")}
        />
        <ServiceCard
          name="qdrant"
          status={data.qdrant}
          label={t("services.qdrant")}
        />
        <ServiceCard
          name="redis"
          status={data.redis}
          label={t("services.redis")}
        />
        <ServiceCard
          name="llm"
          status={data.llm}
          label={t("services.llm")}
        />
      </div>
    </div>
  );
}
