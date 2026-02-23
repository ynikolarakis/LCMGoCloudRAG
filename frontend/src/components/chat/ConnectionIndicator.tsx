"use client";

import { useTranslations } from "next-intl";
import type { ConnectionStatus } from "@/lib/websocket";
import { cn } from "@/lib/utils";

interface ConnectionIndicatorProps {
  status: ConnectionStatus;
}

const statusColors: Record<ConnectionStatus, string> = {
  connected: "bg-green-500",
  reconnecting: "bg-yellow-500 animate-pulse",
  disconnected: "bg-red-500",
};

export function ConnectionIndicator({ status }: ConnectionIndicatorProps) {
  const t = useTranslations("chat.connectionStatus");

  return (
    <div className="flex items-center gap-2" data-testid="connection-indicator">
      <div className={cn("h-2 w-2 rounded-full", statusColors[status])} />
      <span className="text-xs text-muted-foreground">{t(status)}</span>
    </div>
  );
}
