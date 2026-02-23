import type { DocumentStatus } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface DocumentStatusBadgeProps {
  status: DocumentStatus;
}

const statusStyles: Record<DocumentStatus, string> = {
  queued: "bg-yellow-100 text-yellow-800 border-yellow-200",
  processing: "bg-blue-100 text-blue-800 border-blue-200 animate-pulse",
  completed: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
};

export function DocumentStatusBadge({ status }: DocumentStatusBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn("capitalize", statusStyles[status])}
      data-testid={`status-badge-${status}`}
    >
      {status}
    </Badge>
  );
}
