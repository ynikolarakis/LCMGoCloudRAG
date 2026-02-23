import { useQuery } from "@tanstack/react-query";
import { fetchAuditLogs, type PaginatedAuditLogsResponse } from "@/lib/api";

export function useAuditLogs(
  page: number = 1,
  pageSize: number = 20,
  action?: string,
) {
  return useQuery<PaginatedAuditLogsResponse>({
    queryKey: ["audit-logs", page, pageSize, action],
    queryFn: () => fetchAuditLogs(page, pageSize, action),
  });
}
