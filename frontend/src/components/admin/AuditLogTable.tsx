"use client";

import { useState } from "react";
import { useTranslations, useLocale } from "next-intl";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAuditLogs } from "@/hooks/use-audit-logs";
import { formatDate } from "@/lib/utils";

const AUDIT_ACTIONS = [
  "user_login",
  "user_logout",
  "document_upload",
  "document_view",
  "document_delete",
  "query_submitted",
  "response_generated",
  "guardrail_triggered",
  "guardrail_blocked",
  "admin_action",
];

export function AuditLogTable() {
  const t = useTranslations("admin.auditLog");
  const locale = useLocale();
  const [page, setPage] = useState(1);
  const [actionFilter, setActionFilter] = useState<string | undefined>(
    undefined,
  );

  const { data, isLoading, isError } = useAuditLogs(page, 20, actionFilter);

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 1;

  return (
    <div data-testid="audit-log-table">
      <div className="mb-4 flex items-center gap-2">
        <Select
          value={actionFilter ?? "all"}
          onValueChange={(v) => {
            setActionFilter(v === "all" ? undefined : v);
            setPage(1);
          }}
        >
          <SelectTrigger className="w-[200px]" data-testid="audit-action-filter">
            <SelectValue placeholder={t("filterByAction")} />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">{t("allActions")}</SelectItem>
            {AUDIT_ACTIONS.map((action) => (
              <SelectItem key={action} value={action}>
                {action}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {isLoading && (
        <p className="text-sm text-muted-foreground">Loading...</p>
      )}

      {isError && (
        <p className="text-sm text-destructive" data-testid="audit-error">
          Error loading audit logs
        </p>
      )}

      {data && data.items.length === 0 && (
        <p className="text-sm text-muted-foreground" data-testid="audit-empty">
          {t("noEntries")}
        </p>
      )}

      {data && data.items.length > 0 && (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("columns.timestamp")}</TableHead>
                <TableHead>{t("columns.user")}</TableHead>
                <TableHead>{t("columns.action")}</TableHead>
                <TableHead>{t("columns.resourceType")}</TableHead>
                <TableHead>{t("columns.details")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.items.map((entry) => (
                <TableRow key={entry.id} data-testid={`audit-row-${entry.id}`}>
                  <TableCell className="text-sm whitespace-nowrap">
                    {formatDate(entry.created_at, locale)}
                  </TableCell>
                  <TableCell className="text-sm">
                    {entry.user_id?.slice(0, 8) ?? "system"}
                  </TableCell>
                  <TableCell className="text-sm font-mono">
                    {entry.action}
                  </TableCell>
                  <TableCell className="text-sm">
                    {entry.resource_type ?? "—"}
                  </TableCell>
                  <TableCell className="text-sm max-w-[200px] truncate">
                    {entry.details
                      ? JSON.stringify(entry.details).slice(0, 80)
                      : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 mt-4">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 1}
                onClick={() => setPage((p) => p - 1)}
                data-testid="audit-prev-page"
              >
                Previous
              </Button>
              <span className="text-sm text-muted-foreground">
                {page} / {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={page === totalPages}
                onClick={() => setPage((p) => p + 1)}
                data-testid="audit-next-page"
              >
                Next
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
