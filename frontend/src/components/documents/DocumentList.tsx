"use client";

import { useState } from "react";
import { useTranslations, useLocale } from "next-intl";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useDocuments, useDeleteDocument } from "@/hooks/use-documents";
import { DocumentStatusBadge } from "./DocumentStatusBadge";
import { DeleteConfirmDialog } from "./DeleteConfirmDialog";
import { formatDate, formatFileSize } from "@/lib/utils";

export function DocumentList() {
  const t = useTranslations("documents");
  const locale = useLocale();
  const [page, setPage] = useState(1);
  const [deleteTarget, setDeleteTarget] = useState<{
    id: string;
    filename: string;
  } | null>(null);

  const { data, isLoading, isError } = useDocuments(page);
  const deleteMutation = useDeleteDocument();

  const handleDelete = () => {
    if (deleteTarget) {
      deleteMutation.mutate(deleteTarget.id);
      setDeleteTarget(null);
    }
  };

  if (isLoading) {
    return (
      <p
        className="text-sm text-muted-foreground"
        data-testid="documents-loading"
      >
        {t("processing")}
      </p>
    );
  }

  if (isError) {
    return (
      <p className="text-sm text-destructive" data-testid="documents-error">
        {t("title")} — error loading
      </p>
    );
  }

  if (!data || data.items.length === 0) {
    return (
      <p
        className="text-sm text-muted-foreground"
        data-testid="documents-empty"
      >
        {t("noDocuments")}
      </p>
    );
  }

  const totalPages = Math.ceil(data.total / data.page_size);

  return (
    <>
      <Table data-testid="documents-table">
        <TableHeader>
          <TableRow>
            <TableHead>{t("columns.filename")}</TableHead>
            <TableHead>{t("columns.status")}</TableHead>
            <TableHead>{t("columns.pages")}</TableHead>
            <TableHead>{t("columns.uploaded")}</TableHead>
            <TableHead>{t("columns.actions")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.items.map((doc) => (
            <TableRow key={doc.id} data-testid={`document-row-${doc.id}`}>
              <TableCell>
                <div>
                  <p className="font-medium text-sm">{doc.original_filename}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(doc.file_size)}
                  </p>
                </div>
              </TableCell>
              <TableCell>
                <DocumentStatusBadge status={doc.status} />
              </TableCell>
              <TableCell className="text-sm">
                {doc.page_count ?? "—"}
              </TableCell>
              <TableCell className="text-sm">
                {formatDate(doc.created_at, locale)}
              </TableCell>
              <TableCell>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() =>
                    setDeleteTarget({
                      id: doc.id,
                      filename: doc.original_filename,
                    })
                  }
                  aria-label={`Delete ${doc.original_filename}`}
                  data-testid={`delete-btn-${doc.id}`}
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
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
            data-testid="prev-page"
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
            data-testid="next-page"
          >
            Next
          </Button>
        </div>
      )}

      <DeleteConfirmDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
        onConfirm={handleDelete}
        filename={deleteTarget?.filename ?? ""}
      />
    </>
  );
}
