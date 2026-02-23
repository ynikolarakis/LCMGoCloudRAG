"use client";

import { useCallback, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Upload } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useUploadDocument } from "@/hooks/use-documents";

const ACCEPTED_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
];
const MAX_SIZE = 50 * 1024 * 1024;

export function UploadForm() {
  const t = useTranslations("documents");
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const upload = useUploadDocument();

  const validateAndUpload = useCallback(
    (file: File) => {
      setError(null);
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Unsupported file type. Accepted: PDF, DOCX, TXT");
        return;
      }
      if (file.size > MAX_SIZE) {
        setError("File too large. Maximum size: 50MB");
        return;
      }
      upload.mutate(file);
    },
    [upload],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) validateAndUpload(file);
    },
    [validateAndUpload],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndUpload(file);
      e.target.value = "";
    },
    [validateAndUpload],
  );

  return (
    <Card data-testid="upload-form">
      <CardHeader>
        <CardTitle>{t("uploadTitle")}</CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className={cn(
            "flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors cursor-pointer",
            isDragging
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-primary/50",
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          data-testid="upload-dropzone"
        >
          <Upload className="h-8 w-8 text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">{t("uploadDragDrop")}</p>
          <p className="text-xs text-muted-foreground mt-1">
            {t("uploadAccepted")}
          </p>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.docx,.txt"
            onChange={handleFileChange}
            data-testid="upload-file-input"
          />
        </div>
        {upload.isPending && (
          <p className="mt-2 text-sm text-muted-foreground">
            {t("processing")}
          </p>
        )}
        {error && (
          <p className="mt-2 text-sm text-destructive" data-testid="upload-error">
            {error}
          </p>
        )}
        {upload.isError && (
          <p className="mt-2 text-sm text-destructive" data-testid="upload-error">
            {upload.error.message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
