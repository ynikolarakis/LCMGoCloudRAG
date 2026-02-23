const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// --- Types matching backend Pydantic schemas ---

export interface ServiceStatus {
  status: string;
  detail?: string | null;
}

export interface HealthResponse {
  status: string;
  database: ServiceStatus;
  qdrant: ServiceStatus;
  redis: ServiceStatus;
  llm: ServiceStatus;
}

export interface Citation {
  source: string;
  page: number | null;
  content_preview: string;
}

export type DocumentStatus = "queued" | "processing" | "completed" | "failed";

export interface DocumentResponse {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  content_type: string;
  status: DocumentStatus;
  language: string | null;
  page_count: number | null;
  chunk_count: number | null;
  client_id: string;
  created_at: string;
}

export interface PaginatedDocumentsResponse {
  items: DocumentResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface AuditLogEntry {
  id: string;
  user_id: string | null;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  details: Record<string, unknown> | null;
  ip_address: string | null;
  client_id: string;
  created_at: string;
}

export interface PaginatedAuditLogsResponse {
  items: AuditLogEntry[];
  total: number;
  page: number;
  page_size: number;
}

// --- Internal fetch wrapper ---

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      (body as { detail?: string }).detail || `API error: ${res.status}`,
    );
  }

  return res.json() as Promise<T>;
}

// --- API Functions ---

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

export async function fetchDocuments(
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedDocumentsResponse> {
  return apiFetch<PaginatedDocumentsResponse>(
    `/documents?page=${page}&page_size=${pageSize}`,
  );
}

export async function deleteDocument(docId: string): Promise<void> {
  await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
}

export async function uploadDocument(file: File): Promise<{ id: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      (body as { detail?: string }).detail || `Upload failed: ${res.status}`,
    );
  }

  return res.json() as Promise<{ id: string }>;
}

export async function fetchAuditLogs(
  page: number = 1,
  pageSize: number = 20,
  action?: string,
): Promise<PaginatedAuditLogsResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (action) params.set("action", action);

  return apiFetch<PaginatedAuditLogsResponse>(`/admin/audit-logs?${params}`);
}
