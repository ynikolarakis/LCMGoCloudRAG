import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteDocument,
  fetchDocuments,
  uploadDocument,
  type PaginatedDocumentsResponse,
} from "@/lib/api";

export function useDocuments(page: number = 1, pageSize: number = 20) {
  return useQuery<PaginatedDocumentsResponse>({
    queryKey: ["documents", page, pageSize],
    queryFn: () => fetchDocuments(page, pageSize),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      const hasActiveJobs = data.items.some(
        (d) => d.status === "queued" || d.status === "processing",
      );
      return hasActiveJobs ? 5000 : false;
    },
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => uploadDocument(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (docId: string) => deleteDocument(docId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}
