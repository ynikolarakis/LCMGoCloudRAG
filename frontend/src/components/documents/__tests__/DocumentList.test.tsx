import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import "@/test/mocks";

const mockDocuments = {
  items: [
    {
      id: "doc-1",
      filename: "stored.pdf",
      original_filename: "contract.pdf",
      file_size: 1024000,
      content_type: "application/pdf",
      status: "completed" as const,
      language: null,
      page_count: 10,
      chunk_count: 25,
      client_id: "default",
      created_at: "2026-02-23T10:00:00Z",
    },
  ],
  total: 1,
  page: 1,
  page_size: 20,
};

vi.mock("@/hooks/use-documents", () => ({
  useDocuments: () => ({
    data: mockDocuments,
    isLoading: false,
    isError: false,
  }),
  useDeleteDocument: () => ({
    mutate: vi.fn(),
  }),
}));

import { DocumentList } from "../DocumentList";

describe("DocumentList", () => {
  it("renders documents table with data", () => {
    render(<DocumentList />);
    expect(screen.getByTestId("documents-table")).toBeInTheDocument();
    expect(screen.getByText("contract.pdf")).toBeInTheDocument();
  });

  it("renders status badge for each document", () => {
    render(<DocumentList />);
    expect(screen.getByTestId("status-badge-completed")).toBeInTheDocument();
  });

  it("renders delete button for each document", () => {
    render(<DocumentList />);
    expect(screen.getByTestId("delete-btn-doc-1")).toBeInTheDocument();
  });
});
