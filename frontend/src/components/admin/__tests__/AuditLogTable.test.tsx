import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import "@/test/mocks";

vi.mock("@/hooks/use-audit-logs", () => ({
  useAuditLogs: () => ({
    data: {
      items: [
        {
          id: "log-1",
          user_id: "00000000-0000-0000-0000-000000000001",
          action: "query_submitted",
          resource_type: "query",
          resource_id: "q-1",
          details: { query_text: "What is the contract term?" },
          ip_address: "127.0.0.1",
          client_id: "default",
          created_at: "2026-02-23T12:00:00Z",
        },
      ],
      total: 1,
      page: 1,
      page_size: 20,
    },
    isLoading: false,
    isError: false,
  }),
}));

import { AuditLogTable } from "../AuditLogTable";

describe("AuditLogTable", () => {
  it("renders audit log table with entries", () => {
    render(<AuditLogTable />);
    expect(screen.getByTestId("audit-log-table")).toBeInTheDocument();
    expect(screen.getByTestId("audit-row-log-1")).toBeInTheDocument();
  });

  it("displays action type in table row", () => {
    render(<AuditLogTable />);
    expect(screen.getByText("query_submitted")).toBeInTheDocument();
  });

  it("renders action filter dropdown", () => {
    render(<AuditLogTable />);
    expect(screen.getByTestId("audit-action-filter")).toBeInTheDocument();
  });
});
