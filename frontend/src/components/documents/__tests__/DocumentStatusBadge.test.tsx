import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DocumentStatusBadge } from "../DocumentStatusBadge";

describe("DocumentStatusBadge", () => {
  it("renders queued badge", () => {
    render(<DocumentStatusBadge status="queued" />);
    expect(screen.getByTestId("status-badge-queued")).toBeInTheDocument();
    expect(screen.getByText("queued")).toBeInTheDocument();
  });

  it("renders completed badge", () => {
    render(<DocumentStatusBadge status="completed" />);
    expect(screen.getByTestId("status-badge-completed")).toBeInTheDocument();
  });

  it("renders failed badge", () => {
    render(<DocumentStatusBadge status="failed" />);
    expect(screen.getByTestId("status-badge-failed")).toBeInTheDocument();
  });

  it("renders processing badge with pulse animation", () => {
    const { container } = render(<DocumentStatusBadge status="processing" />);
    expect(container.querySelector(".animate-pulse")).toBeInTheDocument();
  });
});
