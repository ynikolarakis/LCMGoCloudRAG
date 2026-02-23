import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import "@/test/mocks";

vi.mock("@/hooks/use-health", () => ({
  useHealth: () => ({
    data: {
      status: "healthy",
      database: { status: "healthy", detail: null },
      qdrant: { status: "healthy", detail: null },
      redis: { status: "healthy", detail: null },
      llm: { status: "unhealthy", detail: "Connection refused" },
    },
    isLoading: false,
    isError: false,
  }),
}));

import { HealthPanel } from "../HealthPanel";

describe("HealthPanel", () => {
  it("renders all 4 service health cards", () => {
    render(<HealthPanel />);
    expect(screen.getByTestId("health-panel")).toBeInTheDocument();
    expect(screen.getByTestId("health-card-database")).toBeInTheDocument();
    expect(screen.getByTestId("health-card-qdrant")).toBeInTheDocument();
    expect(screen.getByTestId("health-card-redis")).toBeInTheDocument();
    expect(screen.getByTestId("health-card-llm")).toBeInTheDocument();
  });

  it("shows unhealthy status for LLM", () => {
    render(<HealthPanel />);
    expect(screen.getByText("Connection refused")).toBeInTheDocument();
  });
});
