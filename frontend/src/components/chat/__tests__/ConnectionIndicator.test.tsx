import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import "@/test/mocks";
import { ConnectionIndicator } from "../ConnectionIndicator";

describe("ConnectionIndicator", () => {
  it("renders connected status", () => {
    render(<ConnectionIndicator status="connected" />);
    expect(screen.getByTestId("connection-indicator")).toBeInTheDocument();
    expect(screen.getByText("connected")).toBeInTheDocument();
  });

  it("renders disconnected status", () => {
    render(<ConnectionIndicator status="disconnected" />);
    expect(screen.getByText("disconnected")).toBeInTheDocument();
  });

  it("renders reconnecting status", () => {
    render(<ConnectionIndicator status="reconnecting" />);
    expect(screen.getByText("reconnecting")).toBeInTheDocument();
  });
});
