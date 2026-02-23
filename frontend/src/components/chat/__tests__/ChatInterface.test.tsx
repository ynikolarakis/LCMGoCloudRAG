import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import "@/test/mocks";

// Mock WebSocketManager before importing ChatInterface.
// The factory must return a class (function constructor) — not an arrow function —
// so that `new WebSocketManager()` works correctly in the component.
vi.mock("@/lib/websocket", () => {
  const mockConnect = vi.fn();
  const mockSend = vi.fn();
  const mockDisconnect = vi.fn();

  function MockWebSocketManager(this: unknown) {
    (this as Record<string, unknown>).connect = mockConnect;
    (this as Record<string, unknown>).send = mockSend;
    (this as Record<string, unknown>).disconnect = mockDisconnect;
    (this as Record<string, unknown>).getStatus = () => "connected";
  }

  return {
    WebSocketManager: MockWebSocketManager,
  };
});

import { ChatInterface } from "../ChatInterface";

describe("ChatInterface", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders chat interface with input and send button", () => {
    render(<ChatInterface />);
    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
    expect(screen.getByTestId("chat-send-button")).toBeInTheDocument();
  });

  it("sends message when clicking send button", async () => {
    const user = userEvent.setup();
    render(<ChatInterface />);

    const input = screen.getByTestId("chat-input");
    await user.type(input, "What is the contract term?");
    await user.click(screen.getByTestId("chat-send-button"));

    expect(screen.getByTestId("message-user")).toBeInTheDocument();
  });

  it("disables send button when input is empty", () => {
    render(<ChatInterface />);
    expect(screen.getByTestId("chat-send-button")).toBeDisabled();
  });
});
