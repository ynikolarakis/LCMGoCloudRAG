import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import "@/test/mocks";
import { MessageBubble } from "../MessageBubble";

describe("MessageBubble", () => {
  it("renders user message", () => {
    render(<MessageBubble message={{ role: "user", content: "Hello" }} />);
    expect(screen.getByTestId("message-user")).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("renders assistant message", () => {
    render(
      <MessageBubble
        message={{ role: "assistant", content: "The answer is 42." }}
      />,
    );
    expect(screen.getByTestId("message-assistant")).toBeInTheDocument();
    expect(screen.getByText("The answer is 42.")).toBeInTheDocument();
  });

  it("renders citations when provided", () => {
    render(
      <MessageBubble
        message={{
          role: "assistant",
          content: "Answer text",
          citations: [
            { source: "contract.pdf", page: 3, content_preview: "preview" },
          ],
        }}
      />,
    );
    expect(screen.getByTestId("citations-list")).toBeInTheDocument();
    expect(screen.getByTestId("citation-0")).toBeInTheDocument();
    expect(screen.getByText("contract.pdf")).toBeInTheDocument();
  });

  it("shows streaming cursor when isStreaming", () => {
    const { container } = render(
      <MessageBubble
        message={{ role: "assistant", content: "Streaming...", isStreaming: true }}
      />,
    );
    expect(container.querySelector(".animate-pulse")).toBeInTheDocument();
  });
});
