import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import "@/test/mocks";

vi.mock("@/hooks/use-documents", () => ({
  useUploadDocument: () => ({
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    error: null,
  }),
}));

import { UploadForm } from "../UploadForm";

describe("UploadForm", () => {
  it("renders upload form with dropzone", () => {
    render(<UploadForm />);
    expect(screen.getByTestId("upload-form")).toBeInTheDocument();
    expect(screen.getByTestId("upload-dropzone")).toBeInTheDocument();
  });

  it("has hidden file input with correct accept types", () => {
    render(<UploadForm />);
    const input = screen.getByTestId("upload-file-input");
    expect(input).toHaveAttribute("accept", ".pdf,.docx,.txt");
  });
});
