import { test, expect } from "@playwright/test";
import path from "path";

test.describe("Document upload", () => {
  test("upload a file and see it in the document list", async ({ page }) => {
    await page.goto("/en/documents");
    await expect(page.getByTestId("upload-form")).toBeVisible();
    const filePath = path.join(__dirname, "fixtures", "sample.txt");
    const fileInput = page.getByTestId("file-input");
    await fileInput.setInputFiles(filePath);
    await page.getByTestId("upload-button").click();
    await expect(page.getByText("sample.txt")).toBeVisible({ timeout: 10000 });
    await expect(page.getByTestId("document-status").first()).toBeVisible();
  });
});
