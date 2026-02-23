import { test, expect } from "@playwright/test";

test.describe("Chat query", () => {
  test("send a question and receive a streamed response", async ({ page }) => {
    await page.goto("/en/chat");
    await expect(page.getByTestId("chat-interface")).toBeVisible();
    await expect(page.getByTestId("connection-status")).toHaveAttribute(
      "data-status",
      "connected",
      { timeout: 10000 },
    );
    const input = page.getByTestId("chat-input");
    await input.fill("What are the payment terms?");
    await page.getByTestId("chat-send-button").click();
    await expect(page.getByText("What are the payment terms?")).toBeVisible();
    await expect(
      page.locator("[data-testid='message-assistant']").first(),
    ).toBeVisible({ timeout: 30000 });
    await expect(page.getByTestId("chat-send-button")).toBeEnabled({
      timeout: 60000,
    });
  });
});
