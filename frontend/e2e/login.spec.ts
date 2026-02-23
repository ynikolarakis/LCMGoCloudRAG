import { test, expect } from "@playwright/test";

test.describe("Login flow", () => {
  test("authenticated user lands on chat page with sidebar visible", async ({ page }) => {
    await page.goto("/en/chat");
    await expect(page).toHaveURL(/\/en\/chat/);
    await expect(page.getByTestId("sidebar")).toBeVisible();
    await expect(page.getByTestId("app-title")).toHaveText("LCM DocIntel");
    await expect(page.getByTestId("nav-chat")).toBeVisible();
    await expect(page.getByTestId("nav-documents")).toBeVisible();
    await expect(page.getByTestId("user-email")).toBeVisible();
    await expect(page.getByTestId("logout-button")).toBeVisible();
  });

  test("admin user can see admin nav link", async ({ page }) => {
    await page.goto("/en/chat");
    await expect(page.getByTestId("nav-admin")).toBeVisible();
  });
});
