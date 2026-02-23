import { test, expect } from "@playwright/test";

test.describe("Admin dashboard", () => {
  test("health panel shows service status cards", async ({ page }) => {
    await page.goto("/en/admin");
    await expect(page.getByTestId("admin-page")).toBeVisible();
    await expect(page.getByTestId("health-panel")).toBeVisible();
    await expect(page.getByTestId("health-database")).toBeVisible();
    await expect(page.getByTestId("health-qdrant")).toBeVisible();
    await expect(page.getByTestId("health-redis")).toBeVisible();
    await expect(page.getByTestId("health-llm")).toBeVisible();
  });

  test("audit log table loads entries", async ({ page }) => {
    await page.goto("/en/admin");
    await page.getByTestId("audit-tab").click();
    await expect(page.getByTestId("audit-log-table")).toBeVisible();
  });
});
