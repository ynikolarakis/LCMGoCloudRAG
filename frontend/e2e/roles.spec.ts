import { test, expect } from "@playwright/test";

test.describe("Role-based access", () => {
  test("non-admin user cannot access admin page", async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto("/en/chat");
    await page.waitForSelector("#username", { timeout: 15000 });
    await page.fill("#username", "user@test.com");
    await page.fill("#password", "user123");
    await page.click("#kc-login");
    await page.waitForURL("**/en/chat", { timeout: 15000 });
    await expect(page.getByTestId("nav-admin")).not.toBeVisible();
    await page.goto("/en/admin");
    await expect(page.getByTestId("access-denied")).toBeVisible();
    await context.close();
  });
});
