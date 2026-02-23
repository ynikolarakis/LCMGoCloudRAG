import { test as setup, expect } from "@playwright/test";

const ADMIN_USER = {
  username: process.env.E2E_ADMIN_USER || "admin@test.com",
  password: process.env.E2E_ADMIN_PASSWORD || "admin123",
};

setup("authenticate as admin", async ({ page }) => {
  await page.goto("/en/chat");
  await page.waitForSelector("#username", { timeout: 15000 });
  await page.fill("#username", ADMIN_USER.username);
  await page.fill("#password", ADMIN_USER.password);
  await page.click("#kc-login");
  await page.waitForURL("**/en/chat", { timeout: 15000 });
  await expect(page.getByTestId("sidebar")).toBeVisible();
  await page.context().storageState({ path: "e2e/.auth/user.json" });
});
