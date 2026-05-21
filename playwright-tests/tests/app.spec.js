const { test, expect } = require('@playwright/test');

test('home page loads at expected URL', async ({ page }) => {
  const response = await page.goto('/');
  expect(response).not.toBeNull();
  await expect(page).toHaveURL(/erickwendel\.github\.io\/?/i);
});

test('main heading is visible', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
});
