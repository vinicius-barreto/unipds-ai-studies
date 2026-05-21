const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  timeout: 5000,
  expect: {
    timeout: 5000,
  },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'https://erickwendel.github.io/vanilla-js-web-app-example/',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
