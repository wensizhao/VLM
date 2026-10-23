import { chromium } from 'playwright-core';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const base = process.env.VLM_WEB_URL || 'http://127.0.0.1:8000';
const output = process.env.VLM_UI_OUTPUT_DIR || '../results/verification/web-ui';
await mkdir(output, { recursive: true });
const errors = [];
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome-stable',
  headless: true,
  args: ['--no-sandbox', '--disable-gpu'],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
page.on('console', (message) => {
  if (message.type() === 'error') errors.push('console: ' + message.text());
});
page.on('pageerror', (error) => errors.push('pageerror: ' + error.message));

async function goto(hash) {
  await page.goto(base + '/#' + hash, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1500);
}

await goto('overview');
await page.waitForSelector('#overview-stats .stat', { timeout: 20000 });
await page.screenshot({ path: join(output, 'studio_overview.png') });

await goto('datasets');
await page.waitForSelector('.dataset-card', { timeout: 20000 });
await page.click('.dataset-card');
await page.waitForSelector('#dataset-detail:not(.hidden)', { timeout: 20000 });
await page.waitForSelector('.video-card', { timeout: 20000 });
await page.screenshot({ path: join(output, 'studio_datasets.png'), fullPage: true });
await page.click('.video-card');
await page.waitForSelector('#preview-img', { timeout: 20000 });
await page.waitForFunction(() => {
  const image = document.querySelector('#preview-img');
  return image && image.complete && image.naturalWidth > 0;
}, { timeout: 20000 });
const preview = await page.evaluate(() => {
  const image = document.querySelector('#preview-img');
  return { width: image.naturalWidth, height: image.naturalHeight };
});
await page.screenshot({ path: join(output, 'studio_preview.png'), fullPage: true });
await page.click('[data-action="close-preview"]');

await goto('train');
await page.waitForFunction(() => document.querySelectorAll('#train-dataset option').length > 1, { timeout: 20000 });
const train = await page.evaluate(() => ({
  datasets: document.querySelectorAll('#train-dataset option').length,
  runName: document.querySelector('#train-run-name').value,
  submitDisabled: document.querySelector('#train-submit').disabled,
}));
await page.screenshot({ path: join(output, 'studio_train.png') });

await goto('validate');
await page.waitForSelector('#validate-content:not(.hidden)', { timeout: 20000 });
await page.waitForSelector('#validate-table table', { timeout: 20000 });
const validate = await page.evaluate(() => ({
  metrics: document.querySelectorAll('#validate-metrics .stat').length,
  rows: document.querySelectorAll('#validate-table tbody tr').length,
  plots: document.querySelectorAll('#validate-gallery .gallery-item').length,
}));
await page.screenshot({ path: join(output, 'studio_validate.png') });
await page.selectOption('#validate-run', 'live_full_official');
await page.waitForTimeout(1200);
const official = await page.evaluate(() => document.querySelector('#validate-header').textContent);

await goto('compare');
await page.waitForSelector('#compare-runs .check-row', { timeout: 20000 });
const compare = await page.evaluate(() => {
  const canvas = document.querySelector('#compare-chart');
  const rect = canvas.getBoundingClientRect();
  const parent = canvas.parentElement.getBoundingClientRect();
  return { width: canvas.width, cssWidth: Math.round(rect.width), parentWidth: Math.round(parent.width) };
});
await page.screenshot({ path: join(output, 'studio_compare.png') });

await browser.close();
console.log(JSON.stringify({ preview, train, validate, official: official.slice(0, 80), compare, errors }, null, 2));
if (errors.length) { process.exitCode = 1; }
