// R16-T7 report-completeness condition 5 (docs/ai/r16_t7_scoring_freeze.md §6): the live
// `/app/report/{id}` renders in EN and TR with HTTP 200, no page error, and the analysis id and
// the media SHA-256 visible. Prints one JSON line; screenshots go next to the case record.
//   PW_CORE=... CHROME=... node r16_t7_render_check.mjs <web> <token> <analysis_id> <sha256> <out_dir>
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)(process.env.PW_CORE);

const [web, token, id, sha, outDir] = process.argv.slice(2);
const result = { ok: true, locales: {} };
const browser = await chromium.launch({ executablePath: process.env.CHROME });
try {
  for (const locale of ["en", "tr"]) {
    const errors = [];
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    await ctx.addCookies([{ name: "deepguard_session", value: token, url: web }]);
    await ctx.addInitScript((l) => localStorage.setItem("inspectroot.locale", l), locale);
    const page = await ctx.newPage();
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    const res = await page.goto(`${web}/app/report/${id}`, { waitUntil: "networkidle", timeout: 120000 });
    await page.waitForTimeout(500);
    const text = await page.evaluate(() => document.body.innerText);
    const htmlLang = await page.evaluate(() => document.documentElement.lang);
    const r = {
      http_status: res ? res.status() : null,
      final_url: page.url(),
      page_errors: errors,
      id_visible: text.includes(id),
      sha256_visible: text.includes(sha),
      html_lang: htmlLang,
      text_chars: text.length,
    };
    r.ok = r.http_status === 200 && errors.length === 0 && r.id_visible && r.sha256_visible
      && r.final_url.includes(`/app/report/${id}`);
    await page.screenshot({ path: `${outDir}/report-${locale}.png` });
    result.locales[locale] = r;
    result.ok = result.ok && r.ok;
    await ctx.close();
  }
} catch (e) {
  result.ok = false;
  result.error = String(e && e.message ? e.message : e).slice(0, 500);
} finally {
  await browser.close();
}
console.log(JSON.stringify(result));
