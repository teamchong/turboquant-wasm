import { test, expect } from "@playwright/test";

test.describe("GPU Shader Tests", () => {
  test("all shaders pass on GPU", async ({ page }) => {
    // Navigate to the GPU test page
    await page.goto("/gpu-test.html");

    // Wait for tests to complete (look for the summary line)
    await page.waitForSelector("pre:has-text('Results:')", { timeout: 15000 });

    // Get all test results
    const results = await page.locator("#log pre").allTextContents();

    // Collect failures
    const failures = results.filter(r => r.startsWith("FAIL"));
    const passes = results.filter(r => r.startsWith("PASS"));

    // Log all results for visibility
    for (const r of results) {
      if (r.startsWith("PASS") || r.startsWith("FAIL") || r.startsWith("---")) {
        console.log(r);
      }
    }

    console.log(`\n${passes.length} passed, ${failures.length} failed`);

    // Assert no failures
    expect(failures, `GPU shader failures:\n${failures.join("\n")}`).toHaveLength(0);
    expect(passes.length).toBeGreaterThan(0);
  });
});
