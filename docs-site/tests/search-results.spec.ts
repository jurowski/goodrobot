import { test, expect } from '@playwright/test';

test.describe('Search Results Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/test-search');
    // Ensure the page is fully loaded
    await page.waitForLoadState('networkidle');
  });

  test('should render initial search results correctly', async ({ page }) => {
    // Check basic rendering
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
    
    // Take a screenshot for visual comparison
    await expect(page).toHaveScreenshot('initial-search-results.png', {
      mask: [page.locator('[data-testid="last-updated"]')] // Mask dynamic content
    });
  });

  test('should handle hover states correctly', async ({ page }) => {
    const firstResult = page.locator('[data-testid="search-result"]').first();
    
    // Capture before hover
    await expect(firstResult).toHaveScreenshot('result-before-hover.png');
    
    // Hover and capture
    await firstResult.hover();
    await expect(firstResult).toHaveScreenshot('result-after-hover.png');
    
    // Verify styles
    const hoverStyles = await firstResult.evaluate((el) => {
      const styles = window.getComputedStyle(el);
      return {
        transform: styles.transform,
        boxShadow: styles.boxShadow
      };
    });
    
    expect(hoverStyles.transform).toContain('translateY(-2px)');
    expect(hoverStyles.boxShadow).toContain('rgba(0, 0, 0, 0.1)');
  });

  test('should handle search interactions', async ({ page }) => {
    const searchInput = page.locator('[data-testid="search-input"]');
    
    // Test search input
    await searchInput.fill('lock');
    await page.waitForTimeout(300); // Wait for debounce
    
    // Take screenshot of search results
    await expect(page).toHaveScreenshot('search-results-filtered.png');
    
    // Verify filtered results
    const results = await page.locator('[data-testid="search-result"]').count();
    expect(results).toBeGreaterThan(0);
  });

  test('should handle category filtering', async ({ page }) => {
    const categoryFilter = page.locator('[data-testid="category-filter"]').first();
    
    // Click category and verify filter
    await categoryFilter.click();
    await page.waitForTimeout(300);
    
    // Take screenshot of filtered results
    await expect(page).toHaveScreenshot('category-filtered-results.png');
  });

  test('should handle focus states', async ({ page }) => {
    const searchInput = page.locator('[data-testid="search-input"]');
    
    // Test focus states
    await searchInput.focus();
    await expect(page).toHaveScreenshot('search-input-focused.png');
  });

  test('should handle keyboard navigation', async ({ page }) => {
    // Tab to first result
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    
    // Take screenshot of focused result
    await expect(page).toHaveScreenshot('keyboard-navigation.png');
    
    // Verify focus indicator
    const focusedElement = await page.evaluate(() => {
      const activeElement = document.activeElement;
      return activeElement ? activeElement.getAttribute('data-testid') : null;
    });
    
    expect(focusedElement).toBe('search-result');
  });

  test('should handle loading states', async ({ page }) => {
    // Simulate slow network
    await page.route('**/api/documentation', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.continue();
    });
    
    await page.reload();
    
    // Verify loading state
    await expect(page.locator('[data-testid="loading-state"]')).toBeVisible();
    await expect(page).toHaveScreenshot('loading-state.png');
  });

  test('should handle error states', async ({ page }) => {
    // Simulate error
    await page.route('**/api/documentation', route => route.abort());
    
    await page.reload();
    
    // Verify error state
    await expect(page.locator('[data-testid="error-state"]')).toBeVisible();
    await expect(page).toHaveScreenshot('error-state.png');
  });
});
