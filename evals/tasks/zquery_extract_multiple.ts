/**
 * Custom evaluation for zQuery's extractFromMultiple method
 * Tests our unique dynamic extraction capability
 */

import type { StagehandInitResult } from "@/types/evals";
// Import our custom wrapper instead of official Stagehand
import { UnifiedStagehandManager } from "../../../src/stagehand_wrapper/stagehand.js";

export const zquery_extract_multiple = async ({}: StagehandInitResult) => {
  try {
    // Create our custom wrapper instance for testing
    const zqueryStagehand = new UnifiedStagehandManager({
      modelName: 'claude-3-5-sonnet-latest',
      verbose: 1,
      headless: true
    });

    await zqueryStagehand.init();

    // Test our custom extractFromMultiple method
    const results = await zqueryStagehand.extractFromMultiple({
      url: "https://news.ycombinator.com",
      itemDescription: "top story",
      extractionFields: ["title", "points"],
      count: 3
    });

    await zqueryStagehand.close();

    // Validate results
    const success = results.items.length >= 2 && 
                   results.items.some(item => item.title && item.title !== "title not found");

    return {
      _success: success,
      extractedItems: results.items.length,
      successfulExtractions: results.items.filter(item => item.success).length,
      results: results
    };

  } catch (error) {
    return {
      _success: false,
      error: error.message
    };
  }
};