/**
 * ZQUERY CUSTOM EVALUATION INITIALIZER
 * This version uses our UnifiedStagehandManager wrapper instead of base Stagehand
 * Tests that our wrapper is fully compatible with all official evaluations
 */

import { enableCaching } from "./env";
import { AvailableModel, LLMClient } from "@browserbasehq/stagehand";
import { EvalLogger } from "./logger";
import type { StagehandInitResult } from "@/types/evals";
// Import our custom wrapper
import { UnifiedStagehandManager } from "../../src/stagehand_wrapper/stagehand.js";

/**
 * Configuration optimized for LOCAL-only evaluation using our custom wrapper
 */
const ZQueryStagehandConfig = {
  env: "LOCAL" as const,
  verbose: 2 as const,
  debugDom: true,
  headless: true, // Headless for faster evaluation
  enableCaching,
  domSettleTimeoutMs: 30_000,
  disablePino: true,
  experimental: true,
};

/**
 * Initialize our UnifiedStagehandManager for evaluation
 * This tests our wrapper's compatibility with the official evaluation framework
 */
export const initStagehandZQuery = async ({
  llmClient,
  domSettleTimeoutMs,
  logger,
  configOverrides,
  actTimeoutMs,
  modelName,
}: {
  llmClient: LLMClient;
  domSettleTimeoutMs?: number;
  logger: EvalLogger;
  configOverrides?: any;
  actTimeoutMs?: number;
  modelName: AvailableModel;
}): Promise<StagehandInitResult> => {
  
  // Create config for our wrapper
  const wrapperConfig = {
    ...ZQueryStagehandConfig,
    modelName: modelName,
    ...(domSettleTimeoutMs && { domSettleTimeoutMs }),
    ...(actTimeoutMs && { actTimeoutMs }),
    ...configOverrides,
  };

  // Create our custom wrapper instance
  const zqueryStagehand = new UnifiedStagehandManager(wrapperConfig);

  // Initialize the wrapper
  await zqueryStagehand.init();

  // Associate logger with the underlying Stagehand instance
  logger.init(zqueryStagehand.stagehand);

  console.log(`🎯 ZQUERY EVALUATION: Testing UnifiedStagehandManager with ${modelName}`);

  return {
    stagehand: zqueryStagehand.stagehand, // Return underlying Stagehand for compatibility
    stagehandConfig: wrapperConfig,
    logger,
    debugUrl: "", // Our wrapper doesn't expose debugUrl yet
    sessionUrl: "", // Our wrapper doesn't expose sessionUrl yet
    modelName,
    // Add our wrapper instance for custom testing
    zqueryWrapper: zqueryStagehand,
  };
};