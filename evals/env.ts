/**
 * Environment is always LOCAL for dblitz-com fork.
 * We run all evaluations locally with headless Chromium.
 */
export const env = "LOCAL" as const;

/**
 * Enable or disable caching based on the EVAL_ENABLE_CACHING environment variable.
 * Caching may improve performance by not re-fetching or re-computing certain results.
 * By default, caching is disabled unless explicitly enabled.
 */
export const enableCaching =
  process.env.EVAL_ENABLE_CACHING?.toLowerCase() === "true";
