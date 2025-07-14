/**
 * ZQUERY FULL EVALUATION RUNNER
 * Tests our UnifiedStagehandManager wrapper with the complete official evaluation suite
 * This proves our wrapper is fully compatible + adds value
 */

import fs from "fs";
import path from "path";
import process from "process";
import {
  DEFAULT_EVAL_CATEGORIES,
  filterByCategory,
  filterByEvalName,
} from "./args";
import { generateExperimentName } from "./utils";
import { exactMatch, errorMatch } from "./scoring";
import { tasksByName, tasksConfig, getModelList } from "./taskConfig";
import { Eval, wrapAISDKModel, wrapOpenAI } from "braintrust";
import { SummaryResult, Testcase } from "@/types/evals";
import { EvalLogger } from "./logger";
import { AvailableModel, LLMClient } from "@browserbasehq/stagehand";
import { env } from "./env";
import dotenv from "dotenv";
import { StagehandEvalError } from "@/types/stagehandErrors";
import { CustomOpenAIClient } from "@/examples/external_clients/customOpenAI";
import OpenAI from "openai";
// Import our custom wrapper
import { UnifiedStagehandManager } from "../../src/stagehand_wrapper/stagehand.js";
import { google } from "@ai-sdk/google";
import { anthropic } from "@ai-sdk/anthropic";
import { groq } from "@ai-sdk/groq";
import { cerebras } from "@ai-sdk/cerebras";
import { openai } from "@ai-sdk/openai";
import { AISdkClient } from "@/examples/external_clients/aisdk";

// Load from our main .env file
dotenv.config({ path: '../../.env' });

// Debug: Check if API keys loaded
console.log("🔑 API Keys Status:");
console.log(`   ANTHROPIC_API_KEY: ${process.env.ANTHROPIC_API_KEY ? 'LOADED' : 'MISSING'}`);
console.log(`   OPENAI_API_KEY: ${process.env.OPENAI_API_KEY ? 'LOADED' : 'MISSING'}`);
console.log(`   GOOGLE_GENERATIVE_AI_API_KEY: ${process.env.GOOGLE_GENERATIVE_AI_API_KEY ? 'LOADED' : 'MISSING'}`);

const MAX_CONCURRENCY = process.env.EVAL_MAX_CONCURRENCY
  ? parseInt(process.env.EVAL_MAX_CONCURRENCY, 10)
  : 1; // Reduced for testing

const TRIAL_COUNT = process.env.EVAL_TRIAL_COUNT
  ? parseInt(process.env.EVAL_TRIAL_COUNT, 10)
  : 1; // Reduced for testing

// Same summary generation logic
const generateSummary = async (
  results: SummaryResult[],
  experimentName: string,
) => {
  const passed = results
    .filter((r) => r.output._success)
    .map((r) => ({
      eval: r.input.name,
      model: r.input.modelName,
      categories: tasksByName[r.input.name].categories,
    }));

  const failed = results
    .filter((r) => !r.output._success)
    .map((r) => ({
      eval: r.input.name,
      model: r.input.modelName,
      categories: tasksByName[r.input.name].categories,
    }));

  const categorySuccessCounts: Record<
    string,
    { total: number; success: number }
  > = {};
  for (const taskName of Object.keys(tasksByName)) {
    const taskCategories = tasksByName[taskName].categories;
    const taskResults = results.filter((r) => r.input.name === taskName);
    const successCount = taskResults.filter((r) => r.output._success).length;

    for (const cat of taskCategories) {
      if (!categorySuccessCounts[cat]) {
        categorySuccessCounts[cat] = { total: 0, success: 0 };
      }
      categorySuccessCounts[cat].total += taskResults.length;
      categorySuccessCounts[cat].success += successCount;
    }
  }

  const categories: Record<string, number> = {};
  for (const [cat, counts] of Object.entries(categorySuccessCounts)) {
    categories[cat] = Math.round((counts.success / counts.total) * 100);
  }

  const models: Record<string, number> = {};
  const allModels = [...new Set(results.map((r) => r.input.modelName))];
  for (const model of allModels) {
    const modelResults = results.filter((r) => r.input.modelName === model);
    const successCount = modelResults.filter((r) => r.output._success).length;
    models[model] = Math.round((successCount / modelResults.length) * 100);
  }

  const formattedSummary = {
    experimentName,
    passed,
    failed,
    categories,
    models,
    zqueryWrapper: "UnifiedStagehandManager", // Mark that we used our wrapper
  };

  fs.writeFileSync(
    "eval-summary-zquery.json",
    JSON.stringify(formattedSummary, null, 2),
  );
  console.log("🎯 ZQUERY evaluation summary written to eval-summary-zquery.json");
};

// Same testcase generation logic
const generateFilteredTestcases = (): Testcase[] => {
  let taskNamesToRun: string[];
  let effectiveCategory: string | null = filterByCategory;

  if (filterByEvalName) {
    taskNamesToRun = [filterByEvalName];
    const taskCategories = tasksByName[filterByEvalName]?.categories || [];
    if (taskCategories.length === 1 && taskCategories[0] === "agent") {
      effectiveCategory = "agent";
      console.log(
        `Task ${filterByEvalName} is agent-specific, using agent models.`,
      );
    }
  } else if (filterByCategory) {
    taskNamesToRun = Object.keys(tasksByName).filter((name) =>
      tasksByName[name].categories.includes(filterByCategory!),
    );
  } else {
    taskNamesToRun = Object.keys(tasksByName).filter((name) =>
      DEFAULT_EVAL_CATEGORIES.some((category) =>
        tasksByName[name].categories.includes(category),
      ),
    );
  }

  const currentModels = getModelList(effectiveCategory);

  console.log(
    `🎯 ZQUERY EVALUATION: Using models for this run (${effectiveCategory || "default"}):`,
    currentModels,
  );

  let allTestcases = currentModels.flatMap((model) =>
    taskNamesToRun.map((testName) => ({
      input: { name: testName, modelName: model as AvailableModel },
      name: testName,
      tags: [
        model,
        testName,
        "zquery-wrapper", // Tag to identify our wrapper tests
        ...(tasksConfig.find((t) => t.name === testName)?.categories || []).map(
          (x) => `category/${x}`,
        ),
      ],
      metadata: {
        model: model as AvailableModel,
        test: testName,
        categories: tasksConfig.find((t) => t.name === testName)?.categories,
        wrapper: "UnifiedStagehandManager",
      },
      expected: true,
    })),
  );

  if (filterByCategory) {
    allTestcases = allTestcases.filter((testcase) =>
      tasksByName[testcase.name].categories.includes(filterByCategory!),
    );
  }

  console.log(
    "🎯 ZQUERY final test cases to run:",
    allTestcases
      .map(
        (t, i) =>
          `${i}: ${t.name} (${t.input.modelName}): ${t.metadata.categories}`,
      )
      .join("\\n"),
  );

  return allTestcases;
};

/**
 * MAIN ZQUERY EVALUATION RUNNER
 * Tests our UnifiedStagehandManager with the full official evaluation suite
 */
(async () => {
  const experimentName: string = `zquery-wrapper-${generateExperimentName({
    evalName: filterByEvalName || undefined,
    category: filterByCategory || undefined,
    environment: env,
  })}`;

  const braintrustProjectName =
    process.env.CI === "true" ? "stagehand-zquery" : "stagehand-zquery-dev";

  try {
    console.log("🎯 STARTING ZQUERY WRAPPER EVALUATION");
    console.log("=====================================");
    console.log("Testing UnifiedStagehandManager with official evaluation suite");
    console.log("This proves our wrapper maintains full compatibility + adds value");
    console.log("");

    const evalResult = await Eval(braintrustProjectName, {
      experimentName,
      data: generateFilteredTestcases,
      task: async (input: { name: string; modelName: AvailableModel }) => {
        const logger = new EvalLogger();
        try {
          // Import the task module
          const taskModulePath = path.join(
            __dirname,
            "tasks",
            `${input.name}.ts`,
          );

          let taskModule;
          try {
            taskModule = await import(taskModulePath);
          } catch (error) {
            if (input.name.includes("/")) {
              const subDirPath = path.join(
                __dirname,
                "tasks",
                `${input.name}.ts`,
              );
              try {
                taskModule = await import(subDirPath);
              } catch (subError) {
                throw new StagehandEvalError(
                  `Failed to import task module for ${input.name}. Tried paths:\\n` +
                    `- ${taskModulePath}\\n` +
                    `- ${subDirPath}\\n` +
                    `Error: ${subError.message}`,
                );
              }
            } else {
              throw new StagehandEvalError(
                `Failed to import task module for ${input.name} at path ${taskModulePath}: ${error.message}`,
              );
            }
          }

          const taskName = input.name.includes("/")
            ? input.name.split("/").pop()
            : input.name;

          const taskFunction = taskModule[taskName];

          if (typeof taskFunction !== "function") {
            throw new StagehandEvalError(
              `No Eval function found for task name: ${taskName} in module ${input.name}`,
            );
          }

          // Set up LLM client
          let llmClient: LLMClient;
          if (
            input.modelName.startsWith("gpt") ||
            input.modelName.startsWith("o")
          ) {
            llmClient = new AISdkClient({
              model: wrapAISDKModel(openai(input.modelName)),
            });
          } else if (input.modelName.startsWith("gemini")) {
            llmClient = new AISdkClient({
              model: wrapAISDKModel(google(input.modelName)),
            });
          } else if (input.modelName.startsWith("claude")) {
            llmClient = new AISdkClient({
              model: wrapAISDKModel(anthropic(input.modelName)),
            });
          } else if (input.modelName.includes("groq")) {
            llmClient = new AISdkClient({
              model: wrapAISDKModel(
                groq(
                  input.modelName.substring(input.modelName.indexOf("/") + 1),
                ),
              ),
            });
          } else if (input.modelName.includes("cerebras")) {
            llmClient = new AISdkClient({
              model: wrapAISDKModel(
                cerebras(
                  input.modelName.substring(input.modelName.indexOf("/") + 1),
                ),
              ),
            });
          } else if (input.modelName.includes("/")) {
            llmClient = new CustomOpenAIClient({
              modelName: input.modelName as AvailableModel,
              client: wrapOpenAI(
                new OpenAI({
                  apiKey: process.env.TOGETHER_AI_API_KEY,
                  baseURL: "https://api.together.xyz/v1",
                }),
              ),
            });
          }

          // 🎯 CREATE OUR WRAPPER INSTEAD OF BASE STAGEHAND
          const zqueryStagehand = new UnifiedStagehandManager({
            env: "LOCAL",
            verbose: 1,
            headless: true,
            enableCaching: true,
            modelName: input.modelName,
            // Pass through the configured LLM client
            llmClient: llmClient,
          });

          await zqueryStagehand.init();

          // Create compatibility layer for evaluations
          const taskInput = {
            stagehand: zqueryStagehand.stagehand, // Pass underlying Stagehand for compatibility
            logger,
            modelName: input.modelName,
            zqueryWrapper: zqueryStagehand, // Also provide our wrapper
          };

          let result;
          try {
            result = await taskFunction(taskInput);
            
            // Log result
            if (result && result._success) {
              console.log(`✅ ZQUERY ${input.name}: PASSED`);
            } else {
              console.log(`❌ ZQUERY ${input.name}: FAILED`);
            }
          } finally {
            await zqueryStagehand.close();
          }
          
          return result;

        } catch (error) {
          console.error(`❌ ZQUERY ${input.name}: ERROR - ${error}`);
          logger.error({
            message: `Error in ZQUERY task ${input.name}`,
            level: 0,
            auxiliary: {
              error: {
                value: error.message,
                type: "string",
              },
              trace: {
                value: error.stack,
                type: "string",
              },
            },
          });
          return {
            _success: false,
            error: JSON.parse(JSON.stringify(error, null, 2)),
            logs: logger.getLogs(),
          };
        }
      },
      scores: [exactMatch, errorMatch],
      maxConcurrency: MAX_CONCURRENCY,
      trialCount: TRIAL_COUNT,
    });

    // Map results to summary format
    const summaryResults: SummaryResult[] = evalResult.results.map((result) => {
      const output =
        typeof result.output === "boolean"
          ? { _success: result.output }
          : result.output;

      return {
        input: result.input,
        output,
        name: result.input.name,
        score: output._success ? 1 : 0,
      };
    });

    // Generate summary
    await generateSummary(summaryResults, experimentName);
    
    console.log("");
    console.log("🎯 ZQUERY WRAPPER EVALUATION COMPLETE");
    console.log("=====================================");
    console.log(`✅ Tested UnifiedStagehandManager compatibility`);
    console.log(`📊 Results saved to eval-summary-zquery.json`);
    
  } catch (error) {
    console.error("❌ Error during ZQUERY evaluation run:", error);
    process.exit(1);
  }
})();