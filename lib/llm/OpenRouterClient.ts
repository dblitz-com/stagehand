import OpenAI from "openai";
import type { ClientOptions } from "openai";
import {
  ChatCompletionContentPartImage,
  ChatCompletionContentPartText,
  ChatCompletionMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
} from "openai/resources/chat";
import zodToJsonSchema from "zod-to-json-schema";
import { LogLine } from "../../types/log";
import { AvailableModel } from "../../types/model";
import { LLMCache } from "../cache/LLMCache";
import { validateZodSchema } from "../utils";
import {
  CreateChatCompletionOptions,
  LLMClient,
  LLMResponse,
} from "./LLMClient";
import {
  CreateChatCompletionResponseError,
  ZodSchemaValidationError,
} from "@/types/stagehandErrors";

export class OpenRouterClient extends LLMClient {
  public type = "openrouter" as const;
  private client: OpenAI;
  private cache: LLMCache | undefined;
  private enableCaching: boolean;
  public clientOptions: ClientOptions;
  public hasVision = true; // Grok 4 supports vision

  constructor({
    enableCaching = false,
    cache,
    modelName,
    clientOptions,
    userProvidedInstructions,
  }: {
    logger: (message: LogLine) => void;
    enableCaching?: boolean;
    cache?: LLMCache;
    modelName: AvailableModel;
    clientOptions?: ClientOptions;
    userProvidedInstructions?: string;
  }) {
    super(modelName, userProvidedInstructions);

    // Create OpenAI client with OpenRouter API
    this.client = new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: clientOptions?.apiKey || process.env.OPENROUTER_API_KEY,
      defaultHeaders: {
        "HTTP-Referer": "https://stagehand.dev",
        "X-Title": "Stagehand",
      },
      ...clientOptions,
    });

    this.cache = cache;
    this.enableCaching = enableCaching;
    this.modelName = modelName;
    this.clientOptions = clientOptions;
  }

  async createChatCompletion<T = LLMResponse>({
    options,
    retries,
    logger,
  }: CreateChatCompletionOptions): Promise<T> {
    logger({
      category: "openrouter",
      message: "creating chat completion",
      level: 2,
      auxiliary: {
        options: {
          value: JSON.stringify(options),
          type: "object",
        },
      },
    });

    const cacheOptions = {
      model: this.modelName,
      messages: options.messages,
      temperature: options.temperature,
      top_p: options.top_p,
      frequency_penalty: options.frequency_penalty,
      presence_penalty: options.presence_penalty,
      image: options.image,
      response_model: options.response_model,
      tools: options.tools,
      tool_choice: options.tool_choice,
      maxTokens: options.maxTokens,
    };

    if (this.enableCaching) {
      const cachedResponse = await this.cache.get<T>(
        cacheOptions,
        options.requestId,
      );
      if (cachedResponse) {
        logger({
          category: "llm_cache",
          message: "LLM cache hit - returning cached response",
          level: 1,
          auxiliary: {
            requestId: {
              value: options.requestId,
              type: "string",
            },
            cachedResponse: {
              value: JSON.stringify(cachedResponse),
              type: "object",
            },
          },
        });
        return cachedResponse;
      } else {
        logger({
          category: "llm_cache",
          message: "LLM cache miss - no cached response found",
          level: 1,
          auxiliary: {
            requestId: {
              value: options.requestId,
              type: "string",
            },
          },
        });
      }
    }

    // Format messages for OpenRouter API (using OpenAI format)
    const formattedMessages: ChatCompletionMessageParam[] =
      options.messages.map((message) => {
        if (Array.isArray(message.content)) {
          const contentParts = message.content.map((content) => {
            if ("image_url" in content) {
              const imageContent: ChatCompletionContentPartImage = {
                image_url: {
                  url: content.image_url.url,
                },
                type: "image_url",
              };
              return imageContent;
            } else {
              const textContent: ChatCompletionContentPartText = {
                text: content.text,
                type: "text",
              };
              return textContent;
            }
          });

          if (message.role === "system") {
            const formattedMessage: ChatCompletionSystemMessageParam = {
              ...message,
              role: "system",
              content: contentParts.filter(
                (content): content is ChatCompletionContentPartText =>
                  content.type === "text",
              ),
            };
            return formattedMessage;
          } else if (message.role === "user") {
            const formattedMessage: ChatCompletionUserMessageParam = {
              ...message,
              role: "user",
              content: contentParts,
            };
            return formattedMessage;
          } else {
            const formattedMessage: ChatCompletionAssistantMessageParam = {
              ...message,
              role: "assistant",
              content: contentParts.filter(
                (content): content is ChatCompletionContentPartText =>
                  content.type === "text",
              ),
            };
            return formattedMessage;
          }
        }

        if (message.role === "system") {
          const formattedMessage: ChatCompletionSystemMessageParam = {
            role: "system",
            content: message.content,
          };
          return formattedMessage;
        } else if (message.role === "assistant") {
          const formattedMessage: ChatCompletionAssistantMessageParam = {
            role: "assistant",
            content: message.content,
          };
          return formattedMessage;
        } else {
          const formattedMessage: ChatCompletionUserMessageParam = {
            role: "user",
            content: message.content,
          };
          return formattedMessage;
        }
      });

    // Add image if provided
    if (options.image) {
      const base64Image = options.image.buffer.toString("base64");
      const imageMessage = {
        role: "user" as const,
        content: [
          {
            type: "text" as const,
            text: options.image.description || "Please analyze this image.",
          },
          {
            type: "image_url" as const,
            image_url: {
              url: `data:image/png;base64,${base64Image}`,
            },
          },
        ],
      };
      formattedMessages.push(imageMessage);
    }

    // Format tools if provided (only user-defined tools, not response models)
    const tools = options.tools?.map((tool) => ({
      type: "function" as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: {
          type: "object",
          properties: tool.parameters.properties,
          required: tool.parameters.required,
        },
      },
    }));

    // Handle response format for structured outputs
    // OpenRouter only supports response_format for certain models (OpenAI, Nitro, etc.)
    // Since we can't reliably detect which models support it, always use instruction-based approach
    if (options.response_model) {
      logger({
        category: "openrouter",
        message: "Using instruction-based approach for response model",
        level: 1,
        auxiliary: {
          requestId: {
            value: options.requestId,
            type: "string",
          },
        },
      });

      const parsedSchema = JSON.stringify(
        zodToJsonSchema(options.response_model.schema),
      );
      formattedMessages.push({
        role: "user",
        content: `Respond with ONLY valid JSON that matches this exact schema:\n${parsedSchema}\n\nIMPORTANT: Your response must be valid JSON with no additional text, explanations, or markdown formatting. Start your response with '{' and end with '}'. Do not use \`\`\`json or any other formatting.`,
      });
    }

    try {
      // Use OpenAI client with OpenRouter API
      const apiResponse = await this.client.chat.completions.create({
        model: this.modelName,
        messages: formattedMessages,
        temperature: options.temperature || 0.7,
        max_tokens: options.maxTokens,
        tools: tools,
        tool_choice:
          tools && tools.length > 0 ? options.tool_choice || "auto" : undefined,
        top_p: options.top_p,
        frequency_penalty: options.frequency_penalty,
        presence_penalty: options.presence_penalty,
      });

      // Format the response to match the expected LLMResponse format
      const response: LLMResponse = {
        id: apiResponse.id,
        object: "chat.completion",
        created: apiResponse.created,
        model: this.modelName,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: apiResponse.choices[0]?.message?.content || null,
              tool_calls: apiResponse.choices[0]?.message?.tool_calls || [],
            },
            finish_reason: apiResponse.choices[0]?.finish_reason || "stop",
          },
        ],
        usage: {
          prompt_tokens: apiResponse.usage?.prompt_tokens || 0,
          completion_tokens: apiResponse.usage?.completion_tokens || 0,
          total_tokens: apiResponse.usage?.total_tokens || 0,
        },
      };

      logger({
        category: "openrouter",
        message: "OpenRouter chat completion finished",
        level: 1,
        auxiliary: {
          response: {
            value: JSON.stringify(response),
            type: "object",
          },
          requestId: {
            value: options.requestId,
            type: "string",
          },
        },
      });

      // Handle response_model extraction
      if (options.response_model) {
        let extractedData;

        // Check if we have content in the message
        if (response.choices[0].message.content) {
          extractedData = response.choices[0].message.content;
        }
        // Check if the model returned a tool call instead (some models do this)
        else if (
          response.choices[0].message.tool_calls &&
          response.choices[0].message.tool_calls.length > 0
        ) {
          // Try to extract from tool calls as a fallback
          const toolCall = response.choices[0].message.tool_calls[0];
          extractedData = toolCall.function.arguments;
        } else {
          logger({
            category: "openrouter",
            message: "No content or tool calls found in OpenRouter response",
            level: 0,
            auxiliary: {
              response: {
                value: JSON.stringify(response),
                type: "object",
              },
              requestId: {
                value: options.requestId,
                type: "string",
              },
            },
          });

          if (retries > 0) {
            return this.createChatCompletion({
              options,
              retries: retries - 1,
              logger,
            });
          }

          throw new CreateChatCompletionResponseError(
            "No content or tool calls found in OpenRouter response for response_model",
          );
        }

        let parsedData;
        try {
          parsedData = JSON.parse(extractedData);
        } catch (parseError) {
          // Try to extract JSON from the response if it's wrapped in text or markdown
          let cleanedData = extractedData;

          // Remove markdown code blocks
          cleanedData = cleanedData
            .replace(/```json\n?/g, "")
            .replace(/```\n?/g, "");

          // Try to find JSON object in the text
          const jsonMatch = cleanedData.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            try {
              parsedData = JSON.parse(jsonMatch[0]);
            } catch (secondParseError) {
              logger({
                category: "openrouter",
                message: "Failed to parse cleaned response as JSON",
                level: 0,
                auxiliary: {
                  originalData: {
                    value: extractedData,
                    type: "string",
                  },
                  cleanedData: {
                    value: jsonMatch[0],
                    type: "string",
                  },
                  parseError: {
                    value: secondParseError.message,
                    type: "string",
                  },
                  requestId: {
                    value: options.requestId,
                    type: "string",
                  },
                },
              });

              if (retries > 0) {
                return this.createChatCompletion({
                  options,
                  retries: retries - 1,
                  logger,
                });
              }

              throw new CreateChatCompletionResponseError(
                `Failed to parse OpenRouter response as JSON: ${secondParseError.message}`,
              );
            }
          } else {
            logger({
              category: "openrouter",
              message: "No JSON found in response",
              level: 0,
              auxiliary: {
                extractedData: {
                  value: extractedData,
                  type: "string",
                },
                parseError: {
                  value: parseError.message,
                  type: "string",
                },
                requestId: {
                  value: options.requestId,
                  type: "string",
                },
              },
            });

            if (retries > 0) {
              return this.createChatCompletion({
                options,
                retries: retries - 1,
                logger,
              });
            }

            throw new CreateChatCompletionResponseError(
              `No JSON found in OpenRouter response: ${parseError.message}`,
            );
          }
        }

        try {
          validateZodSchema(options.response_model.schema, parsedData);
        } catch (e) {
          logger({
            category: "openrouter",
            message: "Response failed Zod schema validation",
            level: 0,
            auxiliary: {
              parsedData: {
                value: JSON.stringify(parsedData),
                type: "object",
              },
              requestId: {
                value: options.requestId,
                type: "string",
              },
            },
          });
          if (retries > 0) {
            return this.createChatCompletion({
              options,
              retries: retries - 1,
              logger,
            });
          }

          if (e instanceof ZodSchemaValidationError) {
            logger({
              category: "openrouter",
              message: `Error during OpenRouter chat completion: ${e.message}`,
              level: 0,
              auxiliary: {
                errorDetails: {
                  value: `Message: ${e.message}${e.stack ? "\nStack: " + e.stack : ""}`,
                  type: "string",
                },
                requestId: { value: options.requestId, type: "string" },
              },
            });
            throw new CreateChatCompletionResponseError(e.message);
          }
          throw e;
        }

        const result = {
          data: parsedData,
          usage: response.usage,
        } as T;

        if (this.enableCaching) {
          await this.cache.set(cacheOptions, result, options.requestId);
        }

        return result;
      }

      if (this.enableCaching) {
        await this.cache.set(cacheOptions, response, options.requestId);
      }

      return response as T;
    } catch (error) {
      logger({
        category: "openrouter",
        message: `OpenRouter request failed: ${error.message}`,
        level: 0,
        auxiliary: {
          error: {
            value: error.message,
            type: "string",
          },
          requestId: {
            value: options.requestId,
            type: "string",
          },
        },
      });

      if (retries > 0) {
        logger({
          category: "openrouter",
          message: `retrying OpenRouter request, ${retries} attempts remaining`,
          level: 1,
          auxiliary: {
            requestId: {
              value: options.requestId,
              type: "string",
            },
          },
        });
        return this.createChatCompletion({
          options,
          retries: retries - 1,
          logger,
        });
      }

      throw new CreateChatCompletionResponseError(
        `OpenRouter request failed: ${error.message}`,
      );
    }
  }
}
