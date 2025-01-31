import { z } from "zod"
import { groq } from "@ai-sdk/groq"
import {
  generateText,
  extractReasoningMiddleware,
  wrapLanguageModel,
  streamText,
  generateObject,
  streamObject,
  type LanguageModel,
  type LanguageModelUsage,
  type CoreMessage
} from "ai"


const groqModel: LanguageModel = wrapLanguageModel({
  model: groq("deepseek-r1-distill-llama-70b"),
  middleware: extractReasoningMiddleware({ tagName: "think" }),
})


export type AITextResponse = {
  text: string
  reasoning: string | undefined
  usage: LanguageModelUsage
}

export type AITextResponseStream = {
  textStream: AsyncIterable<string>
  reasoning: Promise<string | undefined>
}

export type AIObjectResponse = {
  object: any
  usage: LanguageModelUsage
}


/*
 * Chat with an AI
 */
export async function chat(messages: CoreMessage[], model: LanguageModel = groqModel): Promise<AITextResponse> {
  const { text, reasoning, usage } = await generateText({
    model,
    messages
  })
  return { text, reasoning, usage }
}


/*
 * Chat with an AI (stream)
 */
export async function streamChat(messages: CoreMessage[], model: LanguageModel = groqModel): Promise<AITextResponseStream> {
  const { textStream, reasoning } = await streamText({
    model,
    messages
  })
  // we can get the final text block by awaiting
  // const finalText = await textStream
  return { textStream, reasoning }
}


/*
 * Generate structured output from an AI
 */
export async function createObject(messages: CoreMessage[], schema: z.ZodSchema, model: LanguageModel = groqModel): Promise<AIObjectResponse> {
  const { object, usage } = await generateObject({
    model,
    messages,
    schema
  })
  return { object, usage }
}


/*
 * Generate structured output from an AI (stream)
 */
export async function streamCreateObject(messages: CoreMessage[], schema: z.ZodSchema, model: LanguageModel = groqModel): Promise<any> {
  const { partialObjectStream } = await streamObject({
    model,
    messages,
    schema
  })
  // we can get the final object by awaiting
  // const finalObject = await partialObjectStream
  return { partialObjectStream }
}


/*
 * Classify text into a category
 */
export async function classifyText(messages: CoreMessage[], categories: string[], model: LanguageModel = groqModel): Promise<AIObjectResponse> {
  const { object, usage } = await generateObject({
    model,
    messages,
    output: "enum",
    enum: categories
  })
  return { object, usage }
}


/*
 * Share an image (file path) and chat about it
 */
export async function chatWithImageFilePath(messages: CoreMessage[], imageFilePath: string, model: LanguageModel = groqModel): Promise<AITextResponse> {
  const imageAsUint8Array = await Bun.file(imageFilePath).bytes()

  const { text, reasoning, usage } = await generateText({
    model,
    messages: [
      ...messages,
      {
        role: "user",
        content: [
          {
            type: "image",
            image: imageAsUint8Array
          }
        ]
      }
    ]
  })
  return { text, reasoning, usage }
}


/*
 * Share an image (url) and chat about it
 */
export async function chatWithImageURL(messages: CoreMessage[], imageURL: string, model: LanguageModel = groqModel): Promise<AITextResponse> {
  const { text, reasoning, usage } = await generateText({
    model,
    messages: [
      ...messages,
      {
        role: "user",
        content: [
          {
            type: "image",
            image: new URL(imageURL)
          }
        ]
      }
    ]
  })
  return { text, reasoning, usage }
}


/*
 * Share a file (pdf, json, etc) and chat about it
 */
export async function chatWithFile(messages: CoreMessage[], filePath: string, mimeType: string, model: LanguageModel = groqModel): Promise<AITextResponse> {
  const fileAsUint8Array = await Bun.file(filePath).bytes()

  const { text, reasoning, usage } = await generateText({
    model,
    messages: [
      ...messages,
      {
        role: "user",
        content: [
          {
            type: "file",
            data: fileAsUint8Array,
            mimeType
          }
        ]
      }
    ]
  })
  return { text, reasoning, usage }
}


/*
 * Extract text from files (pdf, json, etc)
 */
export async function extractDataFromFile(messages: CoreMessage[], schema: z.ZodSchema, filePath: string, mimeType: string, model: LanguageModel = groqModel): Promise<AIObjectResponse> {
  const fileAsUint8Array = await Bun.file(filePath).bytes()

  const { object, usage } = await generateObject({
    model,
    schema,
    messages: [
      ...messages,
      {
        role: "user",
        content: [
          {
            type: "file",
            data: fileAsUint8Array,
            mimeType
          }
        ]
      }
    ]
  })
  return { object, usage }
}
