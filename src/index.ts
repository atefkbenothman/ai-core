import * as fs from "fs"
import { z } from "zod"
import { createGroq } from "@ai-sdk/groq"
import {
  generateText,
  extractReasoningMiddleware,
  wrapLanguageModel,
  streamText,
  smoothStream,
  generateObject,
  streamObject,
  type LanguageModel,
  type LanguageModelUsage,
  type CoreMessage
} from "ai"


export type AITextResponse = {
  success: boolean
  text?: string
  reasoning?: string | undefined
  usage?: LanguageModelUsage
  error?: string
}

export type AITextStreamResponse = {
  success: boolean
  textStream?: AsyncIterable<string>
  reasoning?: Promise<string | undefined>
  error?: string
}

export type AIObjectResponse = {
  success: boolean
  object?: any
  usage?: LanguageModelUsage
  error?: string
}

export type AIObjectStreamResponse = {
  success: boolean
  partialObjectStream?: any
  error?: string
}


export class AI {
  private provider: string
  private apiKey: string
  private modelName: string
  private model: LanguageModel | undefined

  constructor(provider: string, modelName: string, apiKey: string) {
    this.provider = provider
    this.apiKey = apiKey
    this.modelName = modelName
    this.model = undefined
    this.initializeAIModel(provider, modelName, apiKey)
  }

  private initializeAIModel(provider: string, modelName: string, apiKey: string) {
    switch (provider) {
      case "groq":
        const groq = createGroq({ apiKey })
        this.model = wrapLanguageModel({
          model: groq(modelName),
          middleware: extractReasoningMiddleware({ tagName: "think" })
        })
    }
  }

  /* chat with an AI */
  async chat(messages: CoreMessage[]): Promise<AITextResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const { text, reasoning, usage } = await generateText({
        model: this.model,
        messages,
      })
      return {
        success: true,
        text,
        reasoning,
        usage
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* chat with an AI (stream) */
  async streamChat(messages: CoreMessage[]): Promise<AITextStreamResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const { textStream, reasoning } = await streamText({
        model: this.model,
        messages,
        experimental_transform: smoothStream()
      })
      // we can get the final text block by awaiting
      // const finalText = await textStream 
      return {
        success: true,
        textStream,
        reasoning,
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* generate structured output */
  async createObject(messages: CoreMessage[], schema: z.ZodSchema): Promise<AIObjectResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const { object, usage } = await generateObject({
        model: this.model,
        messages,
        schema
      })
      return {
        success: true,
        object,
        usage,
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* generate structured output (stream) */
  async streamCreateObject(messages: CoreMessage[], schema: z.ZodSchema): Promise<AIObjectStreamResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const { partialObjectStream } = await streamObject({
        model: this.model,
        messages,
        schema,
      })
      // we can get the final object by awaiting
      // const finalObject = await partialObjectStream
      return {
        success: true,
        partialObjectStream
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* classify text into a category */
  async classifyText(messages: CoreMessage[], categories: string[]): Promise<AIObjectResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const { object, usage } = await generateObject({
        model: this.model,
        messages,
        output: "enum",
        enum: categories
      })
      return {
        success: true,
        object,
        usage
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* share an image (file path) and chat about it */
  async chatWithImageFilePath(messages: CoreMessage[], imageFilePath: string): Promise<AITextResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const imageAsUint8Array = await fs.readFileSync(imageFilePath)
      const { text, reasoning, usage } = await generateText({
        model: this.model,
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
      return {
        success: true,
        text,
        reasoning,
        usage
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* share an image (url) and chat about it */
  async chatWithImageURL(messages: CoreMessage[], imageURL: string): Promise<AITextResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const { text, reasoning, usage } = await generateText({
        model: this.model,
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
      return {
        success: true,
        text,
        reasoning,
        usage
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* share a file (pdf, json, etc) and chat about it */
  async chatWithFile(messages: CoreMessage[], filePath: string, mimeType: string): Promise<AITextResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const fileAsUint8Array = await fs.readFileSync(filePath)
      const { text, reasoning, usage } = await generateText({
        model: this.model,
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
      return {
        success: true,
        text,
        reasoning,
        usage
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }

  /* extract text from files (pdf, json, etc) */
  async extractDataFromFile(messages: CoreMessage[], schema: z.ZodSchema, filePath: string, mimeType: string): Promise<AIObjectResponse> {
    if (!this.model) {
      return {
        success: false,
        error: "ai model not set"
      }
    }
    try {
      const fileAsUint8Array = await fs.readFileSync(filePath)
      const { object, usage } = await generateObject({
        model: this.model,
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
      return {
        success: true,
        object,
        usage
      }
    } catch (err) {
      console.error(err)
      return {
        success: false,
        error: String(err)
      }
    }
  }
}
