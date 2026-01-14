import { ChatOpenAI } from "@langchain/openai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOllama } from "@langchain/ollama";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { Embeddings } from "@langchain/core/embeddings";
import { OpenAIEmbeddings } from "@langchain/openai";
import { OllamaEmbeddings } from "@langchain/ollama";

export type LLMProvider = "openai" | "google" | "ollama" | "openrouter" | "lmstudio";

export interface LLMConfig {
    provider: LLMProvider;
    apiKey?: string;
    modelName?: string;
    baseUrl?: string;
}

export class LLMFactory {
    static createModel(config: LLMConfig): BaseChatModel {
        switch (config.provider) {
            case "openai":
                return new ChatOpenAI({
                    openAIApiKey: config.apiKey,
                    modelName: config.modelName || "gpt-4o",
                    temperature: 0,
                }) as unknown as BaseChatModel;
            case "google":
                return new ChatGoogleGenerativeAI({
                    apiKey: config.apiKey,
                    modelName: config.modelName || "gemini-pro",
                    temperature: 0,
                }) as unknown as BaseChatModel;
            case "ollama":
                return new ChatOllama({
                    baseUrl: config.baseUrl || "http://localhost:11434",
                    model: config.modelName || "llama3",
                    temperature: 0,
                }) as unknown as BaseChatModel;
            case "openrouter":
            case "lmstudio":
                // Both work with OpenAI-compatible endpoint
                return new ChatOpenAI({
                    openAIApiKey: config.apiKey || "not-needed", // LM Studio might not need it
                    configuration: {
                        baseURL: config.baseUrl // e.g., http://localhost:1234/v1 for LM Studio
                    },
                    modelName: config.modelName || "local-model",
                    temperature: 0,
                }) as unknown as BaseChatModel;
            default:
                throw new Error(`Unsupported provider: ${config.provider}`);
        }
    }

    static createEmbeddings(config: LLMConfig): Embeddings {
        // Simple mapping: if ollama is used for chat, use ollama for embeddings by default
        // otherwise default to OpenAI, unless specific overrides are handled later.
        if (config.provider === 'ollama') {
             return new OllamaEmbeddings({
                 baseUrl: config.baseUrl || "http://localhost:11434",
                 model: "nomic-embed-text", // Good default for ollama
             });
        }

        // Google GenAI embeddings not yet fully standard in LangChain JS interface used here,
        // fallback to OpenAI or expect OpenAI key for embeddings even if using Gemini for Chat?
        // Ideally we should allow separate config for embeddings.
        // For now, let's assume OpenAI embeddings for robustness unless Ollama.

        return new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_API_KEY, // Fallback to env
            modelName: "text-embedding-3-small"
        });
    }
}
