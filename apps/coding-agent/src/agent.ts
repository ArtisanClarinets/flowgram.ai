import fs from 'fs';
import path from 'path';
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { Embeddings } from "@langchain/core/embeddings";
import { HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage } from '@langchain/core/messages';
import { DocumentChunk } from './indexer';
import dotenv from 'dotenv';
import { z } from "zod";
import { LLMFactory, LLMConfig } from "./llm-factory";

dotenv.config();

function cosineSimilarity(vecA: number[], vecB: number[]) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Define Tool Interface
interface ToolDef {
    name: string;
    description: string;
    schema: z.ZodObject<any>;
    func: (args: any) => Promise<string>;
}

export class Agent {
    private index: DocumentChunk[] = [];
    private embeddings: Embeddings | null = null;
    private llm: BaseChatModel;
    private config: LLMConfig;

    constructor(indexPath: string, config?: LLMConfig) {
        if (fs.existsSync(indexPath)) {
            try {
                this.index = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
            } catch (e) {
                console.error("Failed to parse index file:", e);
            }
        } else {
            console.warn(`Index file not found at ${indexPath}. Agent will work without RAG.`);
        }

        // Default to OpenAI from env if no config provided
        this.config = config || {
            provider: "openai",
            apiKey: process.env.OPENAI_API_KEY
        };

        try {
            this.llm = LLMFactory.createModel(this.config);
            this.embeddings = LLMFactory.createEmbeddings(this.config);
        } catch (e) {
            console.error("Failed to initialize LLM:", e);
            throw e;
        }
    }

    async retrieveContext(query: string, k: number = 5): Promise<string> {
        if (this.index.length === 0 || !this.embeddings) {
            return "";
        }

        if (!this.index[0].embedding) {
            return "";
        }

        try {
            const queryEmbedding = await this.embeddings.embedQuery(query);

            const scored = this.index.map(chunk => ({
                chunk,
                score: chunk.embedding ? cosineSimilarity(queryEmbedding, chunk.embedding) : -1
            }));

            scored.sort((a, b) => b.score - a.score);

            return scored.slice(0, k).map(s =>
                `File: ${s.chunk.path}\nContent:\n${s.chunk.content}\n`
            ).join("\n---\n");
        } catch (e) {
            console.error("Error retrieving context:", e);
            return "";
        }
    }

    // Exposed for API streaming
    async *runStream(query: string, messageHistory: BaseMessage[] = []): AsyncGenerator<string, void, unknown> {
        // console.log(`User Query: ${query}`);

        const context = await this.retrieveContext(query);
        // if (context) {
        //    console.log(`Retrieved context from ${context.split('File: ').length - 1} chunks.`);
        // }

        const tools: ToolDef[] = [
            {
                name: "read_file",
                description: "Read the content of a file",
                schema: z.object({
                    path: z.string().describe("The path to the file to read")
                }),
                func: async ({ path }: { path: string }) => {
                    try {
                        if (!fs.existsSync(path)) return `File not found: ${path}`;
                        return fs.readFileSync(path, 'utf-8');
                    } catch (e: any) {
                        return `Error reading file: ${e.message}`;
                    }
                }
            },
            {
                name: "write_file",
                description: "Write content to a file. Overwrites existing file.",
                schema: z.object({
                    path: z.string().describe("The path to the file to write"),
                    content: z.string().describe("The content to write")
                }),
                func: async ({ path: filePath, content }: { path: string, content: string }) => {
                    try {
                        const dir = path.dirname(filePath);
                        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
                        fs.writeFileSync(filePath, content);
                        return `Successfully wrote to ${filePath}`;
                    } catch (e: any) {
                        return `Error writing file: ${e.message}`;
                    }
                }
            },
            {
                name: "list_files",
                description: "List files in a directory",
                schema: z.object({
                    path: z.string().describe("The directory path to list")
                }),
                func: async ({ path: dirPath }: { path: string }) => {
                    try {
                         if (!fs.existsSync(dirPath)) return `Directory not found: ${dirPath}`;
                         const files = fs.readdirSync(dirPath);
                         return files.join('\n');
                    } catch (e: any) {
                        return `Error listing files: ${e.message}`;
                    }
                }
            }
        ];

        // Bind tools to the model
        // We use 'any' here because bindTools returns a Runnable which is not exactly BaseChatModel
        // but supports .invoke. For the purpose of this script, we treat it as the runnable to invoke.
        let modelWithTools: any = this.llm;

        // Check if the LLM supports bindTools
        if (typeof this.llm.bindTools === 'function') {
            modelWithTools = this.llm.bindTools(tools.map(t => ({
                name: t.name,
                description: t.description,
                schema: t.schema
            })));
        } else {
            console.warn("This LLM does not support bindTools. Tools will not be available.");
        }

        // Construct initial messages
        const currentMessages: BaseMessage[] = [
             new SystemMessage(`You are a skilled software engineer agent.
            You have access to tools to read, write, and list files.
            You also have context from the codebase provided below.

            When asked to create or modify code, first understand the codebase context, then plan your changes, and finally use the tools to apply them.

            Codebase Context:
            ${context}
            `),
            ...messageHistory,
            new HumanMessage(query)
        ];

        let turns = 0;
        const maxTurns = 15;

        while (turns < maxTurns) {
            try {
                // We use invoke for tool calling loop generally, but if we want to stream the text response to user...
                // Tool calls usually don't need to be streamed to user until they are done?
                // Or we can stream the final response.

                // For simplicity in this iteration:
                // We will stream the FINAL response. Intermediate tool calls we just execute.
                // However, LangChain streaming with tools is tricky.
                // We'll use .invoke() for tool steps, and only stream if the result is a final answer.

                const response = await modelWithTools.invoke(currentMessages);
                currentMessages.push(response);

                if (response.tool_calls && response.tool_calls.length > 0) {
                    // console.log(`Agent wants to use tools: ${response.tool_calls.map(tc => tc.name).join(', ')}`);
                    yield JSON.stringify({ type: 'tool_start', tools: response.tool_calls.map((tc: any) => tc.name) }) + "\n";

                    for (const toolCall of response.tool_calls) {
                        const tool = tools.find(t => t.name === toolCall.name);
                        if (tool) {
                            // console.log(`Executing ${tool.name}...`);
                            const result = await tool.func(toolCall.args);

                            // Send tool output to client if needed?
                            yield JSON.stringify({ type: 'tool_result', tool: tool.name, result: result }) + "\n";

                            currentMessages.push(new ToolMessage({
                                tool_call_id: toolCall.id || "unknown",
                                content: result
                            }));
                        } else {
                            currentMessages.push(new ToolMessage({
                                tool_call_id: toolCall.id || "unknown",
                                content: "Error: Tool not found"
                            }));
                        }
                    }
                } else {
                    // Final response
                    // console.log("Agent Response:", response.content);
                    yield JSON.stringify({ type: 'message', content: response.content }) + "\n";
                    break;
                }
            } catch (e: any) {
                console.error("Error in agent loop:", e);
                yield JSON.stringify({ type: 'error', content: e.message }) + "\n";
                break;
            }
            turns++;
        }
    }

    async run(query: string) {
        // CLI Wrapper around runStream
        for await (const chunk of this.runStream(query)) {
            try {
                const data = JSON.parse(chunk);
                if (data.type === 'message') {
                     console.log(data.content);
                } else if (data.type === 'tool_start') {
                    console.log(`Using tools: ${data.tools.join(', ')}`);
                } else if (data.type === 'tool_result') {
                    console.log(`Tool ${data.tool} finished.`);
                }
            } catch (e) {}
        }
    }
}
