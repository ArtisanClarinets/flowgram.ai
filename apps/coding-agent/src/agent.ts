import fs from 'fs';
import path from 'path';
import { ChatOpenAI } from '@langchain/openai';
import { OpenAIEmbeddings } from '@langchain/openai';
import { HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage } from '@langchain/core/messages';
import { DocumentChunk } from './indexer';
import dotenv from 'dotenv';
import { z } from "zod";

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
    private embeddings: OpenAIEmbeddings | null = null;
    private llm: ChatOpenAI;

    constructor(indexPath: string) {
        if (fs.existsSync(indexPath)) {
            try {
                this.index = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
            } catch (e) {
                console.error("Failed to parse index file:", e);
            }
        } else {
            console.warn(`Index file not found at ${indexPath}. Agent will work without RAG.`);
        }

        if (process.env.OPENAI_API_KEY) {
            this.embeddings = new OpenAIEmbeddings({
                 modelName: "text-embedding-3-small"
            });

            this.llm = new ChatOpenAI({
                modelName: "gpt-4o",
                temperature: 0
            });
        } else {
            console.error("OPENAI_API_KEY is missing. Agent cannot function.");
            process.exit(1);
        }
    }

    async retrieveContext(query: string, k: number = 5): Promise<string> {
        if (this.index.length === 0 || !this.embeddings) {
            return "";
        }

        // If index doesn't have embeddings (e.g. key was missing during index), fallback to text search?
        // For now, assume embeddings exist if we are here.
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

    async run(query: string) {
        console.log(`User Query: ${query}`);

        const context = await this.retrieveContext(query);
        if (context) {
            console.log(`Retrieved context from ${context.split('File: ').length - 1} chunks.`);
        }

        const tools: ToolDef[] = [
            {
                name: "read_file",
                description: "Read the content of a file",
                schema: z.object({
                    path: z.string().describe("The path to the file to read")
                }),
                func: async ({ path }: { path: string }) => {
                    try {
                        // Security check: prevent reading outside cwd? For now, allow it as it is a dev tool.
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
        const modelWithTools = this.llm.bindTools(tools.map(t => ({
            name: t.name,
            description: t.description,
            schema: t.schema
        })));

        const messages: BaseMessage[] = [
            new SystemMessage(`You are a skilled software engineer agent.
            You have access to tools to read, write, and list files.
            You also have context from the codebase provided below.

            When asked to create or modify code, first understand the codebase context, then plan your changes, and finally use the tools to apply them.

            Codebase Context:
            ${context}
            `)
        ];

        messages.push(new HumanMessage(query));

        let turns = 0;
        const maxTurns = 15; // Allow more turns for complex tasks

        while (turns < maxTurns) {
            try {
                const response = await modelWithTools.invoke(messages);
                messages.push(response);

                if (response.tool_calls && response.tool_calls.length > 0) {
                    console.log(`Agent wants to use tools: ${response.tool_calls.map(tc => tc.name).join(', ')}`);

                    for (const toolCall of response.tool_calls) {
                        const tool = tools.find(t => t.name === toolCall.name);
                        if (tool) {
                            console.log(`Executing ${tool.name}...`);
                            const result = await tool.func(toolCall.args);
                            // console.log(`Tool result: ${result.slice(0, 100)}...`);
                            messages.push(new ToolMessage({
                                tool_call_id: toolCall.id || "unknown",
                                content: result
                            }));
                        } else {
                            messages.push(new ToolMessage({
                                tool_call_id: toolCall.id || "unknown",
                                content: "Error: Tool not found"
                            }));
                        }
                    }
                } else {
                    // Final response (or just a message without tool calls)
                    console.log("Agent Response:", response.content);
                    break;
                }
            } catch (e) {
                console.error("Error in agent loop:", e);
                break;
            }
            turns++;
        }
    }
}
