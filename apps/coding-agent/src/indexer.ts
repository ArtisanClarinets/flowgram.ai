import fs from 'fs';
import path from 'path';
import glob from 'fast-glob';
import { Embeddings } from '@langchain/core/embeddings';
import { OpenAIEmbeddings } from '@langchain/openai';
import dotenv from 'dotenv';

dotenv.config();

// Simple implementation of RecursiveCharacterTextSplitter since we don't have the package
class SimpleTextSplitter {
    chunkSize: number;
    chunkOverlap: number;

    constructor(chunkSize: number = 1000, chunkOverlap: number = 200) {
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
    }

    splitText(text: string): string[] {
        if (!text) return [];
        const chunks: string[] = [];
        let startIndex = 0;

        while (startIndex < text.length) {
            let endIndex = startIndex + this.chunkSize;

            if (endIndex >= text.length) {
                chunks.push(text.slice(startIndex));
                break;
            }

            // Try to find a good break point (look back from expected end)
            const slice = text.slice(startIndex, endIndex);

            // Priority of separators
            const separators = ['\n\n', '\n', ' '];
            let splitIndex = -1;

            for (const sep of separators) {
                const lastSepIndex = slice.lastIndexOf(sep);
                if (lastSepIndex !== -1) {
                    // Check if the split is too close to the start (avoid tiny chunks)
                    // unless the chunk itself is small (e.g. at end)
                    if (lastSepIndex > this.chunkOverlap / 2) {
                        splitIndex = lastSepIndex + sep.length; // Include separator in previous chunk? or after?
                        // Usually recursive splitter drops the separator or keeps it attached.
                        // For simplicity, let's just split after.
                        break;
                    }
                }
            }

            if (splitIndex !== -1) {
                endIndex = startIndex + splitIndex;
            }

            chunks.push(text.slice(startIndex, endIndex));

            // Move start index for next chunk, considering overlap
            const nextStartIndex = endIndex - this.chunkOverlap;

            // Ensure we always move forward at least by 1 character (or more reasonably, avoids getting stuck)
            // If the chunk was smaller than overlap (e.g. cut short), nextStartIndex might be <= startIndex.
            // We must advance.
            startIndex = Math.max(startIndex + 1, nextStartIndex);

            if (startIndex < 0) startIndex = 0; // Should not happen with max(startIndex+1, ...)
        }
        return chunks;
    }
}

export interface DocumentChunk {
    path: string;
    content: string;
    embedding?: number[];
}

export async function indexCodebase(
    rootDir: string,
    outputFile: string,
    embeddings?: Embeddings,
    globPattern: string[] = ['**/*.{ts,tsx,js,jsx,py,json,md}'],
    exclude: string[] = ['**/node_modules/**', '**/dist/**', '**/.git/**', '**/common/temp/**']
) {
    console.log(`Scanning ${rootDir} for files...`);

    // Convert glob patterns to fit fast-glob expectations (it handles ignore separately)
    const files = await glob(globPattern, {
        cwd: rootDir,
        ignore: exclude,
        absolute: true,
        onlyFiles: true
    });

    console.log(`Found ${files.length} files.`);

    const splitter = new SimpleTextSplitter(2000, 200);
    const chunks: DocumentChunk[] = [];

    for (const file of files) {
        try {
            // Skip large files to avoid memory issues/token limits
            const stats = fs.statSync(file);
            if (stats.size > 100 * 1024) { // > 100KB
                console.warn(`Skipping large file: ${file}`);
                continue;
            }

            const content = fs.readFileSync(file, 'utf-8');
            const fileChunks = splitter.splitText(content);

            for (const chunkContent of fileChunks) {
                chunks.push({
                    path: path.relative(rootDir, file),
                    content: chunkContent
                });
            }
        } catch (e) {
            console.warn(`Failed to read/split file ${file}:`, e);
        }
    }

    console.log(`Generated ${chunks.length} chunks. Generating embeddings...`);

    if (!embeddings && !process.env.OPENAI_API_KEY) {
        console.warn("No embeddings model provided and OPENAI_API_KEY not found. Skipping embedding generation. Index will be text-only.");
        // We still save the chunks, so at least we have the text.
    } else {
        const embedder = embeddings || new OpenAIEmbeddings({
            modelName: "text-embedding-3-small"
        });

        // Batch processing to avoid rate limits or huge payloads
        const batchSize = 50;
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            try {
                const vectors = await embedder.embedDocuments(batch.map(c => c.content));
                batch.forEach((chunk, idx) => {
                    chunk.embedding = vectors[idx];
                });
                console.log(`Processed batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(chunks.length / batchSize)}`);
            } catch (e) {
                console.error(`Error embedding batch ${i}:`, e);
            }
        }
    }

    const outputDir = path.dirname(outputFile);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    fs.writeFileSync(outputFile, JSON.stringify(chunks, null, 2));
    console.log(`Index saved to ${outputFile}`);
}
