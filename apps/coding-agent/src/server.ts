import express from 'express';
import cors from 'cors';
import { Agent } from './agent';
import { LLMConfig, LLMFactory } from './llm-factory';
import path from 'path';
import fs from 'fs';
import { indexCodebase } from './indexer';

const app = express();
// It's important to restrict CORS to allow only trusted origins.
// Replace 'https://example.com' with your actual frontend origin(s). 
// You can also use an environment variable for flexibility in different environments.
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',')
  : ['https://example.com'];

app.use(cors({
    origin: allowedOrigins,
}));
app.use(express.json());

const PORT = 3001;
const CONFIG_DIR = path.resolve('.flowgram-agent');
const DEFAULT_INDEX_PATH = path.join(CONFIG_DIR, 'index.json');
const CONFIG_PATH = path.join(CONFIG_DIR, 'config.json');
const DEFAULT_ROOT_DIR = process.cwd();

let currentAgent: Agent | null = null;
let currentConfig: LLMConfig = { provider: 'openai' }; // Default

// Ensure config dir exists
if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
}

// Load config if exists
if (fs.existsSync(CONFIG_PATH)) {
    try {
        currentConfig = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
    } catch (e) {
        console.warn("Failed to load config:", e);
    }
}

// Initialize agent if index exists
if (fs.existsSync(DEFAULT_INDEX_PATH)) {
    try {
        currentAgent = new Agent(DEFAULT_INDEX_PATH, currentConfig);
        console.log("Agent initialized with config from disk.");
    } catch (e) {
        console.warn("Could not initialize agent on startup:", e);
    }
}

app.post('/api/config', (req, res) => {
    try {
        const config = req.body as LLMConfig;
        currentConfig = config;

        // Save to disk
        fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));

        // Re-init agent
        currentAgent = new Agent(DEFAULT_INDEX_PATH, currentConfig);
        res.json({ success: true, message: "Configuration updated" });
    } catch (e: any) {
        res.status(500).json({ success: false, message: e.message });
    }
});

app.get('/api/config', (req, res) => {
    res.json(currentConfig);
});

app.post('/api/index', async (req, res) => {
    try {
        const { rootDir = DEFAULT_ROOT_DIR } = req.body;
        console.log(`Indexing ${rootDir}...`);

        const embeddings = LLMFactory.createEmbeddings(currentConfig);
        await indexCodebase(rootDir, DEFAULT_INDEX_PATH, embeddings);

        // Reload agent
        currentAgent = new Agent(DEFAULT_INDEX_PATH, currentConfig);

        res.json({ success: true, message: "Indexing complete" });
    } catch (e: any) {
        console.error(e);
        res.status(500).json({ success: false, message: e.message });
    }
});

app.post('/api/chat', async (req, res) => {
    if (!currentAgent) {
        // Try to init if not ready (e.g. if index was missing but now exists)
        if (fs.existsSync(DEFAULT_INDEX_PATH)) {
             currentAgent = new Agent(DEFAULT_INDEX_PATH, currentConfig);
        } else {
             return res.status(400).json({ message: "Agent not initialized. Please run indexing first." });
        }
    }

    const { query, history } = req.body;

    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Transfer-Encoding', 'chunked');

    try {
        // history is passed as plain objects, need to be converted to LangChain messages?
        // For simplicity, agent implementation above handles history in its runStream but currently we passed []
        // We should update Agent.runStream to accept history. (Done in previous step)

        // Convert simplified history to LangChain format if needed, or pass as is if compatible.
        // But runStream expects BaseMessage[].
        // For now, let's just pass the query. Implementing full history reconstruction is a bit more involved
        // (mapping roles to HumanMessage/AIMessage).

        for await (const chunk of currentAgent.runStream(query, [])) {
            res.write(chunk);
        }
        res.end();
    } catch (e: any) {
        console.error(e);
        res.write(JSON.stringify({ type: 'error', content: e.message }));
        res.end();
    }
});

app.get('/api/files', (req, res) => {
    const dir = req.query.path as string || DEFAULT_ROOT_DIR;
    try {
// Resolve user-supplied dir relative to DEFAULT_ROOT_DIR and block access if outside
const rawPath = req.query.path as string || "";
const dir = path.resolve(DEFAULT_ROOT_DIR, rawPath);
if (!dir.startsWith(DEFAULT_ROOT_DIR)) {
    return res.status(403).json({ message: "Access denied" });
}
if (!fs.existsSync(dir)) return res.status(404).json({ message: "Path not found" });

const files = fs.readdirSync(dir, { withFileTypes: true }).map(dirent => ({
    name: dirent.name,
    isDirectory: dirent.isDirectory(),
    path: path.join(dir, dirent.name)
}));

res.json({ files, currentPath: dir });

        const files = fs.readdirSync(dir, { withFileTypes: true }).map(dirent => ({
            name: dirent.name,
            isDirectory: dirent.isDirectory(),
// Ensure returned path is always within DEFAULT_ROOT_DIR
// Resolve dirent path and verify it starts with DEFAULT_ROOT_DIR
// If it doesn't, do not include it (prevent path traversal listing)
// Otherwise, include the securely resolved path
path: (() => {
    const reqPath = typeof req.query.path === 'string' ? req.query.path : '';
    const baseDir = path.resolve(DEFAULT_ROOT_DIR, reqPath);
    const entryPath = path.resolve(baseDir, dirent.name);
    if (!entryPath.startsWith(DEFAULT_ROOT_DIR)) {
        // If path traversal attempt, do not reveal actual path
        return null;
    }
    return entryPath;
})()
        }));

        res.json({ files, currentPath: dir });
    } catch (e: any) {
        res.status(500).json({ message: e.message });
    }
});

app.get('/api/file', (req, res) => {
    const filePath = req.query.path as string;
    if (!filePath) return res.status(400).json({ message: "Path required" });
    try {
        const content = fs.readFileSync(filePath, 'utf-8');
        res.json({ content });
    } catch (e: any) {
        res.status(500).json({ message: e.message });
    }
});

app.listen(PORT, () => {
    console.log(`Coding Agent Server running on http://localhost:${PORT}`);
});
