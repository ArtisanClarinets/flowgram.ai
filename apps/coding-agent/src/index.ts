import { Command } from 'commander';
import path from 'path';
import { indexCodebase } from './indexer';
import { Agent } from './agent';
import dotenv from 'dotenv';

dotenv.config();

const program = new Command();

program
  .name('coding-agent')
  .description('AI Coding Agent and RAG Indexer')
  .version('0.1.0');

program.command('index')
  .description('Index the current codebase')
  .option('-r, --root <path>', 'Root directory of the codebase', process.cwd())
  .option('-o, --output <path>', 'Output file for the index', '.flowgram-agent/index.json')
  .action(async (options) => {
    const rootDir = path.resolve(options.root);
    const outputFile = path.resolve(options.output);
    console.log(`Indexing codebase at ${rootDir}...`);
    try {
        await indexCodebase(rootDir, outputFile);
    } catch (error) {
        console.error("Indexing failed:", error);
        process.exit(1);
    }
  });

program.command('run')
  .description('Run the coding agent')
  .argument('<query>', 'Query for the agent')
  .option('-i, --index <path>', 'Path to the index file', '.flowgram-agent/index.json')
  .action(async (query, options) => {
    const indexPath = path.resolve(options.index);
    try {
        const agent = new Agent(indexPath);
        await agent.run(query);
    } catch (error) {
        console.error("Agent execution failed:", error);
        process.exit(1);
    }
  });

program.parse();
