import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Settings, FolderOpen, RefreshCw, FileText, ChevronRight, ChevronDown, Check, Terminal } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { cn } from './lib/utils'

interface Message {
  role: 'user' | 'assistant' | 'tool'
  content: string
  toolName?: string
  isStreaming?: boolean
}

interface FileNode {
  name: string
  path: string
  isDirectory: boolean
}

interface LLMConfig {
  provider: "openai" | "google" | "ollama" | "openrouter" | "lmstudio"
  apiKey?: string
  modelName?: string
  baseUrl?: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isSidebarOpen, setSidebarOpen] = useState(true)
  const [isSettingsOpen, setSettingsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [files, setFiles] = useState<FileNode[]>([])
  const [currentPath, setCurrentPath] = useState<string>('')
  const [config, setConfig] = useState<LLMConfig>({ provider: 'openai' })
  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchConfig()
    fetchFiles()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const fetchConfig = async () => {
    try {
      const res = await fetch('/api/config')
      const data = await res.json()
      setConfig(data)
    } catch (e) {
      console.error(e)
    }
  }

  const fetchFiles = async (path?: string) => {
    try {
      const url = path ? `/api/files?path=${encodeURIComponent(path)}` : '/api/files'
      const res = await fetch(url)
      const data = await res.json()
      setFiles(data.files)
      setCurrentPath(data.currentPath)
    } catch (e) {
      console.error(e)
    }
  }

  const saveConfig = async () => {
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      setSettingsOpen(false)
    } catch (e) {
      console.error(e)
    }
  }

  const reIndex = async () => {
    try {
      setIsLoading(true)
      await fetch('/api/index', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rootDir: currentPath }) })
      alert("Indexing Complete")
    } catch (e) {
      console.error(e)
      alert("Indexing Failed")
    } finally {
      setIsLoading(false)
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMsg: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg.content, history: [] }) // Pass simplified history if needed
      })

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) return

      let assistantMsg: Message = { role: 'assistant', content: '', isStreaming: true }
      setMessages(prev => [...prev, assistantMsg])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n').filter(line => line.trim() !== '')

        for (const line of lines) {
          try {
            const data = JSON.parse(line)
            if (data.type === 'message') {
                setMessages(prev => {
                    const newMsgs = [...prev]
                    const lastMsg = newMsgs[newMsgs.length - 1]
                    if (lastMsg.role === 'assistant' && !lastMsg.toolName) {
                        lastMsg.content += data.content
                    }
                    return newMsgs
                })
            } else if (data.type === 'tool_start') {
                 setMessages(prev => [...prev, { role: 'tool', content: `Using tools: ${data.tools.join(', ')}`, toolName: 'System', isStreaming: false }])
            } else if (data.type === 'tool_result') {
                 setMessages(prev => [...prev, { role: 'tool', content: `Tool ${data.tool} result:\n${data.result.slice(0, 200)}...`, toolName: data.tool, isStreaming: false }])
                 // Prepare for next assistant message
                 setMessages(prev => [...prev, { role: 'assistant', content: '', isStreaming: true }])
            }
          } catch (e) {
             // console.warn("Failed to parse chunk:", line)
             // Sometimes chunks are concatenated JSONs
          }
        }
      }

      setMessages(prev => {
         const newMsgs = [...prev]
         const lastMsg = newMsgs[newMsgs.length - 1]
         lastMsg.isStreaming = false
         return newMsgs
      })

    } catch (e) {
      console.error(e)
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: Failed to send message' }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-screen bg-white text-gray-900 font-sans overflow-hidden">
      {/* Sidebar */}
      <motion.div
        initial={{ width: 0, opacity: 0 }}
        animate={{ width: isSidebarOpen ? 280 : 0, opacity: isSidebarOpen ? 1 : 0 }}
        className="h-full border-r border-gray-200 bg-gray-50 flex flex-col"
      >
        <div className="p-4 border-b border-gray-200 flex justify-between items-center">
            <h2 className="font-semibold text-gray-700">Project Files</h2>
            <button onClick={reIndex} className="p-1 hover:bg-gray-200 rounded" title="Re-index Codebase">
                <RefreshCw size={16} className={isLoading ? "animate-spin" : ""} />
            </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
            <div className="text-xs text-gray-500 mb-2 px-2 truncate" title={currentPath}>{currentPath}</div>
            {files.map((file, idx) => (
                <div
                    key={idx}
                    className="flex items-center gap-2 p-2 hover:bg-gray-200 rounded cursor-pointer text-sm"
                    onClick={() => file.isDirectory ? fetchFiles(file.path) : null}
                >
                    {file.isDirectory ? <FolderOpen size={16} className="text-blue-500" /> : <FileText size={16} className="text-gray-500" />}
                    <span className="truncate">{file.name}</span>
                </div>
            ))}
            {currentPath !== '/' && (
                <div
                    className="flex items-center gap-2 p-2 hover:bg-gray-200 rounded cursor-pointer text-sm text-gray-500"
                    onClick={() => fetchFiles(currentPath.split('/').slice(0, -1).join('/') || '/')}
                >
                    <ChevronDown size={16} />
                    <span>..</span>
                </div>
            )}
        </div>
        <div className="p-4 border-t border-gray-200">
            <button
                onClick={() => setSettingsOpen(true)}
                className="flex items-center gap-2 w-full p-2 hover:bg-gray-200 rounded text-sm font-medium"
            >
                <Settings size={18} />
                <span>Settings</span>
            </button>
        </div>
      </motion.div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full relative">
        <header className="h-14 border-b border-gray-200 flex items-center px-4 bg-white/80 backdrop-blur-md z-10 sticky top-0">
             <button onClick={() => setSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-gray-100 rounded mr-4">
                {isSidebarOpen ? <ChevronDown className="rotate-90" /> : <ChevronRight />}
             </button>
             <h1 className="font-semibold text-lg text-gray-800">Coding Agent</h1>
             <div className="ml-auto flex items-center gap-2">
                <span className="text-xs px-2 py-1 bg-gray-100 rounded-full text-gray-600 font-medium uppercase tracking-wider">
                    {config.provider}
                </span>
             </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {messages.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-gray-400">
                    <Terminal size={64} className="mb-4 opacity-50" />
                    <p className="text-xl font-medium">How can I help you code today?</p>
                </div>
            )}
            {messages.map((msg, idx) => (
                <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={cn(
                        "max-w-3xl mx-auto flex gap-4",
                        msg.role === 'user' ? "flex-row-reverse" : ""
                    )}
                >
                    <div className={cn(
                        "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                        msg.role === 'user' ? "bg-black text-white" : msg.role === 'tool' ? "bg-amber-100 text-amber-600" : "bg-blue-600 text-white"
                    )}>
                        {msg.role === 'user' ? "U" : msg.role === 'tool' ? "T" : "AI"}
                    </div>
                    <div className={cn(
                        "flex-1 p-4 rounded-2xl shadow-sm text-sm leading-relaxed",
                        msg.role === 'user' ? "bg-gray-100 rounded-tr-sm" : msg.role === 'tool' ? "bg-amber-50 border border-amber-100 font-mono text-xs text-amber-800" : "bg-white border border-gray-100 rounded-tl-sm"
                    )}>
                        {msg.role === 'tool' ? (
                            <pre className="whitespace-pre-wrap">{msg.content}</pre>
                        ) : (
                            <ReactMarkdown
                                components={{
                                    code({node, className, children, ...props}) {
                                        return (
                                            <code className={cn("bg-gray-100 px-1 py-0.5 rounded text-red-500 font-mono text-xs", className)} {...props}>
                                                {children}
                                            </code>
                                        )
                                    },
                                    pre({children}) {
                                         return <pre className="bg-gray-900 text-gray-100 p-3 rounded-lg overflow-x-auto my-2">{children}</pre>
                                    }
                                }}
                            >
                                {msg.content}
                            </ReactMarkdown>
                        )}
                        {msg.isStreaming && <span className="inline-block w-2 h-4 bg-blue-500 ml-1 animate-pulse"/>}
                    </div>
                </motion.div>
            ))}
            <div ref={chatEndRef} />
        </div>

        <div className="p-4 bg-white border-t border-gray-200">
            <div className="max-w-3xl mx-auto relative">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            sendMessage()
                        }
                    }}
                    placeholder="Ask anything about your codebase..."
                    className="w-full min-h-[50px] max-h-[200px] p-4 pr-12 rounded-xl border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none shadow-sm"
                />
                <button
                    onClick={sendMessage}
                    disabled={!input.trim() || isLoading}
                    className="absolute right-3 bottom-3 p-2 bg-black text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    <Send size={16} />
                </button>
            </div>
        </div>
      </div>

      {/* Settings Modal */}
      <AnimatePresence>
        {isSettingsOpen && (
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 flex items-center justify-center"
                onClick={() => setSettingsOpen(false)}
            >
                <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.95, opacity: 0 }}
                    className="bg-white rounded-2xl shadow-xl w-[480px] p-6"
                    onClick={e => e.stopPropagation()}
                >
                    <h2 className="text-xl font-semibold mb-6">Settings</h2>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium mb-1">LLM Provider</label>
                            <select
                                value={config.provider}
                                onChange={(e) => setConfig({...config, provider: e.target.value as any})}
                                className="w-full p-2 border border-gray-300 rounded-lg outline-none focus:border-blue-500"
                            >
                                <option value="openai">OpenAI</option>
                                <option value="google">Google Gemini</option>
                                <option value="ollama">Ollama (Local)</option>
                                <option value="openrouter">OpenRouter</option>
                                <option value="lmstudio">LM Studio</option>
                            </select>
                        </div>

                        {config.provider !== 'ollama' && (
                            <div>
                                <label className="block text-sm font-medium mb-1">API Key</label>
                                <input
                                    type="password"
                                    value={config.apiKey || ''}
                                    onChange={(e) => setConfig({...config, apiKey: e.target.value})}
                                    placeholder="sk-..."
                                    className="w-full p-2 border border-gray-300 rounded-lg outline-none focus:border-blue-500"
                                />
                            </div>
                        )}

                        <div>
                            <label className="block text-sm font-medium mb-1">Model Name</label>
                            <input
                                type="text"
                                value={config.modelName || ''}
                                onChange={(e) => setConfig({...config, modelName: e.target.value})}
                                placeholder={config.provider === 'openai' ? 'gpt-4o' : config.provider === 'google' ? 'gemini-pro' : 'llama3'}
                                className="w-full p-2 border border-gray-300 rounded-lg outline-none focus:border-blue-500"
                            />
                        </div>

                        {(config.provider === 'ollama' || config.provider === 'openrouter' || config.provider === 'lmstudio') && (
                            <div>
                                <label className="block text-sm font-medium mb-1">Base URL</label>
                                <input
                                    type="text"
                                    value={config.baseUrl || ''}
                                    onChange={(e) => setConfig({...config, baseUrl: e.target.value})}
                                    placeholder="http://localhost:11434"
                                    className="w-full p-2 border border-gray-300 rounded-lg outline-none focus:border-blue-500"
                                />
                            </div>
                        )}
                    </div>

                    <div className="mt-8 flex justify-end gap-3">
                        <button
                            onClick={() => setSettingsOpen(false)}
                            className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={saveConfig}
                            className="px-4 py-2 bg-black text-white rounded-lg hover:bg-gray-800 flex items-center gap-2"
                        >
                            <Check size={16} />
                            Save Changes
                        </button>
                    </div>
                </motion.div>
            </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
