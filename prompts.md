# OpenCode AI Implementation Sequence: Interview Copilot

> **Instructions:** Run these prompts in order. Do not skip steps. 
> Wait for the AI to finish each step and verify the file creation before moving to the next.

---

## Step 1: Project Scaffolding
**Goal:** Initialize the VS Code folder structure and dependencies.

**Prompt:**
"Initialize a new project structure for an AI Interview Copilot on Windows. 
1. Create the following folders: `engine/`, `overlay/`, `knowledge/`.
2. Create a `requirements.txt` with: `fastapi`, `uvicorn`, `websockets`, `faster-whisper`, `ollama`, `watchdog`, `pymupdf4llm`, `sentence-transformers`, `faiss-cpu`, `pyaudio`, `markitdown`.
3. Create a `package.json` for Electron with `electron` as a devDependency and a start script `"start": "electron ./"`.
4. Create a `.env` file for API keys (Deepgram, Gemini, Groq)."

---

## Step 2: The Stealth HUD (Frontend)
**Goal:** Build the transparent, click-through overlay.

**Prompt:**
"Build the Electron frontend in the `overlay/` folder.
1. Create `main.js`: Setup a transparent, borderless, always-on-top window. Disable taskbar visibility. Set `setContentProtection(true)` to hide it from screen sharing.
2. Implement a global shortcut `Ctrl+Shift+X` to toggle HUD visibility.
3. Create `index.html` and `style.css`: Use a 'Glassmorphism' HUD design (semi-transparent dark background, neon green text, high contrast).
4. Create `renderer.js`: Setup a WebSocket client to connect to `ws://localhost:8000` and stream text character-by-character into the HUD."

---

## Step 3: Audio & STT Engine
**Goal:** Capture meeting audio and convert to text without delay.

**Prompt:**
"Create `engine/audio_handler.py`. 
1. Use `pyaudio` to capture system audio from the 'VB-Audio Cable' loopback.
2. Use `faster-whisper` (base model, int8) to transcribe audio chunks in real-time.
3. Filter out silence and background noise. 
4. Pass the final transcribed segments to the main orchestrator."

---

## Step 4: Knowledge Base & RAG
**Goal:** Index resumes, code, and PDFs.

**Prompt:**
"Create `engine/document_parser.py`. 
1. Use `pymupdf4llm` to parse PDFs in the `knowledge/` folder.
2. Use `markitdown` to parse Excel/CSV files.
3. Implement a local `FAISS` vector store using `sentence-transformers` (all-MiniLM-L6-v2).
4. Create a function to search this index for relevant context based on a transcribed question."

---

## Step 5: The Orchestrator (Full Integration)
**Goal:** Connect everything and start the service.

**Prompt:**
"Create `engine/app.py`. 
1. Setup a FastAPI server with a WebSocket endpoint on port 8000.
2. When audio is transcribed, use the `document_parser` to find context.
3. Send the transcript + context to **Ollama (Llama 3.1)**.
4. Stream the Ollama tokens back through the WebSocket to the HUD.
5. Add error handling to auto-reconnect if the WebSocket or audio stream drops."

---

## Step 6: Launch Script
**Goal:** One-click execution for Windows.

**Prompt:**
"Create a Windows batch script `run_copilot.bat` that:
1. Checks if Ollama is running (`ollama list`).
2. Starts the Python backend in a hidden terminal.
3. Starts the Electron HUD.
4. Prints a message explaining how to route Zoom/Teams audio to VB-Cable."