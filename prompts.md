# OpenCode AI Implementation Sequence: Interview Copilot

> **Instructions:** Run these prompts in order. Do not skip steps. 
> Wait for the AI to finish each step and verify the file creation before moving to the next.

---

## Step 1: Project Scaffolding
**Goal:** Initialize the VS Code folder structure and dependencies.

**Prompt:**
"Initialize a new project structure for an AI Interview Copilot on Windows. 
1. Create the following folders: `engine/`, `overlay/`, `knowledge/`, `corebrum-extension/`.
2. Create a `requirements.txt` with: `fastapi`, `uvicorn`, `websockets`, `faster-whisper`, `ollama`, `watchdog`, `pymupdf4llm`, `sentence-transformers`, `faiss-cpu`, `markitdown`.
3. Create a `package.json` for Electron with `electron` as a devDependency and a start script `"start": "electron ./"`.
4. Create a `.env` file for configuration."

---

## Step 2: The Stealth HUD (Frontend)
**Goal:** Build the transparent, click-through overlay.

**Prompt:**
"Build the Electron frontend in the `overlay/` folder.
1. Create `main.js`: Setup a transparent, borderless, always-on-top window. Disable taskbar visibility. Set `setContentProtection(true)` to hide it from screen sharing.
2. Implement a global shortcut `Ctrl+Shift+X` to toggle HUD visibility.
3. Create `index.html` and `style.css`: Use a 'Glassmorphism' HUD design (semi-transparent dark background, neon green text, high contrast).
4. Create `renderer.js`: Setup a WebSocket client to connect to `ws://localhost:8001/ws/ui` and stream text character-by-character into the HUD."

---

## Step 3: Chrome Extension (Browser Tab Capture)
**Goal:** Capture audio directly from browser tabs using the tabCapture API.

**Prompt:**
"Create a Chrome Extension (Manifest V3) in `corebrum-extension/`:
1. `manifest.json`: Set permissions for `tabCapture`, `tabs`, `offscreen`. Add host permissions for `ws://localhost:8001/*`.
2. `background.js`: Listen for messages from popup, use `chrome.tabCapture.getMediaStreamId()` to capture the active tab's audio, create an offscreen document for processing.
3. `offscreen.js`: Process audio using ScriptProcessorNode, convert to PCM16 @ 16kHz, send via WebSocket to `ws://localhost:8001/ws/audio`.
4. `popup.html/js`: Simple UI with a Start/Stop button.
5. `offscreen.html`: Minimal HTML to load offscreen.js."

---

## Step 4: Audio & STT Engine
**Goal:** Receive browser audio and transcribe to text using Whisper.

**Prompt:**
"Create `engine/audio_handler.py`. 
1. Accept audio chunks from WebSocket (PCM16 @ 16kHz).
2. Use `faster-whisper` (tiny.en model, int8) to transcribe audio chunks in real-time.
3. Implement voice activity detection (VAD) to filter silence.
4. Implement a prebuffer for smooth speech detection.
5. Pass transcribed segments to the orchestrator."

---

## Step 5: Knowledge Base & RAG
**Goal:** Index resumes, code, and PDFs for context retrieval.

**Prompt:**
"Create `engine/document_parser.py`. 
1. Use `pymupdf4llm` to parse PDFs in the `knowledge/` folder.
2. Use `markitdown` to parse Excel/CSV/DOCX files.
3. Implement a local `FAISS` vector store using `sentence-transformers` (all-MiniLM-L6-v2).
4. Create a function to search this index for relevant context based on a transcribed question."

---

## Step 6: The Orchestrator (Full Integration)
**Goal:** Connect everything and start the service.

**Prompt:**
"Create `engine/app.py`. 
1. Setup a FastAPI server with WebSocket endpoints:
   - `/ws/ui` - HUD display connection
   - `/ws/audio` - Browser extension audio ingestion
2. When audio is transcribed, use the `document_parser` to find relevant context.
3. Send the transcript + context to **Ollama (Llama 3.2)**.
4. Stream the Ollama tokens back through the WebSocket to the HUD.
5. Add error handling to auto-reconnect if the WebSocket or audio stream drops."

---

## Step 7: Launch Script
**Goal:** One-click execution for Windows.

**Prompt:**
"Create a Windows batch script `run_copilot.bat` that:
1. Checks if Ollama is running (`ollama list`) and the model is available.
2. Starts the Python backend on port 8001.
3. Starts the Electron HUD.
4. Prints instructions to install the Chrome extension and use it."