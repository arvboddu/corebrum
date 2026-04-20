# Corebrum HUD Merged — Challenge Analysis & Fix Guide

> **File audited:** `corebrum-hud-merged.html`
> **Total issues found:** 10
> **Estimated fix time:** ~2 hours total
> **Fix before first live session:** Issues 1, 2, 3 (Critical)
> **Status:** All 10 issues FIXED and committed in `72b14ec` and `437f3a6`

---

## ✅ Fix Status (All Applied)

| Issue | Status | Commit |
|-------|--------|--------|
| 1 Dual STT guard | ✅ FIXED | `72b14ec` |
| 2 PDF extraction | ✅ FIXED | `72b14ec` |
| 3 Hidden inputs | ✅ FIXED | `72b14ec` |
| 4 useBackend guard | ✅ FIXED | `72b14ec` |
| 5 Resume URL timing | ✅ FIXED | `72b14ec` |
| 6 Modal scroll CSS | ✅ FIXED | `72b14ec` |
| 7 Silence flush | ✅ FIXED | `72b14ec` |
| 8 Backend URL input | ✅ FIXED | `72b14ec` |
| 9 Ollama timeout | ✅ FIXED | `72b14ec` |
| 10 WS retry cap | ✅ FIXED | `72b14ec` |
| + AudioWorklet | ✅ FIXED | `437f3a6` |
| + Inline favicon | ✅ FIXED | `437f3a6` |

---

## Severity Legend

| Symbol | Level | Meaning |
|--------|-------|---------|
| 🔴 | Critical | Will break visibly on first use |
| 🟡 | High | Degrades core functionality |
| 🟢 | Medium | Polish / edge case |

---

## Issue 1 🔴 — Dual STT Conflict

**Category:** Logic | **Fix effort:** 5 minutes

### Problem

`startCapture()` unconditionally calls `startSpeechRecognition()` every time. When the Corebrum backend is connected, `faster-whisper` sends transcripts over `/ws/ui`. The browser Web Speech API also transcribes independently at the same time. Every word appears twice in the transcript panel.

### Root cause

```javascript
// Current code — always fires regardless of backend state
setChip('stt', 'live');
startSpeechRecognition();  // ← no guard
toast('Capture started', 'ok');
```

### Fix

```javascript
// Replace the last 3 lines of startCapture() with:
setChip('stt', 'live');

// Only start browser STT if backend is NOT handling transcription
if (!CFG.useBackend) {
  startSpeechRecognition();
} else {
  toast('Using backend STT via Corebrum', 'info');
}

toast('Capture started', 'ok');
```

---

## Issue 2 🔴 — `readFileAsText()` Corrupts PDF Files

**Category:** Data | **Fix effort:** 20 minutes

### Problem

`FileReader.readAsText()` treats a PDF as a plain text file. PDFs are a binary format — reading them as UTF-8 text produces garbage bytes like `%PDF-1.4 ... endobj %%EOF`. This garbage string gets injected directly into the LLM system prompt, wasting token budget and confusing the model with meaningless content.

### Root cause

```javascript
// Current code — works for .txt and .docx but destroys PDF
async function readFileAsText(file) {
  return new Promise(res => {
    const reader = new FileReader();
    reader.onload = e => {
      CFG._resumeText = e.target.result.slice(0, 4000); // ← binary PDF = garbage
      res(CFG._resumeText);
    };
    reader.readAsText(file); // ← wrong method for PDF
  });
}
```

### Fix — Option A: Simple (no PDF.js dependency)

```javascript
async function readFileAsText(file) {
  return new Promise(res => {
    if (file.name.toLowerCase().endsWith('.pdf')) {
      // Signal that a resume was uploaded — FAISS will handle it when backend is available
      CFG._resumeText = `[Resume uploaded: ${file.name}. Use job description context for answers.]`;
      res(CFG._resumeText);
      return;
    }
    // DOCX / TXT — plain text read is valid
    const reader = new FileReader();
    reader.onload = e => {
      CFG._resumeText = e.target.result.slice(0, 4000);
      res(CFG._resumeText);
    };
    reader.readAsText(file);
  });
}
```

### Fix — Option B: Full PDF text extraction (recommended)

Add PDF.js to `<head>`:
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
```

Replace `readFileAsText`:
```javascript
async function readFileAsText(file) {
  if (file.name.toLowerCase().endsWith('.pdf')) {
    return await extractPdfText(file);
  }
  // DOCX / TXT
  return new Promise(res => {
    const reader = new FileReader();
    reader.onload = e => {
      CFG._resumeText = e.target.result.slice(0, 4000);
      res(CFG._resumeText);
    };
    reader.readAsText(file);
  });
}

async function extractPdfText(file) {
  try {
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    const ab = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: ab }).promise;
    let text = '';
    for (let i = 1; i <= Math.min(pdf.numPages, 4); i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map(s => s.str).join(' ') + '\n';
    }
    CFG._resumeText = text.slice(0, 4000);
    return CFG._resumeText;
  } catch (e) {
    CFG._resumeText = `[Resume: ${file.name} — could not extract text. Ensure backend is running for RAG indexing.]`;
    return CFG._resumeText;
  }
}
```

---

## Issue 3 🔴 — Hidden Inputs Lose Values Across Browsers

**Category:** Compatibility | **Fix effort:** 10 minutes

### Problem

The merged code uses two `<input type="hidden">` elements (`sel-llm`, `inp-model`) and sets their `.value` via JavaScript. In Chrome this works. In Firefox with strict CSP, and in some Safari versions, JS-written values on hidden inputs are not reliably persisted between the write and the launch handler read. The model selection silently reverts to the HTML default.

### Root cause

```html
<!-- HTML — these hidden inputs are set by JS but unreliable -->
<input type="hidden" id="sel-llm" value="ollama"/>
<input type="hidden" id="inp-model" value="llama3.2"/>
```

```javascript
// selectModelCard() writes to them — unreliable
document.getElementById('sel-llm').value = llm;
document.getElementById('inp-model').value = model;

// Launch handler reads them back — may get wrong value
CFG.llm   = document.getElementById('sel-llm').value;
CFG.model = document.getElementById('inp-model').value.trim() || 'llama3.2';
```

### Fix

**Step 1:** Remove both hidden inputs from the HTML entirely.

**Step 2:** Update `selectModelCard()` to write directly to CFG:

```javascript
function selectModelCard(el) {
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
  el.classList.add('active');

  // Store directly in CFG — no hidden input needed
  CFG.llm   = el.dataset.llm;
  CFG.model = el.dataset.model;

  const isOllama = CFG.llm === 'ollama';
  document.getElementById('f-ollama-url').style.display = isOllama ? '' : 'none';
  document.getElementById('f-api-fields').style.display = isOllama ? 'none' : '';
}
```

**Step 3:** Remove these two lines from the `btn-launch` click handler:

```javascript
// DELETE these lines — CFG.llm and CFG.model are already set by selectModelCard()
// CFG.llm   = document.getElementById('sel-llm').value;
// CFG.model = document.getElementById('inp-model').value.trim() || 'llama3.2';
```

---

## Issue 4 🟡 — `generateAnswer()` Ignores `useBackend` Flag

**Category:** Logic | **Fix effort:** 10 minutes

### Problem

When the Corebrum backend is connected (`CFG.useBackend = true`), the LLM response comes from Ollama running server-side and is pushed to the HUD via `/ws/ui`. However `generateAnswer()` still makes a direct browser-to-Ollama API call in parallel. The result: two answer cards appear for every question — one from the backend, one from the browser.

### Root cause

```javascript
// generateAnswer() — no check for backend mode
async function generateAnswer(question) {
  if (ST.abortCtrl) ST.abortCtrl.abort();
  ST.generating = true;
  // ... always calls Ollama directly, regardless of useBackend
```

### Fix

```javascript
async function generateAnswer(question) {
  // When backend is connected, it handles LLM — answers arrive via handleBackendAnswer()
  if (CFG.useBackend) {
    // Optionally show a "waiting for backend" indicator
    setChip('llm', 'proc');
    return;
  }

  // Browser-only mode — proceed with direct Ollama call
  if (ST.abortCtrl) ST.abortCtrl.abort();
  ST.generating = true;
  setChip('llm', 'proc');
  // ... rest of function unchanged
```

---

## Issue 5 🟡 — Resume Upload Fires Before `CFG.backendUrl` Is Set

**Category:** Timing | **Fix effort:** 15 minutes

### Problem

The file input `change` event fires as soon as the user selects a file — before they click Launch. At that moment, `CFG.backendUrl` is still the JavaScript default (`'http://localhost:8001'`) regardless of what the user may type into the backend URL field. If they're using a custom port or remote host, the index upload goes to the wrong URL and silently fails.

### Fix

Read the backend URL field value directly at upload time, not from CFG:

```javascript
document.getElementById('resume-file-input').addEventListener('change', async function() {
  if (!this.files.length) return;
  const file = this.files[0];

  // Read backend URL from input directly — CFG not set yet at this point
  const backendUrlInput = document.getElementById('inp-backend-url');
  const backendUrl = (backendUrlInput?.value || 'http://localhost:8001').trim().replace(/\/$/, '');

  const zone      = document.getElementById('resume-drop-zone');
  const statusEl  = document.getElementById('resume-upload-status');
  const nameEl    = document.getElementById('resume-filename');

  nameEl.textContent = file.name;
  zone.classList.add('loaded');
  statusEl.style.display = 'block';
  statusEl.textContent   = 'Indexing into RAG…';

  try {
    const fd = new FormData();
    fd.append('file', file);
    const resp = await fetch(`${backendUrl}/api/index-context`, {
      method: 'POST',
      body: fd
    });
    if (resp.ok) {
      statusEl.textContent   = '✓ Resume indexed — AI answers will use your resume as context.';
      statusEl.style.color   = 'var(--green)';
      ST.resumeIndexed = true;
    } else {
      throw new Error(`Server returned ${resp.status}`);
    }
  } catch (e) {
    statusEl.textContent = '⚠ Backend offline — resume context will be added to prompt directly.';
    statusEl.style.color = 'var(--amber)';
    await readFileAsText(file);
  }
});
```

---

## Issue 6 🟡 — Modal Body Overflows on Small Screens

**Category:** UX | **Fix effort:** 5 minutes

### Problem

The modal now has 9 stacked sections (header, resume, job description, role/language, AI model, answer settings, audio source, info, launch button). On screens with height ≤ 900px — most laptops — the Launch button is below the fold and unreachable without knowing to scroll.

### Fix

Add `max-height` and `overflow-y: auto` to `.modal-body` in the CSS:

```css
.modal-body {
  padding: 24px 28px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  max-height: 85vh;        /* prevents overflow beyond viewport */
  overflow-y: auto;        /* enables scrolling within modal */
  scrollbar-width: thin;
  scrollbar-color: var(--border2) transparent;
}
```

Also add scrollbar styling for WebKit:
```css
.modal-body::-webkit-scrollbar { width: 4px; }
.modal-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
```

---

## Issue 7 🟡 — Silence Flush Triggers on Every Speech Pause

**Category:** Logic | **Fix effort:** 20 minutes

### Problem

The 2.5-second silence timer fires a full LLM generation call for every pause in speech — including filler words ("um", "let me think"), mid-sentence pauses, and partial thoughts. During a typical interview where the interviewer speaks for 30 seconds, this triggers 8–12 unnecessary Ollama calls before the question is finished.

### Root cause

```javascript
// Fires for ANY speech pause > 2.5s, regardless of content
if (ST.accumulated.trim().length > 12) {
  ST.silenceTimer = setTimeout(() => {
    flushToLLM(ST.accumulated.trim());  // ← too eager
    ST.accumulated = '';
  }, 2500);
}
```

### Fix

Raise the minimum length threshold and require either a question pattern or a complete multi-word statement before flushing:

```javascript
clearTimeout(ST.silenceTimer);
if (ST.accumulated.trim().length > 25) {  // raised from 12 to 25 chars
  ST.silenceTimer = setTimeout(() => {
    const text = ST.accumulated.trim();
    const wordCount = text.split(/\s+/).length;

    // Only flush if content looks substantive
    const looksLikeQuestion = /\?/.test(text) ||
      /^(tell me|can you|could you|what|how|why|describe|explain|walk me|have you|do you|would you)/i.test(text);
    const isCompleteStatement = wordCount >= 8;

    if (looksLikeQuestion || isCompleteStatement) {
      flushToLLM(text);
    }
    ST.accumulated = '';
  }, 2500);
}
```

---

## Issue 8 🟢 — No UI Input for `backendUrl`

**Category:** UX | **Fix effort:** 10 minutes

### Problem

`CFG.backendUrl` is hardcoded to `'http://localhost:8001'` with no way for the user to change it from the setup modal. Anyone running Corebrum on a different port, or connecting remotely, cannot configure this without editing the source file.

### Fix

**Step 1:** Add input field inside the AI Model `modal-section`, after the Ollama URL field:

```html
<div class="mfield" style="margin-top:8px;">
  <label>Corebrum Backend URL</label>
  <input type="text" id="inp-backend-url"
    value="http://localhost:8001"
    placeholder="http://localhost:8001 — leave blank if backend not running"/>
</div>
```

**Step 2:** Read it in the launch handler:

```javascript
document.getElementById('btn-launch').addEventListener('click', () => {
  // ... existing CFG reads ...
  CFG.backendUrl = document.getElementById('inp-backend-url').value.trim().replace(/\/$/, '')
    || 'http://localhost:8001';
  // ... rest unchanged
});
```

---

## Issue 9 🟢 — `streamOllama()` Has No Timeout

**Category:** Reliability | **Fix effort:** 20 minutes

### Problem

If Ollama is slow, the model is still loading, or the context window overflows, `streamOllama()` hangs indefinitely. `ST.abortCtrl` only aborts when a new question arrives — if no new speech is detected, the app freezes silently with the LLM chip showing "processing" forever.

> Note: This was flagged in the original Corebrum architecture audit and was not carried through in the merge.

### Fix

Add a hard 25-second timeout with a combined AbortSignal:

```javascript
// Add this helper function anywhere in the script
function anySignal(signals) {
  const ctrl = new AbortController();
  for (const sig of signals) {
    if (sig.aborted) { ctrl.abort(); break; }
    sig.addEventListener('abort', () => ctrl.abort(), { once: true });
  }
  return ctrl.signal;
}

// Replace streamOllama() with:
async function streamOllama(messages, card) {
  const timeoutCtrl = new AbortController();
  const timeoutId   = setTimeout(() => timeoutCtrl.abort(), 25000); // 25s hard cap

  const signal = anySignal([ST.abortCtrl.signal, timeoutCtrl.signal]);

  try {
    const resp = await fetch(`${CFG.ollamaUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: CFG.model,
        messages,
        stream: true,
        options: { temperature: 0.7, num_predict: 600 }
      }),
      signal
    });

    if (!resp.ok) throw new Error(`Ollama ${resp.status}: ${await resp.text()}`);

    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let full     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const line of dec.decode(value, { stream: true }).split('\n').filter(Boolean)) {
        try {
          const j = JSON.parse(line);
          if (j.message?.content) { full += j.message.content; streamCard(card, full); }
          if (j.done) return full;
        } catch (e) {}
      }
    }
    return full;

  } catch (e) {
    if (e.name === 'AbortError') {
      return '[Response timed out — Ollama may be busy. Try again.]';
    }
    throw e;
  } finally {
    clearTimeout(timeoutId);
  }
}
```

---

## Issue 10 🟢 — Backend WebSocket Reconnects Forever

**Category:** Reliability | **Fix effort:** 10 minutes

### Problem

`connectBackendWS()` retries indefinitely — it caps the delay at 15 seconds but never stops. If the user is intentionally running in browser-only mode, this means the WS reconnect loop fires every 15 seconds for the entire session. No UI feedback is shown after the first failure.

### Root cause

```javascript
ws.onclose = () => {
  CFG.useBackend = false;
  clearTimeout(ST.wsReconnectTimer);
  ST.wsReconnectTimer = setTimeout(() => {
    ST.wsReconnectDelay = Math.min(ST.wsReconnectDelay * 1.5, 15000);
    connectBackendWS();  // ← loops forever, no exit condition
  }, ST.wsReconnectDelay);
};
```

### Fix

Add a retry counter that stops after 5 attempts and shows a clear alert:

```javascript
ws.onclose = () => {
  CFG.useBackend = false;
  ST.wsReconnectDelay = Math.min(ST.wsReconnectDelay * 1.5, 15000);
  ST._wsFailCount = (ST._wsFailCount || 0) + 1;

  if (ST._wsFailCount >= 5) {
    // Give up — inform user and run in browser-only mode
    showAlert(
      'Corebrum backend unreachable after 5 attempts — running in browser-only mode. ' +
      'Start the backend with: uvicorn main:app --port 8001 --reload'
    );
    return; // stop reconnecting
  }

  clearTimeout(ST.wsReconnectTimer);
  ST.wsReconnectTimer = setTimeout(connectBackendWS, ST.wsReconnectDelay);
};

// Reset fail count on successful open
ws.onopen = () => {
  ST.wsReconnectDelay = 1000;
  ST._wsFailCount     = 0;       // ← reset counter on success
  toast('Connected to Corebrum backend', 'ok');
  CFG.useBackend = true;
};
```

---

## Recommended Fix Order

| Priority | Issue | File location | Time |
|----------|-------|---------------|------|
| 1 | Issue 3 — Remove hidden inputs | `selectModelCard()` + HTML | 10 min |
| 2 | Issue 1 — Dual STT guard | `startCapture()` | 5 min |
| 3 | Issue 2 — PDF extraction | `readFileAsText()` | 20 min |
| 4 | Issue 6 — Modal scroll | `.modal-body` CSS | 5 min |
| 5 | Issue 4 — useBackend guard | `generateAnswer()` | 10 min |
| 6 | Issue 8 — Backend URL input | modal HTML + launch handler | 10 min |
| 7 | Issue 5 — Resume URL timing | file input handler | 15 min |
| 8 | Issue 7 — Silence flush gate | `rec.onresult` handler | 20 min |
| 9 | Issue 9 — Ollama timeout | `streamOllama()` | 20 min |
| 10 | Issue 10 — WS retry cap | `connectBackendWS()` | 10 min |

**Total: ~2 hours**

---

## Quick Pre-Launch Checklist

Before running `corebrum-hud-merged.html` in a live interview, verify:
```
[✅] Issue 3 fixed — hidden inputs removed, CFG.llm/model set directly
[✅] Issue 1 fixed — dual STT guard added to startCapture()
[✅] Issue 2 fixed — PDF.js added or Option A fallback in readFileAsText()
[✅] Issue 6 fixed — modal-body has max-height + overflow-y:auto
[ ] Ollama running: OLLAMA_ORIGINS=* ollama serve
[ ] Model pulled: ollama pull llama3.2
[ ] Corebrum backend running (optional): python main.py
[ ] Browser: Chrome or Edge (Web Speech API required)
[ ] Tab audio: meeting open in separate tab, "Share tab audio" ticked
```

---

## Commits

- `72b14ec` - Fix 10 HUD integration issues: dual STT, PDF extraction, WS retry cap, Ollama timeout
- `437f3a6` - Fix AudioWorklet deprecation, add inline favicon, handle cleanup properly
[ ] Issue 3 fixed — hidden inputs removed, CFG.llm/model set directly
[ ] Issue 1 fixed — dual STT guard added to startCapture()
[ ] Issue 2 fixed — PDF.js added or Option A fallback in readFileAsText()
[ ] Issue 6 fixed — modal-body has max-height + overflow-y:auto
[ ] Ollama running: OLLAMA_ORIGINS=* ollama serve
[ ] Model pulled: ollama pull llama3.2
[ ] Corebrum backend running (optional): uvicorn main:app --port 8001 --reload
[ ] Browser: Chrome or Edge (Web Speech API required)
[ ] Tab audio: meeting open in separate tab, "Share tab audio" ticked
```

---

*Audit performed on `corebrum-hud-merged.html` (82,134 bytes, 2,458 lines)*
*All code snippets are drop-in replacements for the corresponding sections in the merged file.*
