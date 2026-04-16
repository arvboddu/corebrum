const heardContent = document.getElementById("heard-content");
const copilotContent = document.getElementById("copilot-content");
const appContainer = document.getElementById("app-container");
const statusIndicator = document.getElementById("status-indicator");
const heardTitle = document.querySelector('#heard-panel .panel-header, [id="heard"] .panel-header, #heard-content + .panel-header, #heard-title') || document.getElementById("heard-title");

const WS_URL = "ws://localhost:8001/ws/ui";
const RECONNECT_INTERVAL = 3000;
const MAX_RECONNECT_ATTEMPTS = 9999;
const SIGNAL_THRESHOLD = 0.01;
let socket = null;
let reconnectAttempts = 0;
let reconnectIntervalId = null;
let lastMessageTime = 0;
let lastTranscript = "";
let systemReady = false;
let isReconnecting = false;
let signalLevel = 0;
let heartbeatTimeout = null;

const DEBUG = false;
const log = (...args) => DEBUG && console.log(...args);

function updateStatusIndicator(status, message) {
  const indicator = statusIndicator || appContainer;
  if (indicator) {
    if (status === "connected") {
      indicator.style.border = "2px solid #00ff00";
    } else if (status === "disconnected") {
      indicator.style.border = "2px solid #ff0000";
    } else if (status === "waiting") {
      indicator.style.border = "2px solid #ffff00";
    }
  }
}

// Auto-Scroll Observer logic
const scrollObserver = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (heardContent && (mutation.target === heardContent || heardContent.contains(mutation.target))) {
      heardContent.scrollTop = heardContent.scrollHeight;
    }
    if (copilotContent && (mutation.target === copilotContent || copilotContent.contains(mutation.target))) {
      copilotContent.scrollTop = copilotContent.scrollHeight;
    }
  });
});

scrollObserver.observe(document.getElementById("app-container"), { 
    childList: true, 
    subtree: true, 
    characterData: true 
});

function connect() {
  const url = WS_URL;
  log("[HUD] Attempting WebSocket connection to:", url);
  socket = new WebSocket(url);

  socket.onopen = () => {
    log("[HUD] ✅ WebSocket OPEN - Connected to Backend!");
    log("[HUD] 🎧 Ready to receive transcripts and answers");
    reconnectTimeout = null;
    reconnectAttempts = 0;
    heardContent.innerHTML = '<div class="heard-block">🟢 CONNECTED</div>';
    copilotContent.innerText = "Ready - Waiting for interview question...";
    lastMessageTime = Date.now();
    systemReady = true;
    updateStatusIndicator("connected");
    
    appContainer.style.border = "1px solid rgba(255, 255, 255, 0.15)";
    appContainer.style.boxShadow = "0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05)";
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      log("[HUD] 📥 Received:", JSON.stringify(data));
      lastMessageTime = Date.now();
      handleMessage(data);
    } catch (err) {
      console.error("[HUD] ❌ JSON parse error:", err, "Raw:", event.data);
    }
  };

  socket.onerror = (error) => {
    console.error("[HUD] ❌ WebSocket Error:", error);
  };

  socket.onclose = (e) => {
    log("[HUD] 🔌 WebSocket closed:", e.code, e.reason);
    updateStatusIndicator("disconnected");
    socket = null;
    if (!isReconnecting) {
      scheduleReconnect();
    }
  };
}

function appendPartialText(text) {
  const activeBlock = getActiveHeardBlock();
  if (!activeBlock) {
    createNewHeardBlock();
  }
  
  const wordSpan = document.createElement("span");
  wordSpan.className = "heard-word";
  wordSpan.textContent = text + " ";
  wordSpan.style.opacity = "0";
  wordSpan.style.transition = "opacity 0.1s ease-in";
  
  const block = getActiveHeardBlock();
  block.appendChild(wordSpan);
  
  requestAnimationFrame(() => {
    wordSpan.style.opacity = "1";
  });
  
  scrollToBottom();
}

function updateHeartbeat() {
  const heardHeader = document.querySelector('#heard-section header');
  if (!heardHeader) return;
  
  if (signalLevel > 0.01) {
    heardHeader.style.color = '#ff6666';
    heardHeader.style.animation = 'pulse 1.5s infinite ease-in-out';
    
    if (heartbeatTimeout) clearTimeout(heartbeatTimeout);
    heartbeatTimeout = setTimeout(() => {
      heardHeader.style.color = '';
      heardHeader.style.animation = '';
    }, 1000); // Keep pulsing for 1s after signal drops to avoid flicker
  }
}

function getActiveHeardBlock() {
  const blocks = heardContent.querySelectorAll(".heard-block");
  return blocks.length > 0 ? blocks[blocks.length - 1] : null;
}

function createNewHeardBlock() {
  const block = document.createElement("div");
  block.className = "heard-block";
  heardContent.appendChild(block);
  return block;
}

function finalizeHeardBlock() {
  const activeBlock = getActiveHeardBlock();
  if (activeBlock && activeBlock.textContent.trim()) {
    activeBlock.classList.add("heard-final");
    setTimeout(() => {
      activeBlock.classList.remove("heard-final");
    }, 800);
  }
}

function scrollToBottom() {
  heardContent.scrollTop = heardContent.scrollHeight;
  window.scrollTo(0, document.body.scrollHeight);
}

function autoScrollToBottom(element) {
  if (element) {
    element.scrollTop = element.scrollHeight;
    window.scrollTo(0, document.body.scrollHeight);
  }
}

function handleMessage(data) {
  log("[HUD] 📝 Handling message type:", data.type);
  
  if (data.type === "status") {
    if (data.message && data.message.includes("Listening")) {
      if (!systemReady) {
        systemReady = true;
        heardContent.innerHTML = '<div class="heard-block">🎤 Listening...</div>';
      }
    }
    log("[HUD] Status:", data.message);
    return;
  }

  if (data.type === "transcript") {
    const content = data.content || data.text;
    log("[HUD] 📄 Transcript:", content);
    if (content && content.trim()) {
      heardContent.innerHTML = '<div class="heard-block heard-final">' + escapeHtml(content.trim()) + '</div>';
      updateStatusIndicator("waiting");
      scrollToBottom();
    }
    return;
  }

  if (data.type === "heard_partial") {
    const text = data.text || data.content;
    if (text && text.trim()) {
      appendPartialText(text.trim());
    }
    return;
  }

  if (data.type === "heard_final") {
    const text = data.text || data.content;
    if (text && text.trim()) {
      finalizeHeardBlock();
    }
    return;
  }

  if (data.type === "clear") {
    copilotContent.innerText = "";
    return;
  }

  if (data.type === "copilot_partial" || data.type === "copilot_answer" || data.type === "advice") {
    const content = data.content || data.text;
    log("[HUD] 🤖 Copilot partial:", content);
    if (content && content.trim()) {
      if (copilotContent.innerText === "Ready - Waiting for interview question..." || copilotContent.innerText === "") {
        copilotContent.innerText = content;
      } else {
        copilotContent.innerText += content;
      }
      autoScrollToBottom(copilotContent);
    }
    return;
  }

  if (data.type === "copilot_final") {
    log("[HUD] ✅ Copilot answer complete");
    // Keep the final content as-is
    return;
  }

  if (data.type === "heartbeat") {
    log("[HUD] 💓 Heartbeat received");
    updateStatusIndicator("connected");
    return;
  }
   
  if (data.type === "amplitude" || data.type === "signal") {
    signalLevel = parseFloat(data.value || data.level || 0);
    updateHeartbeat();
    return;
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function scheduleReconnect() {
  if (isReconnecting || reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) return;
  
  isReconnecting = true;
  reconnectAttempts++;
  log(`[HUD] 🔄 Reconnect attempt ${reconnectAttempts} in ${RECONNECT_INTERVAL}ms...`);
  updateStatusIndicator("disconnected");
  
  reconnectIntervalId = setInterval(() => {
    log("[HUD] 🔄 Attempting reconnect...");
    if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
      connect();
    } else {
      clearInterval(reconnectIntervalId);
      reconnectIntervalId = null;
      isReconnecting = false;
      reconnectAttempts = 0;
    }
  }, RECONNECT_INTERVAL);
}

window.onload = () => {
  log("[HUD] 🚀 Corebrum HUD Starting...");
  log("[HUD] Connecting to backend at:", WS_URL);
  updateStatusIndicator("waiting");
  connect();
};

setInterval(() => {
  if (lastMessageTime && Date.now() - lastMessageTime > 15000) {
    if (!heardContent.querySelector(".heard-block")) {
      heardContent.innerHTML = '<div class="heard-block">⏳ Waiting for audio...</div>';
    }
    updateStatusIndicator("waiting");
  }
}, 2000);