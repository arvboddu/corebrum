const heardContent = document.getElementById("heard-content");
const copilotContent = document.getElementById("copilot-content");
const appContainer = document.getElementById("app-container");

const WS_URL = "ws://127.0.0.1:8001/ws";
const RECONNECT_INTERVAL = 1000;
let socket = null;
let reconnectTimeout = null;
let lastMessageTime = 0;
let lastTranscript = "";
let systemReady = false;

function connect() {
  const url = WS_URL;
  console.log("[WS] Attempting connection to:", url);
  socket = new WebSocket(url);

  socket.onopen = () => {
    console.log("Connected to Backend!");
    reconnectTimeout = null;
    heardContent.innerHTML = '<div class="heard-block">🟢 LIVE</div>';
    copilotContent.innerText = "Ready";
    lastMessageTime = Date.now();
    systemReady = true;
    
    appContainer.style.border = "1px solid rgba(255, 255, 255, 0.15)";
    appContainer.style.boxShadow = "0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05)";
  };

  socket.onmessage = (event) => {
    console.log("Data received:", event.data);
    try {
      const data = JSON.parse(event.data);
      lastMessageTime = Date.now();
      handleMessage(data);
    } catch (err) {
      console.error("[WS] JSON parse error:", err, "Raw:", event.data);
    }
  };

  socket.onerror = (error) => {
    console.log("WebSocket Error:", error);
  };

  socket.onclose = () => {
    console.log("WebSocket Disconnected");
    scheduleReconnect();
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
}

function handleMessage(data) {
  if (data.type === "status") {
    if (data.message && data.message.includes("Listening")) {
      if (!systemReady) {
        systemReady = true;
        heardContent.innerHTML = '<div class="heard-block">🎤 Listening...</div>';
      }
    }
    return;
  }

  if (data.type === "transcript") {
    const content = data.content || data.text;
    if (content && content.trim()) {
      heardContent.innerHTML = '<div class="heard-block heard-final">' + escapeHtml(content.trim()) + '</div>';
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

  if (data.type === "copilot_partial" || data.type === "advice") {
    const content = data.content;
    if (content && content.trim()) {
      copilotContent.innerText += content;
      copilotContent.scrollTop = copilotContent.scrollHeight;
    }
    return;
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function scheduleReconnect() {
  if (reconnectTimeout) return;
  
  heardContent.innerHTML = '<div class="heard-block">🔄 Connecting...</div>';
  
  reconnectTimeout = setTimeout(() => {
    reconnectTimeout = null;
    connect();
  }, RECONNECT_INTERVAL);
}

window.onload = () => {
  console.log("HUD Ready - Connecting to 8001...");
  connect();
};

setInterval(() => {
  if (lastMessageTime && Date.now() - lastMessageTime > 15000) {
    if (!heardContent.querySelector(".heard-block")) {
      heardContent.innerHTML = '<div class="heard-block">Waiting for signal...</div>';
    }
  }
}, 2000);