chrome.runtime.onMessage.addListener(async (msg, sender, sendResponse) => {
  console.log("[OFFSCREEN] Received message:", msg);
  if (msg.target !== 'offscreen' || msg.type !== 'start-capture') {
    console.log("[OFFSCREEN] Message ignored - wrong target or type");
    return true;  // Required for async response
  }

  console.log("[OFFSCREEN] Received start-capture message");
  console.log("[OFFSCREEN] Stream ID:", msg.data);

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: msg.data } }
    });

    console.log("[OFFSCREEN] Got stream with", stream.getAudioTracks().length, "audio tracks");

  const audioContext = new AudioContext({ sampleRate: 16000 });
  const source = audioContext.createMediaStreamSource(stream);
  
  source.connect(audioContext.destination); 

  console.log("[OFFSCREEN] Connecting to WebSocket...");
  const socket = new WebSocket("ws://localhost:8001/ws/audio");
  
  socket.onopen = () => {
    console.log("[OFFSCREEN] WebSocket OPEN!");
    socket.send(JSON.stringify({ type: "debug", message: "BROWSER_CONNECTED" }));
  };
  
  socket.onerror = (err) => {
    console.error("[OFFSCREEN] WebSocket ERROR:", err);
  };
  
  socket.onclose = (e) => {
    console.log("[OFFSCREEN] WebSocket CLOSED:", e.code, e.reason);
  };
  
  socket.onmessage = (e) => {
    console.log("[OFFSCREEN] WebSocket message:", e.data);
  };

  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  source.connect(processor);
  processor.connect(audioContext.destination);

  let bytesSent = 0;
  processor.onaudioprocess = (e) => {
    if (socket.readyState === WebSocket.OPEN) {
      const inputData = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        pcm16[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
      }
      socket.send(pcm16.buffer);
      bytesSent += pcm16.buffer.byteLength;
      if (bytesSent % 40000 === 0) {
        console.log("[OFFSCREEN] Sent", bytesSent, "bytes total");
      }
    }
  };
  
  console.log("[OFFSCREEN] Audio processor started");
  console.log("[OFFSCREEN] Setup complete - waiting for audio...");
  
  sendResponse({ success: true, message: "Audio capture started" });
  } catch (err) {
    console.error("[OFFSCREEN] Error:", err);
    sendResponse({ success: false, error: err.message });
  }
  
  return true;  // Keep message channel open for async response
});