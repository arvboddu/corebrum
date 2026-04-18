chrome.runtime.onMessage.addListener(async (msg, sender, sendResponse) => {
  if (msg.target !== 'offscreen' || msg.type !== 'start-capture') {
    return true;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: msg.data } }
    });

    const audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(audioContext.destination);

    const socket = new WebSocket("ws://localhost:8001/ws/audio");
    
    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "debug", message: "BROWSER_CONNECTED" }));
    };

    socket.onclose = (e) => {
      console.log("[OFFSCREEN] WebSocket closed:", e.code);
    };

    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (e) => {
      if (socket.readyState === WebSocket.OPEN) {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcm16[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
        }
        socket.send(pcm16.buffer);
      }
    };
    
    sendResponse({ success: true, message: "Audio capture started" });
  } catch (err) {
    console.error("[OFFSCREEN] Error:", err);
    sendResponse({ success: false, error: err.message });
  }
  
  return true;
});
