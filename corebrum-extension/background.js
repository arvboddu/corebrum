let currentStreamId = null;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "start_capture") {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const tabId = request.tabId || tabs[0]?.id;
      
      if (!tabId) {
        sendResponse({ success: false, error: "No active tab found" });
        return;
      }
      
      try {
        if (currentStreamId) {
          await chrome.offscreen.closeDocument().catch(() => {});
          currentStreamId = null;
        }
        
        const streamId = await new Promise((resolve, reject) => {
          chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (streamId) => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else {
              resolve(streamId);
            }
          });
        });
        
        currentStreamId = streamId;
        
        const hasDocument = await chrome.offscreen.hasDocument();
        if (hasDocument) {
          await chrome.offscreen.closeDocument();
          await new Promise(r => setTimeout(r, 500));
        }
        
        await chrome.offscreen.createDocument({
          url: 'offscreen.html',
          reasons: ['USER_MEDIA'],
          justification: 'Recording tab audio for transcription'
        });

        await new Promise(r => setTimeout(r, 1500));
        
        chrome.runtime.sendMessage({
          type: 'start-capture',
          target: 'offscreen',
          data: streamId
        }, (response) => {
          if (chrome.runtime.lastError) {
            sendResponse({ success: false, error: chrome.runtime.lastError.message });
          } else {
            sendResponse({ success: true });
          }
        });
      } catch (err) {
        console.error("[BACKGROUND] Error:", err);
        sendResponse({ success: false, error: err.message });
      }
    });
    return true;
  }
  
  if (request.action === "stop_capture") {
    chrome.offscreen.closeDocument().then(() => {
      currentStreamId = null;
      sendResponse({ success: true });
    }).catch(() => {
      currentStreamId = null;
      sendResponse({ success: true });
    });
    return true;
  }
});
