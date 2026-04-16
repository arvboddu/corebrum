chrome.runtime.onMessage.addListener((request) => {
  if (request.action === "start_capture") {
    console.log("[BACKGROUND] Popup requested capture start!");
    
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tabId = request.tabId || tabs[0].id;
      console.log("[BACKGROUND] Targeted Tab ID:", tabId);
      
      chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, async (streamId) => {
        if (chrome.runtime.lastError) {
           console.error("[BACKGROUND] Error getting stream ID:", chrome.runtime.lastError.message);
           return;
        }
        console.log("[BACKGROUND] Successfully generated stream ID:", streamId);
        
        try {
          const hasDocument = await chrome.offscreen.hasDocument();
          if (!hasDocument) {
            console.log("[BACKGROUND] Spinning up Offscreen Document engine...");
            await chrome.offscreen.createDocument({
              url: 'offscreen.html',
              reasons: ['USER_MEDIA'],
              justification: 'Recording tab audio for transcription'
            });
            console.log("[BACKGROUND] Offscreen Document successfully created!");
          } else {
            console.log("[BACKGROUND] Offscreen document already running.");
          }

          console.log("[BACKGROUND] Dispatching ID to offscreen processor...");
          
          // Wait for offscreen document to load its scripts
          await new Promise(r => setTimeout(r, 2000));
          
          // Try sending via runtime.sendMessage (works if offscreen is ready)
          chrome.runtime.sendMessage({
            type: 'start-capture',
            target: 'offscreen',
            data: streamId
          }, (response) => {
            if (chrome.runtime.lastError) {
              console.error("[BACKGROUND] Message error:", chrome.runtime.lastError.message);
              // Fallback: reload offscreen and try again
              chrome.offscreen.closeDocument();
            } else {
              console.log("[BACKGROUND] Message sent, response:", response);
            }
          });
          
          console.log("[BACKGROUND] Message dispatch complete");
        } catch (err) {
          console.error("[BACKGROUND] CRITICAL ERROR creating offscreen document:", err);
        }
      });
    });
  }
});
