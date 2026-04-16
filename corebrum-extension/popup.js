console.log("[POPUP] Script loaded");
document.getElementById('start').addEventListener('click', () => {
  console.log("[POPUP] Start button clicked!");
  chrome.runtime.sendMessage({ action: "start_capture" });
  document.getElementById('status').innerText = "Status: Capturing...";
});
