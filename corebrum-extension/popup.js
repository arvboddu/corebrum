let isCapturing = false;

document.getElementById('start').addEventListener('click', () => {
  if (isCapturing) return;
  isCapturing = true;
  chrome.runtime.sendMessage({ action: "start_capture" });
  document.getElementById('status').innerText = "Status: Capturing...";
  document.getElementById('status').style.color = "green";
});

document.getElementById('stop').addEventListener('click', () => {
  isCapturing = false;
  chrome.runtime.sendMessage({ action: "stop_capture" });
  document.getElementById('status').innerText = "Status: Stopped";
  document.getElementById('status').style.color = "red";
});
