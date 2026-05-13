const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let stream = null;
let socket = null;
let running = false;
let inFlight = false;
let lastSent = 0;
const targetIntervalMs = 1000 / 24;
const hazardLabels = new Set(
  ["person", "low_furniture", "pet", "floor_clutter", "cabinet", "stairs"].map(
    (label) => label.toLowerCase(),
  ),
);
const speechSupported = "speechSynthesis" in window;
const speechCooldownMs = 1500;
const speechDistanceDeltaM = 0.4;
const lastAnnouncementByLabel = new Map();

function setStatus(text, state) {
  statusEl.textContent = text;
  if (state) {
    statusEl.dataset.state = state;
    startBtn.dataset.state = state;
  }
}

function wsUrl() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/ws`;
}

async function startCamera() {
  if (running) {
    await stopCamera();
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
  } catch (err) {
    setStatus("Camera permission denied", "idle");
    return;
  }

  videoEl.srcObject = stream;
  await new Promise((resolve) => {
    videoEl.onloadedmetadata = resolve;
  });

  resizeCanvases();
  openSocket();
  running = true;
  startBtn.textContent = "Stop camera";
  setStatus("Connecting...", "idle");
  requestAnimationFrame(tick);
}

async function stopCamera() {
  running = false;
  inFlight = false;
  if (socket) {
    socket.close();
    socket = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  if (speechSupported) {
    window.speechSynthesis.cancel();
  }
  lastAnnouncementByLabel.clear();
  startBtn.textContent = "Start camera";
  setStatus("Idle", "idle");
  clearOverlay();
}

function isHazardLabel(label) {
  return hazardLabels.has(String(label).toLowerCase());
}

function formatDistance(distanceM) {
  if (!Number.isFinite(distanceM)) {
    return null;
  }
  return distanceM.toFixed(1);
}

function speak(text) {
  if (!speechSupported) {
    return;
  }
  const synth = window.speechSynthesis;
  if (synth.speaking || synth.pending) {
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  synth.speak(utterance);
}

function announceHazards(detections) {
  if (!speechSupported) {
    return;
  }
  const hazards = detections
    .filter((det) => isHazardLabel(det.label))
    .filter((det) => Number.isFinite(det.distance_m))
    .sort((a, b) => a.distance_m - b.distance_m);

  if (!hazards.length) {
    return;
  }

  const now = performance.now();
  for (const det of hazards) {
    const labelKey = String(det.label).toLowerCase();
    const last = lastAnnouncementByLabel.get(labelKey);
    const distance = det.distance_m;
    const distanceText = formatDistance(distance);
    if (!distanceText) {
      continue;
    }
    if (
      last &&
      now - last.time < speechCooldownMs &&
      Math.abs(distance - last.distance) < speechDistanceDeltaM
    ) {
      continue;
    }
    speak(`${det.label} at ${distanceText} meters`);
    lastAnnouncementByLabel.set(labelKey, { time: now, distance });
    break;
  }
}

function resizeCanvases() {
  const width = videoEl.videoWidth || 1280;
  const height = videoEl.videoHeight || 720;
  overlay.width = width;
  overlay.height = height;
  captureCanvas.width = width;
  captureCanvas.height = height;
}

function openSocket() {
  socket = new WebSocket(wsUrl());
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    if (running) {
      setStatus("Connected", "connected");
    }
  };

  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.error) {
      setStatus("Frame decode failed", "idle");
      inFlight = false;
      return;
    }

    drawDetections(payload);
    inFlight = false;
  };

  socket.onclose = () => {
    inFlight = false;
    if (running) {
      setStatus("Connection closed", "idle");
    }
  };

  socket.onerror = () => {
    setStatus("Connection error", "idle");
  };
}

function tick(timestamp) {
  if (!running) {
    return;
  }

  if (
    !inFlight &&
    timestamp - lastSent >= targetIntervalMs &&
    socket?.readyState === 1
  ) {
    sendFrame();
    lastSent = timestamp;
    inFlight = true;
  }

  requestAnimationFrame(tick);
}

function sendFrame() {
  captureCtx.drawImage(
    videoEl,
    0,
    0,
    captureCanvas.width,
    captureCanvas.height,
  );
  captureCanvas.toBlob(
    (blob) => {
      if (blob && socket?.readyState === 1) {
        socket.send(blob);
      }
    },
    "image/jpeg",
    0.8,
  );
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawDetections(payload) {
  clearOverlay();
  const detections = payload.detections || [];
  if (!detections.length) {
    return;
  }

  overlayCtx.lineWidth = 2;
  overlayCtx.font = "15px Sora";

  detections.forEach((det) => {
    const [x1, y1, x2, y2] = det.box;
    overlayCtx.strokeStyle = "#23d3b4";
    overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const distance =
      det.distance_m !== null && det.distance_m !== undefined
        ? `${det.distance_m} m`
        : "--";
    const label = `${det.label} (${distance})`;
    const textWidth = overlayCtx.measureText(label).width;
    const textHeight = 18;

    overlayCtx.fillStyle = "rgba(0, 0, 0, 0.55)";
    overlayCtx.fillRect(
      x1,
      Math.max(0, y1 - textHeight - 6),
      textWidth + 12,
      textHeight + 6,
    );

    overlayCtx.fillStyle = "#f6f0e8";
    overlayCtx.fillText(label, x1 + 6, Math.max(14, y1 - 8));
  });

  announceHazards(detections);
}

window.addEventListener("resize", resizeCanvases);
startBtn.addEventListener("click", startCamera);
setStatus("Idle", "idle");
