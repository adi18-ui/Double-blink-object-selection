/* =====================================================================
   bci_integration.js  —  real gaze + real blink for the grid UI
   =====================================================================
   Drop this in just before </body> of Open_BCI.html:
       <script src="https://webgazer.cs.brown.edu/webgazer.js"></script>
       <script src="bci_integration.js"></script>

   Your UI already listens on window for:
       'bci-gaze'  {detail:{id}}  -> highlight tile id (1..4)
       'bci-gaze-leave'           -> unfocus
       'bci-blink'                -> select the focused tile
   This script fires those from real sources:
     - webcam gaze (webgazer) -> which grid quadrant you're looking at -> bci-gaze
     - the Python detector (via SSE bridge) -> bci-blink

   NOTE on the current model: the trained detector only fires reliably on
   "look up + double-blink". So gaze-highlight of the BOTTOM tiles + blink-select
   won't work well until the model is retrained gaze-invariant. For now:
     - gaze highlighting works for all tiles (impressive, model-independent)
     - blink-select works when you double-blink looking up/at the screen
   Set GAZE_MODE='mouse' to drive highlight with the mouse instead of webcam
   (useful when you want selection to just work during a scripted demo).
===================================================================== */

const BRIDGE_URL = "http://localhost:8765/events";
const GAZE_MODE = "webcam";        // 'webcam' | 'mouse'
const GAZE_DEBOUNCE_MS = 250;      // how long gaze must rest on a tile before highlight
const OFFSCREEN_LEAVE_MS = 600;    // unfocus after gaze leaves for this long

// ---- fire the UI's events ----
function focusTile(id) { window.dispatchEvent(new CustomEvent("bci-gaze", { detail: { id } })); }
function leaveTile()   { window.dispatchEvent(new CustomEvent("bci-gaze-leave")); }
function selectTile()  { window.dispatchEvent(new CustomEvent("bci-blink")); }

// ---- map a screen point to a grid tile id (2x2 quadrants) ----
// webgazer accuracy is coarse (~100-200px), so quadrant granularity matches it.
// Grid layout assumed: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right.
function pointToTile(x, y) {
  const w = window.innerWidth, h = window.innerHeight;
  // ignore the outer margins so glances off the grid don't select
  const mx = w * 0.12, my = h * 0.12;
  if (x < mx || x > w - mx || y < my || y > h - my) return null;
  const col = x < w / 2 ? 0 : 1;
  const row = y < h / 2 ? 0 : 1;
  return row * 2 + col + 1;   // 1..4
}

// ---- debounced gaze -> highlight ----
let lastTile = null, restTile = null, restSince = 0, offSince = 0;
function onGazePoint(x, y, t) {
  const tile = pointToTile(x, y);
  if (tile === null) {
    if (lastTile !== null && t - offSince > OFFSCREEN_LEAVE_MS) { leaveTile(); lastTile = null; }
    restTile = null;
    return;
  }
  offSince = t;
  if (tile !== restTile) { restTile = tile; restSince = t; return; }
  // tile held long enough -> focus it
  if (t - restSince >= GAZE_DEBOUNCE_MS && tile !== lastTile) {
    focusTile(tile);
    lastTile = tile;
  }
}

// ---- gaze source ----
function startWebcamGaze() {
  if (typeof webgazer === "undefined") {
    console.warn("[bci] webgazer not loaded; falling back to mouse gaze");
    return startMouseGaze();
  }
  webgazer.setGazeListener((data) => {
    if (!data) return;
    onGazePoint(data.x, data.y, performance.now());
  }).begin();
  webgazer.showVideoPreview(true).showPredictionPoints(true);
  console.log("[bci] webcam gaze started — calibrate by clicking a few points, then look at tiles");
}

function startMouseGaze() {
  window.addEventListener("mousemove", (e) => onGazePoint(e.clientX, e.clientY, performance.now()));
  console.log("[bci] mouse gaze started — move the mouse over tiles to highlight");
}

// ---- optional signal monitor (canvas; independent of detection) ----
const SHOW_MONITOR = true;
let mon = null;
function initMonitor() {
  if (!SHOW_MONITOR) return null;
  const wrap = document.createElement("div");
  wrap.style.cssText = "position:fixed;left:8px;bottom:8px;width:340px;height:96px;"
    + "background:rgba(10,12,16,.82);border:1px solid #2a2f3a;border-radius:8px;"
    + "padding:6px 8px;z-index:99999;font:11px monospace;color:#8b95a7";
  wrap.innerHTML = '<div id="mon-lbl" style="margin-bottom:2px">ch0 — waiting…</div>'
    + '<canvas id="mon-cv" width="324" height="66"></canvas>';
  document.body.appendChild(wrap);
  const cv = wrap.querySelector("#mon-cv"), ctx = cv.getContext("2d");
  const N = 324, buf = new Array(N).fill(0);
  let blinkFlash = 0, selectFlash = 0, count = 0;
  const state = { buf, blinkFlash: 0, selectFlash: 0, count: 0 };
  function push(v) { buf.push(v); if (buf.length > N) buf.shift(); state.count++; }
  function flashBlink() { state.blinkFlash = 12; }
  function flashSelect() { state.selectFlash = 18; }
  function draw() {
    ctx.clearRect(0, 0, N, 66);
    // autoscale to recent extent
    let m = 1; for (const v of buf) m = Math.max(m, Math.abs(v));
    ctx.strokeStyle = state.selectFlash > 0 ? "#ffd166"
      : state.blinkFlash > 0 ? "#4dd0a0" : "#3d7dff";
    ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 0; i < buf.length; i++) {
      const x = i, yy = 33 - (buf[i] / m) * 30;
      i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy);
    }
    ctx.stroke();
    if (state.blinkFlash > 0) state.blinkFlash--;
    if (state.selectFlash > 0) state.selectFlash--;
    const lbl = document.getElementById("mon-lbl");
    if (lbl) lbl.textContent = "ch0 — live  (samples " + state.count + ")";
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
  return { push, flashBlink, flashSelect };
}

// ---- blink/signal source (Python detector via SSE) ----
function startBridge() {
  let es;
  const connect = () => {
    es = new EventSource(BRIDGE_URL);
    es.onmessage = (e) => {
      try {
        const m = JSON.parse(e.data);
        if (m.event === "SELECT") {
          selectTile();
          if (mon) mon.flashSelect();
          console.log("[bci] SELECT (" + (m.prob != null ? "prob=" + m.prob : "gap=" + m.gap) + ")");
        } else if (m.event === "blink") {
          if (mon) mon.flashBlink();
        } else if (m.event === "signal") {
          if (mon) mon.push(m.v);
        }
      } catch (_) {}
    };
    es.onerror = () => { es.close(); setTimeout(connect, 1500); }; // auto-reconnect
  };
  connect();
  console.log("[bci] listening on " + BRIDGE_URL);
}

// ---- go ----
window.addEventListener("load", () => {
  mon = initMonitor();
  startBridge();
  if (GAZE_MODE === "webcam") startWebcamGaze(); else startMouseGaze();
});
