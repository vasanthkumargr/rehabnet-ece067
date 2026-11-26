// Live exercise session: camera (or a video file) -> MediaPipe in the browser -> landmarks to the server over a
// WebSocket -> repetition count, score and feedback back from the server.
import { api, esc, scoreBg, scoreColor, sideText, state, toast, FACTOR } from "./ui.js";
import { brightness, cameraView, detect, draw, framing, loadPose, PART_JOINTS } from "./pose.js";

const VIEW_OK = { front: ["front", "45°"], "front or 45°": ["front", "45°"], "45° or side": ["45°", "side"], "side or 45°": ["side", "45°"] };

export async function liveView(root, { assignment = null, exercise = null, side = null, target = 10, onDone = null }) {
  const exList = state.exercises.length ? state.exercises : (state.exercises = await api("/api/exercises"));
  const exKey = assignment ? assignment.exercise : exercise;
  const ex = exList.find((e) => e.key === exKey);
  const tgt = assignment ? assignment.sets * assignment.reps : target;
  const sd = assignment ? assignment.side : side;
  let samples = [];
  try { samples = await api("/api/samples"); } catch { /* optional */ }

  root.innerHTML = `
  <div class="row spread"><h1>${esc(ex.name)}${sd ? ` <span class="muted">(${sideText(sd)} side)</span>` : ""}</h1>
    <a class="btn secondary" href="#/">Back</a></div>
  <div class="live">
    <div>
      <div class="stage" id="stage">
        <video id="vid" playsinline muted></video>
        <canvas id="cv"></canvas>
        <div class="badge hide" id="badge"></div>
        <div class="overlay-msg hide" id="msg"></div>
      </div>
      <div class="card" id="setup" style="margin-top:1rem">
        <h3>Before you start</h3>
        <p>${esc(ex.instructions)}</p>
        <p class="note small">Place the camera so it sees your <b>whole body</b>, about 2–3 m away, with you facing it <b>${esc(ex.view)}</b>. Good, even light helps.</p>
        <div class="row">
          <button class="btn big" id="useCam">Start with camera</button>
          <label class="btn secondary" style="margin:0">Use a video file<input type="file" id="file" accept="video/*" class="hide"></label>
          ${samples.filter((s) => s.exercise === exKey).map((s) => `<button class="btn secondary small" data-sample="${esc(s.url)}">Demo video: ${esc(s.label)}</button>`).join("")}
        </div>
        <label class="row small" style="font-weight:400;margin-top:.8rem;gap:.4rem"><input type="checkbox" id="speak" checked style="width:auto"> Read feedback aloud</label>
      </div>
    </div>
    <aside class="grid" style="align-content:start">
      <div class="card center">
        <div class="stat-label">Repetitions</div>
        <div class="counter"><span id="count">0</span><span class="muted" style="font-size:1.6rem"> / ${tgt}</span></div>
        <div class="small muted" id="setinfo">${assignment ? `${assignment.sets} sets × ${assignment.reps}` : ""}</div>
        <div style="margin-top:.8rem" class="stat-label">Movement</div>
        <div class="gauge" title="How far you are into the movement"><span id="gauge" style="width:0%"></span></div>
      </div>
      <div class="card">
        <div class="row spread"><h3>Last repetition</h3><span id="lastScore"></span></div>
        <div id="lastFb" class="muted">Waiting for your first repetition…</div>
      </div>
      <div class="card"><h3>This session</h3><div class="rep-list" id="reps"></div></div>
      <button class="btn danger big hide" id="stop">Finish session</button>
      <div class="small muted" id="perf"></div>
    </aside>
  </div>`;

  const vid = root.querySelector("#vid"), cv = root.querySelector("#cv"), ctx = cv.getContext("2d");
  const msgEl = root.querySelector("#msg"), badge = root.querySelector("#badge");
  let ws = null, running = false, source = "camera", stream = null, lastSend = 0, count = 0, ended = false;
  let highlight = null, highlightUntil = 0, frames = 0, tStart = 0, detectMs = 0, lastFramingMsg = 0;
  const win = {};

  const say = (text) => {
    if (!root.querySelector("#speak")?.checked || !("speechSynthesis" in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.replace(/\(.*?\)/g, ""));
    u.rate = 1.0; speechSynthesis.speak(u);
  };
  const showMsg = (text, ms = 3500) => {
    msgEl.textContent = text; msgEl.classList.remove("hide");
    clearTimeout(msgEl._h); if (ms) msgEl._h = setTimeout(() => msgEl.classList.add("hide"), ms);
  };

  async function start(kind, url = null, file = null) {
    source = kind;
    root.querySelector("#setup").classList.add("hide");
    showMsg("Loading the pose model…", 0);
    try {
      if (kind === "camera") {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
        vid.srcObject = stream; vid.classList.add("mirror"); cv.classList.add("mirror");
      } else {
        vid.src = url || URL.createObjectURL(file); vid.loop = false;
        await new Promise((ok, bad) => { vid.onloadeddata = ok; vid.onerror = () => bad(new Error("unsupported video")); });
      }
      await loadPose("full");
      if (kind === "camera") await vid.play();   // a video file starts once the server is ready (see onEvent)
    } catch (e) {
      showMsg(kind === "camera" ? "Could not open the camera: " + e.message + ". Check the browser's camera permission." : "Could not play the video: " + e.message, 0);
      root.querySelector("#setup").classList.remove("hide");
      return;
    }
    cv.width = vid.videoWidth; cv.height = vid.videoHeight;
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/session");
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", token: state.token, assignment_id: assignment?.id, exercise: exKey, side: sd, target: tgt, source: kind === "camera" ? "camera" : "video" }));
    ws.onmessage = (ev) => onEvent(JSON.parse(ev.data));
    ws.onclose = () => { if (!ended) { running = false; showMsg("Connection to the server was lost. Your repetitions so far are saved.", 0); } };
    root.querySelector("#stop").classList.remove("hide");
    vid.onended = () => finish();
    if (kind !== "camera") {
      // A video file is analysed frame by frame: pause it while the page is hidden (browsers stop animation
      // there) and slow playback down if this device cannot analyse frames fast enough, so no movement is skipped.
      document.addEventListener("visibilitychange", () => {
        if (!running) return;
        if (document.hidden) vid.pause(); else vid.play().catch(() => {});
      });
    }
  }

  function onEvent(e) {
    if (e.type === "ready") {
      running = true; tStart = performance.now();
      showMsg(kindIntro(), 4000);
      badge.textContent = e.personal ? "Using your personal baseline" : (source === "camera" ? "Live" : "Analysing video");
      badge.classList.remove("hide");
      if (source === "camera") requestAnimationFrame(loop);
      else stepVideo();
    } else if (e.type === "error") {
      showMsg(e.message, 0);
    } else if (e.type === "tick") {
      root.querySelector("#gauge").style.width = Math.max(0, Math.min(100, e.progress * 100)) + "%";
    } else if (e.type === "rep_counted") {
      count = e.count; root.querySelector("#count").textContent = count;
      if (navigator.vibrate) navigator.vibrate(60);
      if (count === tgt) { say("Well done, that was your last repetition."); showMsg("Target reached! You can finish the session.", 5000); }
    } else if (e.type === "rep_scored") {
      addRep(e);
    } else if (e.type === "no_person") {
      showMsg("I can't see you - step back into the picture.");
    } else if (e.type === "summary") {
      ended = true; showSummary(e);
    }
  }

  const kindIntro = () => (source === "camera" ? "Start when you are ready - I'll count and give feedback." : "Analysing the video…");

  function addRep(r) {
    const fb = r.feedback[0];
    const ls = root.querySelector("#lastScore");
    ls.innerHTML = `<span class="score-chip" style="background:${scoreBg(r.score)};color:${scoreColor(r.score)}">${Math.round(r.score)}</span>`;
    root.querySelector("#lastFb").innerHTML = r.feedback.map((m) => `<div>${esc(m.text)}</div>`).join("");
    root.querySelector("#reps").insertAdjacentHTML("afterbegin", `<div class="rep-item"><span class="score-chip" style="background:${scoreBg(r.score)};color:${scoreColor(r.score)}">${Math.round(r.score)}</span><div class="small"><b>Rep ${r.n}</b> · ${r.duration.toFixed(1)} s<br>${esc(fb.text)}</div></div>`);
    if (fb.code !== "good") {
      showMsg(fb.text, 4000); say(fb.text);
      const part = Object.keys(PART_JOINTS).sort((a, b) => b.length - a.length).find((p) => fb.text.toLowerCase().includes(p));
      highlight = part ? PART_JOINTS[part] : null; highlightUntil = performance.now() + 3000;
    } else if (r.n % 3 === 0) { say("Good"); }
  }

  // A video file is analysed frame by frame (seek -> detect -> next), 15 frames per second of video. This gives the
  // same result on any device and even when the browser throttles a background page.
  async function stepVideo() {
    const step = 1 / 15;
    vid.pause();
    let t = 0;
    const seek = (to) => new Promise((ok) => { vid.onseeked = () => ok(); vid.currentTime = to; });
    while (running && t < vid.duration) {
      await seek(t);
      const t0 = performance.now();
      const res = detect(vid);
      detectMs = 0.9 * detectMs + 0.1 * (performance.now() - t0); frames++;
      draw(ctx, res?.img, cv.width, cv.height, performance.now() < highlightUntil ? highlight : null);
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "frame", t: t * 1000, w: res ? res.world.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)]) : null, view: res ? cameraView(res.world).view : null }));
      if (frames % 15 === 0) root.querySelector("#perf").textContent = `Video ${Math.round(t)} / ${Math.round(vid.duration)} s analysed · pose ${detectMs.toFixed(0)} ms/frame`;
      t += step;
    }
    if (running) finish();
  }

  function loop() {
    if (!running) return;
    if (vid.readyState >= 2 && !vid.paused) {
      const t0 = performance.now();
      const res = detect(vid);
      detectMs = 0.9 * detectMs + 0.1 * (performance.now() - t0); frames++;
      draw(ctx, res?.img, cv.width, cv.height, performance.now() < highlightUntil ? highlight : null);
      const now = source === "camera" ? performance.now() : vid.currentTime * 1000;
      if (now - lastSend >= 60 || now < lastSend) {
        lastSend = now;
        let view = null;
        if (source === "camera" && performance.now() - lastFramingMsg > 5000 && frames % 5 === 0 && brightness(vid) < 45) {
          showMsg("The room is too dark for me to see you well - please turn on a light or face a window.", 5000);
          lastFramingMsg = performance.now();
        } else if (res) {
          view = cameraView(res.world).view;
          const f = framing(res.img);
          if (!f.ok && performance.now() - lastFramingMsg > 5000) { showMsg(f.msg); lastFramingMsg = performance.now(); }
          const ok = VIEW_OK[ex.view] || [ex.view];
          if (f.ok && !ok.includes(view) && performance.now() - lastFramingMsg > 8000) {
            showMsg(`For this exercise, turn so the camera sees you ${ex.view} (now: ${view}).`); lastFramingMsg = performance.now();
          }
        }
        if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "frame", t: now, w: res ? res.world.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)]) : null, view }));
      }
      if (frames % 10 === 0) {
        root.querySelector("#perf").textContent = `Pose: ${detectMs.toFixed(0)} ms/frame (${landmarkerInfo()})` + (source !== "camera" && vid.playbackRate < 1 ? ` · video slowed to ${Math.round(vid.playbackRate * 100)}%` : "");
        if (source !== "camera") {
          // need at least ~15 analysed frames per second of video: measure it and slow the video if needed
          const vt = vid.currentTime, dv = vt - (win.vt ?? vt), df = frames - (win.f ?? frames);
          if (dv > 0.5) {
            const fpsVideo = df / dv;
            let rate = vid.playbackRate;
            if (fpsVideo < 13) rate = Math.max(0.2, rate * fpsVideo / 15);
            else if (fpsVideo > 22 && rate < 1) rate = Math.min(1, rate * 1.25);
            if (Math.abs(rate - vid.playbackRate) > 0.05) vid.playbackRate = rate;
          }
          win.vt = vt; win.f = frames;
        }
      }
    }
    requestAnimationFrame(loop);
  }
  const landmarkerInfo = () => "on this device";

  function finish() {
    if (ended) return;
    running = false;
    if (ws && ws.readyState === 1) { ws.send(JSON.stringify({ type: "end" })); showMsg("Saving your session…", 0); }
    else showSummary({ reps: 0 });
    stopMedia();
  }
  function stopMedia() {
    vid.pause();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if ("speechSynthesis" in window) speechSynthesis.cancel();
  }

  function showSummary(s) {
    stopMedia();
    msgEl.classList.add("hide");
    const box = document.createElement("div");
    box.className = "card"; box.style.marginTop = "1rem";
    if (!s.reps) {
      box.innerHTML = `<h2>No repetitions recorded</h2><p class="muted">Nothing was saved. Check that the camera sees your whole body and try again.</p><a class="btn" href="#/">Back</a>`;
    } else {
      const f = s.factors || {};
      box.innerHTML = `<h2>Session complete</h2>
        <div class="grid g3">
          <div><div class="stat">${s.reps}</div><div class="stat-label">repetitions (target ${tgt})</div></div>
          <div><div class="stat" style="color:${scoreColor(s.mean_score)}">${Math.round(s.mean_score)}</div><div class="stat-label">average score</div></div>
          <div><div class="stat">${Math.round(s.consistency)}</div><div class="stat-label">consistency</div></div>
        </div>
        <h3 style="margin-top:1rem">Breakdown</h3>
        <table>${Object.entries(f).filter(([, v]) => v != null).map(([k, v]) => `<tr><td>${FACTOR[k] || k}</td><td style="width:60%"><div class="progress"><span style="width:${v}%;background:${scoreColor(v)}"></span></div></td><td>${Math.round(v)}</td></tr>`).join("")}</table>
        ${s.weakest ? `<p class="note small" style="margin-top:.8rem">Focus for next time: <b>${esc(FACTOR[s.weakest] || s.weakest)}</b>.</p>` : ""}
        <div class="row" style="margin-top:1rem"><a class="btn" href="#/">Done</a></div>`;
      toast("Session saved");
    }
    root.querySelector("#setup").replaceWith(box);
    root.querySelector("#stop").classList.add("hide");
    if (onDone) onDone(s);
  }

  root.querySelector("#useCam").onclick = () => start("camera");
  root.querySelector("#file").onchange = (e) => e.target.files[0] && start("file", null, e.target.files[0]);
  root.querySelectorAll("[data-sample]").forEach((b) => (b.onclick = () => start("sample", b.dataset.sample)));
  root.querySelector("#stop").onclick = finish;
  window.addEventListener("hashchange", () => { if (running) finish(); stopMedia(); }, { once: true });
}
