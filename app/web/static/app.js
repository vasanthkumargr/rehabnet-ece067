// RehabNet single-page app: routing, login, doctor portal, patient portal.
import { $, api, barChart, esc, fmtDate, lineChart, logout, modal, scoreColor, scorePill, setToken, sideText, state, toast, FACTOR } from "./ui.js";
import { liveView } from "./live.js";

const app = $("#app");
const COLORS = ["#0F766E", "#122B4A", "#D97706", "#7C3AED", "#B91C1C", "#0369A1"];

// ------------------------------------------------------------------ router
const routes = [
  [/^#\/login$/, loginView],
  [/^#\/settings$/, settingsView],
  [/^#\/library$/, libraryView],
  [/^#\/patient\/(\d+)$/, (m) => patientView(+m[1])],
  [/^#\/session\/(\d+)$/, (m) => sessionView(+m[1])],
  [/^#\/do\/(\d+)$/, (m) => doView(+m[1])],
  [/^#\/try\/([a-z_]+)$/, (m) => tryView(m[1])],
  [/^#\/progress$/, () => myProgressView()],
  [/^#\/?$/, homeView],
];

async function router() {
  const h = location.hash || "#/";
  if (!state.token && h !== "#/login") { location.hash = "#/login"; return; }
  if (state.token && !state.user) {
    try { state.user = await api("/api/me"); } catch { return; }
  }
  renderNav();
  for (const [re, fn] of routes) {
    const m = h.match(re);
    if (m) {
      app.innerHTML = '<p class="muted">Loading…</p>';
      try { await fn(m); } catch (e) { app.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
      app.focus();
      return;
    }
  }
  location.hash = "#/";
}
window.addEventListener("hashchange", router);

function renderNav() {
  const nav = $("#nav");
  if (!state.user) { nav.innerHTML = ""; return; }
  const u = state.user, h = location.hash || "#/";
  const link = (href, txt) => `<a href="${href}" class="${h === href ? "active" : ""}">${txt}</a>`;
  nav.innerHTML = (u.role === "doctor"
    ? link("#/", "Patients") + link("#/library", "Exercise library")
    : link("#/", "My exercises") + link("#/progress", "My progress"))
    + link("#/settings", "Account") + `<button id="logoutBtn" title="Log out ${esc(u.full_name)}">Log out</button>`;
  $("#logoutBtn").onclick = logout;
}

// ------------------------------------------------------------------ login
async function loginView() {
  renderNav();
  app.innerHTML = `
  <div class="card form-narrow">
    <h1>Sign in to RehabNet</h1>
    <p class="muted small">Doctors and patients use the same sign-in.</p>
    <form id="f">
      <label for="u">Username</label><input id="u" autocomplete="username" required>
      <label for="p">Password</label><input id="p" type="password" autocomplete="current-password" required>
      <div id="mfa" class="hide"><label for="c">6-digit code from your authenticator app</label><input id="c" inputmode="numeric" autocomplete="one-time-code" maxlength="6"></div>
      <div id="err" class="error hide"></div>
      <button class="btn big" style="width:100%;margin-top:1rem">Sign in</button>
    </form>
  </div>`;
  let mfaToken = null;
  $("#f").onsubmit = async (e) => {
    e.preventDefault();
    const err = $("#err"); err.classList.add("hide");
    try {
      let r;
      if (mfaToken) r = await api("/api/auth/mfa/verify", { method: "POST", body: { code: $("#c").value, mfa_token: mfaToken } });
      else r = await api("/api/auth/login", { method: "POST", body: { username: $("#u").value, password: $("#p").value } });
      if (r.mfa_required) { mfaToken = r.mfa_token; $("#mfa").classList.remove("hide"); $("#c").focus(); return; }
      setToken(r.token); state.user = r.user;
      if (r.user.must_change_pw) { toast("Please choose your own password"); location.hash = "#/settings"; }
      else location.hash = "#/";
    } catch (ex) { err.textContent = ex.message; err.classList.remove("hide"); }
  };
}

// ------------------------------------------------------------------ home
async function homeView() {
  if (state.user.role === "doctor") return doctorHome();
  return patientHome();
}

// ------------------------------------------------------------------ doctor
async function doctorHome() {
  const ps = await api("/api/patients");
  app.innerHTML = `
  <div class="row spread"><h1>Patients</h1><button class="btn" id="add">+ New patient</button></div>
  ${state.user.mfa_enabled ? "" : `<p class="warn small">Two-factor login is off for your account. <a href="#/settings">Turn it on</a> to protect patient data.</p>`}
  <div class="card">
    ${ps.length ? `<table><thead><tr><th>Name</th><th>Condition</th><th>Exercises</th><th>Last session</th><th>Last score</th><th>This week</th></tr></thead><tbody>
      ${ps.map((p) => `<tr class="click" data-id="${p.id}"><td><b>${esc(p.full_name)}</b><div class="small muted">${esc(p.username)}</div></td>
        <td>${esc(p.condition || "–")}</td><td>${p.assignments}</td><td>${fmtDate(p.last_session)}</td><td>${scorePill(p.last_score)}</td>
        <td>${p.adherence_week == null ? "–" : `<div class="progress" style="width:90px" title="${p.adherence_week}% of planned sessions"><span style="width:${p.adherence_week}%"></span></div>`}</td></tr>`).join("")}
    </tbody></table>` : `<p class="muted">No patients yet. Add one to prescribe exercises.</p>`}
  </div>`;
  app.querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = () => (location.hash = "#/patient/" + tr.dataset.id)));
  $("#add").onclick = newPatientDialog;
}

function newPatientDialog() {
  const m = modal(`<h2>New patient</h2><form id="np">
    <label>Full name</label><input name="full_name" required>
    <label>Username (for signing in)</label><input name="username" required pattern="[a-zA-Z0-9_.-]{3,}">
    <label>Condition (optional)</label><input name="condition" placeholder="e.g. knee replacement, low back pain">
    <label>Year of birth (optional)</label><input name="birth_year" type="number" min="1900" max="2026">
    <div class="error hide" id="e"></div>
    <div class="row" style="margin-top:1rem"><button class="btn">Create</button><button type="button" class="btn secondary" data-close>Cancel</button></div></form>`);
  m.el.querySelector("#np").onsubmit = async (ev) => {
    ev.preventDefault();
    const fd = Object.fromEntries(new FormData(ev.target));
    if (!fd.birth_year) delete fd.birth_year; else fd.birth_year = +fd.birth_year;
    try {
      const r = await api("/api/patients", { method: "POST", body: fd });
      m.el.innerHTML = `<h2>Patient created</h2><p>Give these sign-in details to <b>${esc(r.patient.full_name)}</b>. They will be asked to choose their own password the first time.</p>
        <div class="kv"><span>Username</span><span class="code">${esc(r.patient.username)}</span><span>Temporary password</span><span class="code">${esc(r.temporary_password)}</span></div>
        <div class="row" style="margin-top:1rem"><a class="btn" href="#/patient/${r.patient.id}" data-close>Open patient</a></div>`;
      m.el.querySelector("[data-close]").addEventListener("click", m.close);
    } catch (e) { const el = m.el.querySelector("#e"); el.textContent = e.message; el.classList.remove("hide"); }
  };
}

async function patientView(pid) {
  const [d, prog, exs] = await Promise.all([api(`/api/patients/${pid}`), api(`/api/patients/${pid}/progress`), api("/api/exercises")]);
  state.exercises = exs;
  const p = d.patient;
  app.innerHTML = `
  <div class="row spread"><div><h1>${esc(p.full_name)}</h1><div class="muted small">${esc(p.username)}${p.condition ? " · " + esc(p.condition) : ""}${p.birth_year ? " · born " + p.birth_year : ""}</div></div>
    <div class="row"><a class="btn secondary" href="/api/patients/${pid}/report.pdf" id="pdf" target="_blank">PDF report</a><button class="btn secondary" id="reset">Reset password</button></div></div>
  <div class="tabs" role="tablist"><button class="active" data-t="ov">Overview</button><button data-t="ex">Exercises</button><button data-t="se">Sessions</button></div>
  <section id="t-ov"></section><section id="t-ex" class="hide"></section><section id="t-se" class="hide"></section>`;
  app.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => {
    app.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
    ["ov", "ex", "se"].forEach((t) => $("#t-" + t).classList.toggle("hide", t !== b.dataset.t));
  }));
  $("#pdf").onclick = async (e) => {
    e.preventDefault();
    const blob = await api(`/api/patients/${pid}/report.pdf`);
    window.open(URL.createObjectURL(blob), "_blank");
  };
  $("#reset").onclick = async () => {
    if (!confirm("Create a new temporary password for this patient?")) return;
    const r = await api(`/api/patients/${pid}/reset_password`, { method: "POST" });
    modal(`<h2>New temporary password</h2><p class="code">${esc(r.temporary_password)}</p><button class="btn" data-close>OK</button>`);
  };
  renderOverview($("#t-ov"), prog, d);
  renderAssignments($("#t-ex"), pid, d.assignments, exs);
  renderSessions($("#t-se"), d.sessions, pid);
}

function renderOverview(el, prog, d) {
  const wk = prog.weeks.filter((w) => w.planned);
  el.innerHTML = `
  <div class="grid g3">
    <div class="card"><div class="stat">${prog.total_sessions}</div><div class="stat-label">sessions in total</div></div>
    <div class="card"><div class="stat">${wk.length ? (wk[wk.length - 1].adherence ?? "–") + "%" : "–"}</div><div class="stat-label">adherence this week</div></div>
    <div class="card"><div class="stat">${d.assignments.filter((a) => a.active).length}</div><div class="stat-label">active exercises</div></div>
  </div>
  <div class="card" style="margin-top:1rem"><h2 style="margin-top:0">Session score over time</h2>
    ${lineChart(prog.exercises.map((e, i) => ({ name: e.name, color: COLORS[i % COLORS.length], points: e.series.map((p) => ({ x: new Date(p.date), y: p.score })) })), { ylabel: "score" })}
    <table style="margin-top:.8rem"><thead><tr><th>Exercise</th><th>Sessions</th><th>First 3</th><th>Last 3</th><th>Score trend</th><th>Range of motion</th></tr></thead><tbody>
    ${prog.exercises.map((e) => `<tr><td>${esc(e.name)}</td><td>${e.sessions}</td><td>${scorePill(e.first_avg)}</td><td>${scorePill(e.last_avg)}</td>
      <td>${trendText(e.score_trend, e.score_trend_per_week)}</td><td>${trendText(e.rom_trend, e.rom_trend_per_week)}</td></tr>`).join("") || '<tr><td colspan="6" class="muted">No sessions yet</td></tr>'}
    </tbody></table></div>
  <div class="card" style="margin-top:1rem"><h2 style="margin-top:0">Adherence (last 8 weeks)</h2>
    ${wk.length ? barChart(wk.map((w) => ({ label: w.week_start.slice(5), value: w.adherence, color: w.adherence >= 80 ? "#0F766E" : w.adherence >= 50 ? "#D97706" : "#B91C1C" }))) : '<p class="muted">No exercises prescribed yet.</p>'}
    <p class="small muted">Sessions done as a percentage of sessions prescribed per week.</p></div>`;
}

const trendText = (label, v) => {
  const c = label === "improving" ? "good" : label === "getting worse" ? "bad" : "";
  return `<span class="pill ${c}">${esc(label)}</span>${v != null ? ` <span class="small muted">${v > 0 ? "+" : ""}${v}/week</span>` : ""}`;
};

function renderAssignments(el, pid, asg, exs) {
  el.innerHTML = `
  <div class="card"><h2 style="margin-top:0">Prescribed exercises</h2>
    <table><thead><tr><th>Exercise</th><th>Side</th><th>Sets × reps</th><th>Per week</th><th>Notes</th><th>Status</th><th></th></tr></thead><tbody>
    ${asg.map((a) => `<tr><td>${esc(a.name)}</td><td>${sideText(a.side) || "–"}</td><td>${a.sets} × ${a.reps}</td><td>${a.per_week}</td><td class="small">${esc(a.notes)}</td>
      <td>${a.active ? '<span class="pill good">active</span>' : '<span class="pill">stopped</span>'}</td>
      <td><button class="btn small secondary" data-tog="${a.id}" data-act="${a.active}">${a.active ? "Stop" : "Restart"}</button></td></tr>`).join("") || '<tr><td colspan="7" class="muted">Nothing prescribed yet</td></tr>'}
    </tbody></table></div>
  <div class="card" style="margin-top:1rem"><h2 style="margin-top:0">Prescribe an exercise</h2>
    <form id="af" class="grid g3">
      <div><label>Exercise</label><select name="exercise" id="exsel">${exs.map((e) => `<option value="${e.key}">${esc(e.name)}</option>`).join("")}</select></div>
      <div id="sidebox"><label>Side</label><select name="side"><option value="r">Right</option><option value="l">Left</option></select></div>
      <div><label>Sets</label><input name="sets" type="number" min="1" max="10" value="2"></div>
      <div><label>Repetitions per set</label><input name="reps" type="number" min="1" max="50" value="10"></div>
      <div><label>Sessions per week</label><input name="per_week" type="number" min="1" max="14" value="5"></div>
      <div style="grid-column:1/-1"><label>Notes for the patient</label><textarea name="notes" rows="2" placeholder="e.g. stop if you feel sharp pain"></textarea></div>
      <div><button class="btn">Add exercise</button></div>
    </form>
    <div id="exprev" class="row" style="margin-top:1rem;align-items:flex-start"></div></div>`;
  const exsel = $("#exsel", el);
  const upd = () => {
    const e = exs.find((x) => x.key === exsel.value);
    $("#sidebox", el).classList.toggle("hide", !e.sided);
    $("#exprev", el).innerHTML = `${e.demo ? `<video src="${e.demo}" autoplay muted loop playsinline style="width:240px;border-radius:10px;background:#000"></video>` : ""}<div style="max-width:420px"><b>${esc(e.name)}</b><p class="small">${esc(e.instructions)}</p><p class="small muted">Camera: ${esc(e.view)}</p></div>`;
  };
  exsel.onchange = upd; upd();
  $("#af", el).onsubmit = async (ev) => {
    ev.preventDefault();
    const fd = Object.fromEntries(new FormData(ev.target));
    ["sets", "reps", "per_week"].forEach((k) => (fd[k] = +fd[k]));
    try { await api(`/api/patients/${pid}/assignments`, { method: "POST", body: fd }); toast("Exercise prescribed"); patientView(pid).then(() => app.querySelector('[data-t="ex"]').click()); }
    catch (e) { toast(e.message); }
  };
  el.querySelectorAll("[data-tog]").forEach((b) => (b.onclick = async () => {
    await api(`/api/assignments/${b.dataset.tog}`, { method: "PATCH", body: { active: b.dataset.act !== "true" } });
    patientView(pid).then(() => app.querySelector('[data-t="ex"]').click());
  }));
}

function renderSessions(el, ss, pid) {
  el.innerHTML = `<div class="card"><h2 style="margin-top:0">Sessions</h2>
    <table><thead><tr><th>Date</th><th>Exercise</th><th>Reps</th><th>Score</th><th>Consistency</th><th>Weakest</th><th>Camera</th></tr></thead><tbody>
    ${ss.map((x) => `<tr class="click" data-id="${x.id}"><td>${fmtDate(x.started)}</td><td>${esc(x.name)}${x.side ? ` (${sideText(x.side)})` : ""}</td><td>${x.reps}/${x.target_reps}</td><td>${scorePill(x.mean_score)}</td>
      <td>${x.consistency == null ? "–" : Math.round(x.consistency)}</td><td>${esc(FACTOR[x.weakest] || "–")}</td><td class="small">${esc(x.camera_view || "–")}${x.source === "video" ? " · video" : ""}</td></tr>`).join("") || '<tr><td colspan="7" class="muted">No sessions yet</td></tr>'}
    </tbody></table></div>`;
  el.querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = () => (location.hash = "#/session/" + tr.dataset.id)));
}

async function sessionView(sid) {
  const x = await api(`/api/sessions/${sid}`);
  const d = x.data || {};
  const reps = d.reps || [];
  const isDoc = state.user.role === "doctor";
  const f = d.summary?.factors || {};
  const traceKeys = Object.keys(d.trace || {});
  app.innerHTML = `
  <div class="row spread"><div><h1>${esc(x.name)}${x.side ? ` <span class="muted">(${sideText(x.side)})</span>` : ""}</h1><div class="muted small">${fmtDate(x.started)} · ${d.duration_s ? Math.round(d.duration_s) + " s" : ""} · camera ${esc(x.camera_view || "–")}${x.source === "video" ? " · from video" : ""}</div></div>
    <button class="btn secondary" onclick="history.back()">Back</button></div>
  <div class="grid g3">
    <div class="card"><div class="stat">${x.reps}<span class="muted" style="font-size:1rem"> / ${x.target_reps}</span></div><div class="stat-label">repetitions</div></div>
    <div class="card"><div class="stat" style="color:${scoreColor(x.mean_score ?? 0)}">${x.mean_score == null ? "–" : Math.round(x.mean_score)}</div><div class="stat-label">average score</div></div>
    <div class="card"><div class="stat">${x.consistency == null ? "–" : Math.round(x.consistency)}</div><div class="stat-label">consistency</div></div>
  </div>
  <div class="grid g2" style="margin-top:1rem">
    <div class="card"><h2 style="margin-top:0">Score per repetition</h2>${barChart(reps.map((r) => ({ label: String(r.n), value: r.score, color: scoreColor(r.score) })))}</div>
    <div class="card"><h2 style="margin-top:0">Score breakdown</h2><table>${Object.entries(f).filter(([, v]) => v != null).map(([k, v]) => `<tr><td>${FACTOR[k] || k}</td><td style="width:55%"><div class="progress"><span style="width:${v}%;background:${scoreColor(v)}"></span></div></td><td>${Math.round(v)}</td></tr>`).join("")}</table>
      <p class="small muted">Movement path, shape and stability make up the score. Range of motion, tempo and symmetry are shown for information.</p></div>
  </div>
  ${traceKeys.length ? `<div class="card" style="margin-top:1rem"><h2 style="margin-top:0">Joint angles during the session</h2>
    ${lineChart(traceKeys.map((k, i) => ({ name: k.replace(/_/g, " "), color: COLORS[i % COLORS.length], points: d.trace[k].map((v, j) => ({ x: j / d.trace_rate, y: v })) })), { ymin: 0, ymax: 180, ylabel: "degrees", dates: false, height: 240 })}
    <p class="small muted">Time in seconds.</p></div>` : ""}
  <div class="card" style="margin-top:1rem"><h2 style="margin-top:0">Repetitions and feedback</h2>
    <table><thead><tr><th>#</th><th>Score</th><th>Duration</th><th>Range of motion</th><th>Feedback given</th></tr></thead><tbody>
    ${reps.map((r) => `<tr><td>${r.n}</td><td>${scorePill(r.score)}</td><td>${r.duration.toFixed(1)} s</td>
      <td class="small">${Object.entries(r.rom || {}).filter(([k]) => (r.ref_rom || {})[k] > 15).map(([k, v]) => `${k.replace(/_/g, " ")} ${Math.round(v)}° <span class="muted">(ref ${Math.round(r.ref_rom[k])}°)</span>`).join("<br>")}</td>
      <td class="small">${r.feedback.map((m) => esc(m.text)).join("<br>")}</td></tr>`).join("")}
    </tbody></table></div>
  ${isDoc ? `<div class="card" style="margin-top:1rem"><h2 style="margin-top:0">Personal baseline</h2>
    <p class="small">If you supervised this session and the patient's form was good, use it as their personal reference for this exercise. Later sessions are then compared with this patient's own movement as well as the general references, and the repetition counter adapts to their range of motion.</p>
    <button class="btn secondary" id="bl">Use this session as baseline</button></div>` : ""}`;
  if (isDoc) $("#bl").onclick = async () => {
    try { const r = await api(`/api/patients/${x.patient_id}/baseline/${sid}`, { method: "POST" }); toast(`Baseline saved (${r.reps} repetitions)`); }
    catch (e) { toast(e.message, 4000); }
  };
}

async function libraryView() {
  const exs = (state.exercises = await api("/api/exercises"));
  app.innerHTML = `<h1>Exercise library</h1><p class="muted">Each exercise comes with a demonstration, instructions and reference repetitions recorded from correct performances (REHAB24-6). Use “Try it” to see live analysis yourself.</p>
  <div class="grid g3">${exs.map((e) => `<div class="card ex-card">${e.demo ? `<video src="${e.demo}" autoplay muted loop playsinline></video>` : ""}
    <h3>${esc(e.name)}</h3><p class="small">${esc(e.instructions)}</p><p class="small muted">Camera: ${esc(e.view)}${e.sided ? " · left or right side" : ""}</p>
    <a class="btn secondary small" href="#/try/${e.key}">Try it</a></div>`).join("")}</div>`;
}

async function tryView(key) {
  const exs = state.exercises.length ? state.exercises : (state.exercises = await api("/api/exercises"));
  const sided = exs.find((e) => e.key === key)?.sided;
  await liveView(app, { exercise: key, side: sided ? "r" : null, target: 10 });
}

// ------------------------------------------------------------------ patient
async function patientHome() {
  const [asg, ss] = await Promise.all([api("/api/my/assignments"), api("/api/my/sessions")]);
  state.exercises = await api("/api/exercises");
  const todayDone = asg.filter((a) => a.done_today > 0).length;
  app.innerHTML = `
  <h1>Hello, ${esc(state.user.full_name.split(" ")[0])}</h1>
  ${state.user.must_change_pw ? `<p class="warn">Please <a href="#/settings">choose your own password</a>.</p>` : ""}
  <p class="muted">${asg.length ? `You have ${asg.length} exercise${asg.length > 1 ? "s" : ""}. Done today: ${todayDone} of ${asg.length}.` : "Your doctor has not prescribed any exercises yet."}</p>
  <div class="grid g3">${asg.map((a) => {
    const e = state.exercises.find((x) => x.key === a.exercise);
    const pct = Math.min(100, Math.round((100 * a.done_this_week) / a.per_week));
    return `<div class="card ex-card">${e?.demo ? `<video src="${e.demo}" autoplay muted loop playsinline></video>` : ""}
      <div class="row spread"><h3>${esc(a.name)}${a.side ? ` (${sideText(a.side)})` : ""}</h3>${a.done_today ? '<span class="pill good">done today</span>' : ""}</div>
      <div class="small">${a.sets} sets × ${a.reps} repetitions · ${a.per_week}× per week</div>
      ${a.notes ? `<div class="small note">${esc(a.notes)}</div>` : ""}
      <div class="small muted">This week: ${a.done_this_week} of ${a.per_week}</div><div class="progress"><span style="width:${pct}%"></span></div>
      <a class="btn" href="#/do/${a.id}">Start exercise</a></div>`;
  }).join("")}</div>
  <h2>Recent sessions</h2>
  <div class="card">${ss.length ? `<table><thead><tr><th>Date</th><th>Exercise</th><th>Reps</th><th>Score</th></tr></thead><tbody>
    ${ss.slice(0, 10).map((x) => `<tr class="click" data-id="${x.id}"><td>${fmtDate(x.started)}</td><td>${esc(x.name)}</td><td>${x.reps}/${x.target_reps}</td><td>${scorePill(x.mean_score)}</td></tr>`).join("")}
  </tbody></table>` : '<p class="muted">No sessions yet.</p>'}</div>`;
  app.querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = () => (location.hash = "#/session/" + tr.dataset.id)));
}

async function doView(aid) {
  const asg = await api("/api/my/assignments");
  const a = asg.find((x) => x.id === aid);
  if (!a) throw new Error("Exercise not found");
  await liveView(app, { assignment: a });
}

async function myProgressView() {
  const prog = await api(`/api/patients/${state.user.id}/progress`);
  app.innerHTML = `<div class="row spread"><h1>My progress</h1><button class="btn secondary" id="pdf">Download report (PDF)</button></div>
    <div class="card">${lineChart(prog.exercises.map((e, i) => ({ name: e.name, color: COLORS[i % COLORS.length], points: e.series.map((p) => ({ x: new Date(p.date), y: p.score })) })), { ylabel: "score" })}</div>
    <div class="grid g3" style="margin-top:1rem">${prog.exercises.map((e) => `<div class="card"><h3>${esc(e.name)}</h3><div class="small">${e.sessions} sessions</div><div style="margin-top:.4rem">${trendText(e.score_trend, e.score_trend_per_week)}</div></div>`).join("")}</div>`;
  $("#pdf").onclick = async () => window.open(URL.createObjectURL(await api(`/api/patients/${state.user.id}/report.pdf`)), "_blank");
}

// ------------------------------------------------------------------ account
async function settingsView() {
  const u = (state.user = await api("/api/me"));
  app.innerHTML = `<h1>Account</h1><div class="grid g2">
  <div class="card"><h2 style="margin-top:0">Password</h2>${u.must_change_pw ? '<p class="warn small">You are using a temporary password. Please choose your own.</p>' : ""}
    <form id="pw"><label>Current password</label><input type="password" name="old_password" required autocomplete="current-password">
    <label>New password (at least 8 characters)</label><input type="password" name="new_password" minlength="8" required autocomplete="new-password">
    <button class="btn" style="margin-top:.8rem">Change password</button></form></div>
  <div class="card"><h2 style="margin-top:0">Two-factor login</h2>
    <p class="small">With two-factor login, signing in also needs a 6-digit code from an authenticator app (Google Authenticator, Microsoft Authenticator, …) on your phone.</p>
    <div id="mfabox">${u.mfa_enabled ? `<p class="pill good">On</p><form id="mfaoff"><label>Code to turn it off</label><input name="code" inputmode="numeric" maxlength="6" required><button class="btn secondary" style="margin-top:.6rem">Turn off</button></form>` : `<p class="pill">Off</p><button class="btn" id="mfaon">Set up two-factor login</button>`}</div></div></div>`;
  $("#pw").onsubmit = async (e) => {
    e.preventDefault();
    try { await api("/api/me/password", { method: "POST", body: Object.fromEntries(new FormData(e.target)) }); toast("Password changed"); state.user.must_change_pw = false; location.hash = "#/"; }
    catch (ex) { toast(ex.message); }
  };
  if ($("#mfaon")) $("#mfaon").onclick = async () => {
    const r = await api("/api/me/mfa/setup", { method: "POST" });
    $("#mfabox").innerHTML = `<p class="small">1. Scan this code with your authenticator app (or enter the key <span class="code">${esc(r.secret)}</span>).</p>
      <img src="data:image/png;base64,${r.qr_png_b64}" alt="QR code for the authenticator app" width="180" height="180">
      <form id="mfaen"><label>2. Enter the 6-digit code it shows</label><input name="code" inputmode="numeric" maxlength="6" required><button class="btn" style="margin-top:.6rem">Turn on</button></form>`;
    $("#mfaen").onsubmit = async (e) => {
      e.preventDefault();
      try { await api("/api/me/mfa/enable", { method: "POST", body: Object.fromEntries(new FormData(e.target)) }); toast("Two-factor login is on"); settingsView(); }
      catch (ex) { toast(ex.message); }
    };
  };
  if ($("#mfaoff")) $("#mfaoff").onsubmit = async (e) => {
    e.preventDefault();
    try { await api("/api/me/mfa/disable", { method: "POST", body: Object.fromEntries(new FormData(e.target)) }); toast("Two-factor login is off"); settingsView(); }
    catch (ex) { toast(ex.message); }
  };
}

router();
