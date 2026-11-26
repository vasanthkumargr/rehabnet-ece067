// Shared helpers: API calls, escaping, small UI pieces, and SVG charts.

export const state = { token: sessionStorage.getItem("rn_token") || "", user: null, exercises: [] };

export async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (state.token) headers.Authorization = "Bearer " + state.token;
  const res = await fetch(path, { ...opts, headers, body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body });
  if (res.status === 401 && !path.startsWith("/api/auth")) { logout(); throw new Error("Please log in again"); }
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("json") ? await res.json() : await res.blob();
  if (!res.ok) throw new Error((data && data.detail && (typeof data.detail === "string" ? data.detail : data.detail[0]?.msg)) || "Something went wrong");
  return data;
}

export function setToken(t) { state.token = t; if (t) sessionStorage.setItem("rn_token", t); else sessionStorage.removeItem("rn_token"); }
export function logout() { setToken(""); state.user = null; location.hash = "#/login"; }

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const $ = (sel, el = document) => el.querySelector(sel);
export const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

export function toast(msg, ms = 2600) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), ms);
}

export function scorePill(v) {
  if (v === null || v === undefined) return '<span class="pill">–</span>';
  const c = v >= 80 ? "good" : v >= 60 ? "mid" : "bad";
  return `<span class="pill ${c}">${Math.round(v)}</span>`;
}
export const scoreColor = (v) => (v >= 80 ? "#0F766E" : v >= 60 ? "#D97706" : "#B91C1C");
export const scoreBg = (v) => (v >= 80 ? "#E6F4F2" : v >= 60 ? "#FEF3E2" : "#FDECEC");

export function fmtDate(iso, withTime = true) {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" }) + (withTime ? " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }) : "");
}
export const sideText = (s) => ({ l: "left", r: "right" }[s] || "");
export const FACTOR = { posture: "Movement path", similarity: "Movement shape", rom: "Range of motion", stability: "Stability", tempo: "Tempo", symmetry: "Symmetry" };

export function modal(html) {
  const back = document.createElement("div");
  back.className = "modal-back";
  back.innerHTML = `<div class="modal" role="dialog" aria-modal="true">${html}</div>`;
  document.body.appendChild(back);
  const close = () => back.remove();
  back.addEventListener("click", (e) => { if (e.target === back) close(); });
  back.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", close));
  return { el: back.firstElementChild, close };
}

// ------------------------------------------------------------------ charts (plain SVG)
export function lineChart(series, { height = 220, ymin = 0, ymax = 100, ylabel = "", dates = true } = {}) {
  // series: [{name, color, points:[{x: Date|number, y: number}]}]
  const W = 720, H = height, L = 42, R = 30, T = 12, B = 30;
  const xs = series.flatMap((s) => s.points.map((p) => +p.x));
  if (!xs.length) return '<p class="muted">No data yet.</p>';
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (x1 === x0) { x0 -= 86400000; x1 += 86400000; }
  const sx = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const sy = (y) => T + (1 - (y - ymin) / (ymax - ymin)) * (H - T - B);
  let g = "";
  for (let v = ymin; v <= ymax; v += (ymax - ymin) / 4) {
    g += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="#e2e8f0"/><text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end" font-size="11" fill="#64748b">${Math.round(v)}</text>`;
  }
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const x = x0 + ((x1 - x0) * i) / ticks;
    const lab = dates ? new Date(x).toLocaleDateString(undefined, { day: "numeric", month: "short" }) : Math.round(x);
    g += `<text x="${sx(x)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="#64748b">${lab}</text>`;
  }
  let paths = "", legend = "";
  series.forEach((s, i) => {
    const pts = s.points.filter((p) => p.y !== null && p.y !== undefined && isFinite(p.y));
    if (!pts.length) return;
    paths += `<polyline fill="none" stroke="${s.color}" stroke-width="${pts.length > 60 ? 1.6 : 2.4}" stroke-linejoin="round" points="${pts.map((p) => `${sx(+p.x)},${sy(p.y)}`).join(" ")}"/>`;
    if (pts.length <= 60) paths += pts.map((p) => `<circle cx="${sx(+p.x)}" cy="${sy(p.y)}" r="3.6" fill="${s.color}"><title>${esc(s.name)}: ${Math.round(p.y)}</title></circle>`).join("");
    legend += `<span class="row" style="gap:.3rem"><span style="width:12px;height:12px;border-radius:3px;background:${s.color};display:inline-block"></span>${esc(s.name)}</span>`;
  });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} over time">${g}${paths}
    <text x="12" y="${T + 6}" font-size="11" fill="#64748b" transform="rotate(-90 12 ${H / 2}) translate(-${H / 2 - 20},0)">${esc(ylabel)}</text></svg>
    <div class="row small" style="margin-top:.3rem">${legend}</div>`;
}

export function barChart(items, { height = 160, max = 100, color = "#0F766E" } = {}) {
  // items: [{label, value}]
  const W = 720, H = height, B = 26, T = 10;
  if (!items.length) return "";
  const bw = (W - 20) / items.length;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">${items.map((it, i) => {
    const h = it.value == null ? 0 : (Math.min(it.value, max) / max) * (H - B - T);
    const x = 10 + i * bw;
    const bwi = Math.min(bw * 0.7, 56);
    return `<rect x="${x + (bw - bwi) / 2}" y="${H - B - h}" width="${bwi}" height="${h}" rx="4" fill="${it.color || color}"><title>${esc(it.label)}: ${it.value ?? "–"}</title></rect>
      <text x="${x + bw / 2}" y="${H - 8}" text-anchor="middle" font-size="11" fill="#64748b">${esc(it.label)}</text>
      ${it.value != null ? `<text x="${x + bw / 2}" y="${H - B - h - 4}" text-anchor="middle" font-size="11" fill="#334155">${Math.round(it.value)}</text>` : ""}`;
  }).join("")}</svg>`;
}
