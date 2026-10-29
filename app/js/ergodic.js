// ergodic.js - the ergodicity explorer for the driven Mach-10 run. It reads the central-cell probe straight from the
// binary file the simulation streamed (NTPROBE1, core/src/probe.jl) and the run summary (mach10.json, written by
// core/bench/run_mach10.jl), and compares time statistics at one point with volume statistics over the box.
import { LinePlot, PALETTE } from './viz/plot.js';
import { select, slider, checkbox, readout } from './viz/ui.js';
import { cmapLUT, renderColorbar } from './viz/colormaps.js';
import { loadData } from './explore.js';

const div = (parent, cls, text) => { const e = document.createElement('div'); if (cls) e.className = cls; if (text) e.textContent = text; parent.appendChild(e); return e; };
const fmt = (v, d = 3) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d));

// ---------------------------------------------------------------- NTPROBE1 reader (little-endian)
const probeCache = new Map();
export function loadProbe(file) {
  if (!probeCache.has(file)) {
    const url = new URL(`../data/${file}`, import.meta.url);
    probeCache.set(file, fetch(url).then((r) => { if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`); return r.arrayBuffer(); }).then(parseProbe));
  }
  return probeCache.get(file);
}
export function parseProbe(buf) {
  const dv = new DataView(buf); let o = 0;
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8)); o = 8;
  if (magic !== 'NTPROBE1') throw new Error('not an NTPROBE1 file');
  const version = dv.getUint32(o, true); o += 4;
  const nv = dv.getUint32(o, true); o += 4;
  const vars = [];
  for (let v = 0; v < nv; v++) { vars.push(new TextDecoder().decode(new Uint8Array(buf, o, 8)).replace(/\0+$/, '')); o += 8; }
  const level = dv.getInt32(o, true); o += 4; const every = dv.getInt32(o, true); o += 4;
  const x = [dv.getFloat64(o, true), dv.getFloat64(o + 8, true), dv.getFloat64(o + 16, true)]; o += 24;
  const dx = dv.getFloat64(o, true); o += 8;
  const ml = dv.getUint32(o, true); o += 4;
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, o, ml)) || '{}'); o += ml;
  const rec = 1 + nv, n = Math.floor((buf.byteLength - o) / (8 * rec));
  const raw = new Float64Array(buf.slice(o, o + n * rec * 8));       // copy: the payload offset need not be 8-aligned
  const t = new Float64Array(n); const cols = vars.map(() => new Float64Array(n));
  for (let r = 0; r < n; r++) { t[r] = raw[r * rec]; for (let v = 0; v < nv; v++) cols[v][r] = raw[r * rec + 1 + v]; }
  const col = Object.fromEntries(vars.map((v, k) => [v, cols[k]]));
  return { version, vars, level, every, x, dx, meta, n, t, col, bytes: buf.byteLength };
}

// ---------------------------------------------------------------- statistics
const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
const variance = (a, m = mean(a)) => { let s = 0; for (const v of a) s += (v - m) * (v - m); return s / a.length; };
/** Resample an irregular series (t, y) onto a uniform grid of spacing h by linear interpolation. */
function uniform(t, y, h) {
  const n = Math.max(2, Math.floor((t[t.length - 1] - t[0]) / h) + 1); const out = new Float64Array(n); let j = 0;
  for (let i = 0; i < n; i++) {
    const ti = t[0] + i * h; while (j < t.length - 2 && t[j + 1] < ti) j++;
    const w = (ti - t[j]) / ((t[j + 1] - t[j]) || 1); out[i] = y[j] + Math.min(1, Math.max(0, w)) * (y[j + 1] - y[j]);
  }
  return out;
}
/** Normalised autocorrelation for lags 0..maxLag of a uniformly sampled series. */
function acf(y, maxLag) {
  const m = mean(y), v = variance(y, m), n = y.length, out = new Float64Array(maxLag + 1);
  for (let L = 0; L <= maxLag; L++) { let s = 0; for (let i = 0; i + L < n; i++) s += (y[i] - m) * (y[i + L] - m); out[L] = s / ((n - L) * v || 1); }
  return out;
}
/** Integral time: trapezoid integral of the ACF up to its first zero crossing. */
function tauInt(c, h) { let s = 0; for (let L = 1; L < c.length; L++) { if (c[L] <= 0) break; s += 0.5 * (c[L - 1] + c[L]) * h; } return s; }
function histogram(y, edges) {
  const h = new Float64Array(edges.length - 1), a = edges[0], w = edges[1] - edges[0];
  for (const v of y) { const b = Math.floor((v - a) / w); if (b >= 0 && b < h.length) h[b]++; }
  return h;
}
const centres = (e) => e.slice(0, -1).map((v, k) => 0.5 * (v + e[k + 1]));
const density = (h, w) => { let s = 0; for (const v of h) s += v; return Array.from(h, (v) => v / (s * w || 1)); };
const posOrNull = (a) => a.map((v) => (v > 0 ? v : null));

const VARS = {
  s: { label: 's = ln(ρ/ρ₀)', get: (P) => Array.from(P.col.rho, Math.log), vol: 's' },
  vx: { label: 'v_x / c_s', get: (P) => Array.from(P.col.vx), vol: 'v' },
  vy: { label: 'v_y / c_s', get: (P) => Array.from(P.col.vy), vol: null },
  B: { label: '|B| / B₀', get: (P, B0) => Array.from(P.col.Bx, (bx, k) => Math.hypot(bx, P.col.By[k], P.col.Bz[k]) / B0), vol: null },
};
export const ERGO_VARS = Object.keys(VARS);

// ---------------------------------------------------------------- the explorer
export async function mountErgo(host, S, onChange = () => {}) {
  const [{ data: D, meta }, P] = await Promise.all([loadData('mach10'), loadData('mach10').then((r) => loadProbe(r.data.params.probe_file))]);
  const tc = D.params.t_cross, B0 = Math.sqrt(2 / D.params.beta) * D.params.cs;
  const bar = div(host, 'qd-controls');
  const bar2 = div(host, 'qd-controls');
  const g1 = div(host, 'qd-grid2');
  const pMap = div(g1, 'qd-panel'); const pSer = div(g1, 'qd-panel');
  const g2 = div(host, 'qd-grid2');
  const pPdf = div(g2, 'qd-panel'); const pAcf = div(g2, 'qd-panel');
  const pRun = div(host, 'qd-panel');
  const tab = div(host);

  // --- density map with the probe marked
  div(pMap, 'qd-title', 'log₁₀ ρ, all three levels composited; the cross marks the probe cell');
  const wrap = div(pMap, 'qd-with-bar'); const stage = div(wrap, 'qd-stage');
  const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 512; stage.appendChild(canvas);
  canvas.setAttribute('aria-label', 'Density of the driven Mach-10 run with the probe position');
  const cbar = document.createElement('canvas'); cbar.className = 'qd-colorbar'; wrap.appendChild(cbar);
  const lut = cmapLUT('magma');
  function drawMap() {
    const F = D.frames; const fi = Math.min(S.frame, F.t.length - 1); const levs = F.levels[fi];
    const ctx = canvas.getContext('2d'); ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, 512, 512);
    const lo = -3, hi = 2;
    for (const L of levs) {
      const [nx, ny] = L.n, img = new ImageData(nx, ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const v = Math.min(1, Math.max(0, (L.logrho[i + nx * j] - lo) / (hi - lo)));
        const c = Math.round(v * 255) * 4, p = ((ny - 1 - j) * nx + i) * 4;
        img.data[p] = lut[c]; img.data[p + 1] = lut[c + 1]; img.data[p + 2] = lut[c + 2]; img.data[p + 3] = 255;
      }
      const off = document.createElement('canvas'); off.width = nx; off.height = ny; off.getContext('2d').putImageData(img, 0, 0);
      const s = 512 / L.Nlev[0];
      const x0 = (L.lo[0] - 1) * s, y0 = (L.lo[1] - 1) * s, w = nx * s, h = ny * s;
      ctx.drawImage(off, x0, 512 - y0 - h, w, h);
      if (S.outline && L !== levs[0]) { ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, 512 - y0 - h + 0.5, w - 1, h - 1); }
    }
    const px = P.x[0] * 512, py = 512 - P.x[1] * 512;
    ctx.strokeStyle = '#00e0ff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(px - 9, py); ctx.lineTo(px + 9, py); ctx.moveTo(px, py - 9); ctx.lineTo(px, py + 9); ctx.stroke();
    renderColorbar(cbar, 'magma', lo, hi, { label: 'log₁₀ ρ' });
    tRead.set(`${F.t[fi].toFixed(2)} = ${(F.t[fi] / tc).toFixed(1)} t_c`);
  }

  // --- probe time series and Mach history
  const sPlot = new LinePlot(div(pSer), { width: 470, height: 210 });
  const mPlot = new LinePlot(div(pSer), { width: 470, height: 170 });
  const pdfPlot = new LinePlot(pPdf, { width: 470, height: 320 });
  const acfPlot = new LinePlot(pAcf, { width: 470, height: 320 });
  const runPlot = new LinePlot(pRun, { width: 960, height: 280 });

  function draw() {
    const V = VARS[S.evar]; const y = V.get(P, B0); const t = P.t;
    const t0 = S.w0 * tc, t1 = Math.min(t[t.length - 1], (S.w0 + S.wl) * tc);
    const inW = []; const tW = []; for (let k = 0; k < t.length; k++) if (t[k] >= t0 && t[k] <= t1) { inW.push(y[k]); tW.push(t[k]); }
    // series (thinned for drawing)
    const st = Math.max(1, Math.floor(t.length / 3000)); const tx = [], yx = [];
    for (let k = 0; k < t.length; k += st) { tx.push(t[k] / tc); yx.push(y[k]); }
    let ymin = Infinity, ymax = -Infinity; for (const v of yx) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
    sPlot.setAxes({ xlabel: 't / t_c', ylabel: V.label, title: `Probe cell (${P.x[0].toFixed(4)}, ${P.x[1].toFixed(4)}), every ${P.every}nd finest step — window shaded` });
    sPlot.setSeries([
      { x: [S.w0, S.w0, Math.min(S.w0 + S.wl, t1 / tc), Math.min(S.w0 + S.wl, t1 / tc), S.w0], y: [ymin, ymax, ymax, ymin, ymin], color: '#bbb', width: 1, dash: '3 3' },
      { x: tx, y: yx, width: 0.8, color: PALETTE[0] },
    ]);
    sPlot.draw();
    const Mt = D.mach.t.map((v) => v / tc);
    mPlot.setAxes({ xlabel: 't / t_c', ylabel: 'rms Mach (root)', title: `Feedback target M = ${D.params.mach}` });
    mPlot.setSeries([
      { x: Mt, y: D.mach.mass, label: 'mass-weighted (controlled)', color: PALETTE[1] },
      { x: Mt, y: D.mach.vol, label: 'volume-weighted', color: PALETTE[2], width: 1 },
      { x: [0, Mt[Mt.length - 1]], y: [D.params.mach, D.params.mach], color: '#999', dash: '4 3' },
    ]);
    mPlot.draw();

    // statistics in the window
    const h = (tW[tW.length - 1] - tW[0]) / Math.max(1, tW.length - 1);
    const yu = tW.length > 2 ? uniform(tW, inW, h) : Float64Array.from(inW);
    const m = mean(yu), v = variance(yu, m);
    const maxLag = Math.min(yu.length - 2, Math.round(2 * tc / h));
    const c = acf(yu, Math.max(1, maxLag)); const ti = tauInt(c, h);
    const T = (tW[tW.length - 1] - tW[0]); const neff = T / (2 * ti || 1); const se = Math.sqrt(v / Math.max(1, neff));

    // volume PDFs averaged over the snapshots inside the window
    const pdf = D.pdf; let volF = null, volR = null, edges = null, volMeanF = null, volMeanR = null, volVarF = null;
    if (V.vol) {
      edges = V.vol === 's' ? pdf.s_edges : pdf.v_edges;
      const F = V.vol === 's' ? pdf.fine_s : pdf.fine_v, R = V.vol === 's' ? pdf.root_s : pdf.root_v;
      const sumF = new Float64Array(edges.length - 1), sumR = new Float64Array(edges.length - 1); let ns = 0;
      pdf.t.forEach((tt, k) => { if (tt >= t0 - 1e-9 && tt <= t1 + 1e-9) { ns++; F[k].forEach((v2, b) => { sumF[b] += v2; }); R[k].forEach((v2, b) => { sumR[b] += v2; }); } });
      if (ns) {
        const w = edges[1] - edges[0], xc = centres(edges);
        volF = density(sumF, w); volR = density(sumR, w);
        volMeanF = xc.reduce((a, x, b) => a + x * volF[b] * w, 0); volMeanR = xc.reduce((a, x, b) => a + x * volR[b] * w, 0);
        volVarF = xc.reduce((a, x, b) => a + (x - volMeanF) ** 2 * volF[b] * w, 0);
      }
    }
    if (!edges) { const lo = Math.min(...inW), hi = Math.max(...inW); const nb = 60; const w = (hi - lo) / nb || 1; edges = Array.from({ length: nb + 1 }, (_, k) => lo + k * w); }
    const w = edges[1] - edges[0], xc = centres(edges);
    const tp = density(histogram(inW, edges), w);
    const ser = [{ x: xc, y: S.log ? posOrNull(tp) : tp, label: 'time PDF at the probe (window)', color: PALETTE[0], width: 2 }];
    if (volF) {
      ser.push({ x: xc, y: S.log ? posOrNull(volF) : volF, label: 'volume PDF, finest level', color: PALETTE[1] });
      ser.push({ x: xc, y: S.log ? posOrNull(volR) : volR, label: 'volume PDF, whole box (root)', color: PALETTE[2], dash: '5 3' });
    }
    if (S.evar === 's' && volVarF) {   // lognormal with the volume variance and the mass-conserving mean −σ²/2
      const s2 = volVarF; const g = xc.map((x) => Math.exp(-((x + s2 / 2) ** 2) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2));
      ser.push({ x: xc, y: S.log ? posOrNull(g) : g, label: `lognormal, σ_s² = ${s2.toFixed(2)}, mean −σ²/2`, color: '#999', dash: '2 3' });
    }
    pdfPlot.setAxes({ ylog: S.log, xlabel: V.label, ylabel: 'probability density', title: `One point in time vs the whole volume, ${S.w0.toFixed(1)}–${(t1 / tc).toFixed(1)} t_c` });
    pdfPlot.setSeries(ser); pdfPlot.draw();

    const lags = Array.from(c, (_, L) => (L * h) / tc);
    acfPlot.setAxes({ xlabel: 'lag / t_c', ylabel: 'autocorrelation', title: `ACF at the probe: τ_int = ${(ti / tc).toFixed(3)} t_c, N_eff ≈ ${neff.toFixed(1)}` });
    acfPlot.setSeries([{ x: lags, y: Array.from(c), color: PALETTE[0], width: 2 }, { x: [0, lags[lags.length - 1]], y: [0, 0], color: '#999', dash: '3 3' }]);
    acfPlot.draw();

    // running time average with its error band, against the volume average
    const n = yu.length, rx = [], rm = [], up = [], dn = []; let s = 0;
    const stp = Math.max(1, Math.floor(n / 1500));
    for (let k = 0; k < n; k++) {
      s += yu[k];
      if (k % stp === 0 || k === n - 1) {
        const Tk = (k + 1) * h, mk = s / (k + 1), e = Math.sqrt(v / Math.max(1, Tk / (2 * ti || 1)));
        rx.push(S.w0 + Tk / tc); rm.push(mk); up.push(mk + e); dn.push(mk - e);
      }
    }
    const rser = [
      { x: rx, y: rm, label: 'running time average at the probe', color: PALETTE[0], width: 2 },
      { x: rx, y: up, label: '± σ / √N_eff(T)', color: PALETTE[0], width: 1, dash: '3 3' },
      { x: rx, y: dn, color: PALETTE[0], width: 1, dash: '3 3' },
    ];
    if (volMeanF !== null) {
      rser.push({ x: [rx[0], rx[rx.length - 1]], y: [volMeanF, volMeanF], label: 'volume average, finest level (window mean)', color: PALETTE[1], dash: '6 3' });
      rser.push({ x: [rx[0], rx[rx.length - 1]], y: [volMeanR, volMeanR], label: 'volume average, whole box (window mean)', color: PALETTE[2], dash: '2 3' });
    }
    runPlot.setAxes({ xlabel: 't / t_c (end of the averaging interval)', ylabel: `⟨${V.label}⟩`, title: 'Does the time average converge to the volume average?' });
    runPlot.setSeries(rser); runPlot.draw();

    tab.innerHTML = '';
    const tb = document.createElement('table'); tb.className = 'table table-sm qd-table';
    const rows = [
      ['window', `${S.w0.toFixed(1)}–${(t1 / tc).toFixed(1)} t_c (${inW.length} probe records)`],
      ['time mean ± standard error', `${fmt(m)} ± ${fmt(se)}`],
      ['volume mean, finest level / whole box', volMeanF === null ? 'not stored for this variable' : `${fmt(volMeanF)} / ${fmt(volMeanR)}`],
      ['(time − volume) / standard error', volMeanF === null ? '—' : `${fmt((m - volMeanF) / se, 2)} (finest), ${fmt((m - volMeanR) / se, 2)} (box)`],
      ['time variance / volume variance (finest)', volVarF === null ? '—' : `${fmt(v)} / ${fmt(volVarF)}`],
      ['τ_int, N_eff = T / 2τ_int', `${fmt(ti / tc)} t_c, ${fmt(neff, 1)}`],
      ['probe file', `${P.n} records × ${P.vars.length + 1} Float64, ${(P.bytes / 1e6).toFixed(2)} MB, level ${P.level}, dx = ${P.dx.toExponential(3)}`],
    ];
    const body = tb.createTBody(); rows.forEach((r) => { const tr = body.insertRow(); r.forEach((x) => { tr.insertCell().textContent = x; }); });
    tab.appendChild(tb);
    drawMap();
    onChange(S);
  }
  select(bar, { label: 'variable', options: ERGO_VARS.map((k) => ({ value: k, label: VARS[k].label })), value: S.evar, onChange: (x) => { S.evar = x; draw(); } });
  const tmax = P.t[P.n - 1] / tc;
  slider(bar, { label: 'window start [t_c]', min: 0, max: Math.max(0, tmax - 0.5), step: 0.1, value: S.w0, onInput: (x) => { S.w0 = x; draw(); } });
  slider(bar, { label: 'window length [t_c]', min: 0.5, max: tmax, step: 0.1, value: S.wl, onInput: (x) => { S.wl = x; draw(); } });
  checkbox(bar, { label: 'log PDF', value: S.log, onChange: (x) => { S.log = x; draw(); } });
  slider(bar2, { label: 'map frame', min: 0, max: D.frames.t.length - 1, step: 1, value: Math.min(S.frame, D.frames.t.length - 1), format: (x) => String(Math.round(x)), onInput: (x) => { S.frame = Math.round(x); drawMap(); onChange(S); } });
  checkbox(bar2, { label: 'patch outlines', value: S.outline, onChange: (x) => { S.outline = x; drawMap(); onChange(S); } });
  const tRead = readout(bar2, { label: 't =' });
  draw();
  div(host, 'qd-note', `Run: root ${D.params.N}², 3 levels, CT + HLL, isothermal, β = ${D.params.beta}, ${D.params.modes} solenoidal modes ${D.params.kmin} ≤ |k| ≤ ${D.params.kmax}, seed ${D.params.seed}; steps per level ${D.steps.join(' / ')}; wall ${(D.wall / 60).toFixed(0)} min. Computed ${meta.generated} on ${meta.cpu}, ${meta.threads} thread(s).`);
  return { draw };
}
