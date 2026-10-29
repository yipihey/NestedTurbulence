// explore.js - result explorers for the NestedMHD experiments. Every explorer mounts into a host element, reads a
// precomputed dataset from app/data/<name>.json (written by core/bench/run_experiments.jl), and reports state edits
// through onChange(S) so the laboratory can keep the URL hash in sync. The chapter widgets mount the same functions.
import { LinePlot, PALETTE } from './viz/plot.js';
import { select, slider, checkbox, readout } from './viz/ui.js';
import { cmapLUT, renderColorbar } from './viz/colormaps.js';
import { mountErgo } from './ergodic.js';

const cache = new Map();
export function loadData(name) {
  if (!cache.has(name)) {
    const url = new URL(`../data/${name}.json`, import.meta.url);
    cache.set(name, fetch(url).then((r) => { if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status}`); return r.json(); }));
  }
  return cache.get(name);
}

const div = (parent, cls, text) => { const e = document.createElement('div'); if (cls) e.className = cls; if (text) e.textContent = text; parent.appendChild(e); return e; };
const note = (parent, text) => div(parent, 'qd-note', text);
const fmtE = (v) => (v === null || v === undefined ? '—' : Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toPrecision(3));
const MODE_OPTS = [{ value: 'ct', label: 'CT (face B, GS05 EMF)' }, { value: 'glm', label: 'GLM (Dedner cleaning)' }];
function metaLine(host, meta) {
  note(host, `Computed ${meta.generated} on ${meta.cpu}, ${meta.threads} Julia thread(s), Julia ${meta.julia}, KernelAbstractions CPU backend.`);
}
function table(host, header, rows) {
  const t = document.createElement('table'); t.className = 'table table-sm qd-table';
  const th = t.createTHead().insertRow(); header.forEach((h) => { const c = document.createElement('th'); c.textContent = h; th.appendChild(c); });
  const tb = t.createTBody(); rows.forEach((r) => { const tr = tb.insertRow(); r.forEach((v) => { tr.insertCell().textContent = v; }); });
  host.appendChild(t); return t;
}
const order = (a, b) => Math.log2(a / b);

// ---------------------------------------------------------------- ghost-zone time accuracy
export async function mountGhost(host, S, onChange = () => {}) {
  const { data, meta } = await loadData('ghost');
  const bar = div(host, 'qd-controls'); const ph = div(host); const tab = div(host);
  const plot = new LinePlot(ph, { width: 620, height: 360 });
  const draw = () => {
    const sel = (g) => data.filter((r) => r.mode === S.mode && r.courant === S.courant && r.ghosts === g).sort((a, b) => a.N - b.N);
    const I = sel('interp'), St = sel('stale');
    const N = I.map((r) => r.N);
    const ref = (p, r0) => N.map((n) => r0 * Math.pow(N[0] / n, p));
    plot.setAxes({ xlog: true, ylog: true, xlabel: 'root cells N (child 2N)', ylabel: 'max |ghost − exact|', title: `Child ghost zones vs the exact wave (${S.mode.toUpperCase()}, Courant ${S.courant})` });
    plot.setSeries([
      { x: N, y: I.map((r) => r.err), label: 'θ-interpolated ghosts', points: true },
      { x: N, y: St.map((r) => r.err), label: 'stale ghosts (θ ≡ 0)', points: true },
      { x: N, y: ref(2, I[0].err), label: '∝ N⁻²', dash: '4 3', color: '#999' },
      { x: N, y: ref(1, St[0].err), label: '∝ N⁻¹', dash: '1 3', color: '#999' },
    ]);
    plot.draw();
    tab.innerHTML = '';
    table(tab, ['N', 'θ-interp', 'order', 'stale', 'order'], I.map((r, k) => [r.N, fmtE(r.err), k ? order(I[k - 1].err, r.err).toFixed(2) : '', fmtE(St[k].err), k ? order(St[k - 1].err, St[k].err).toFixed(2) : '']));
    onChange(S);
  };
  select(bar, { label: 'divergence control', options: MODE_OPTS, value: S.mode, onChange: (v) => { S.mode = v; draw(); } });
  select(bar, { label: 'Courant number', options: [{ value: '0.4', label: '0.4' }, { value: '0.8', label: '0.8' }], value: String(S.courant), onChange: (v) => { S.courant = +v; draw(); } });
  draw(); metaLine(host, meta);
  return { draw, plot };
}

// ---------------------------------------------------------------- convergence against the analytic Alfvén wave
export async function mountConv(host, S, onChange = () => {}) {
  const { data, meta } = await loadData('convergence');
  const bar = div(host, 'qd-controls'); const ph = div(host); const tab = div(host);
  const plot = new LinePlot(ph, { width: 620, height: 360 });
  const draw = () => {
    const sel = (k) => data.filter((r) => r.mode === S.mode && r.courant === S.courant && r.kind === k).sort((a, b) => a.N - b.N);
    const I = sel('interp'), St = sel('stale'), U = sel('uniform');
    const N = U.map((r) => r.N);
    plot.setAxes({ xlog: true, ylog: true, xlabel: 'root cells N', ylabel: 'L1 error of B_y after one period', title: `Circularly polarised Alfvén wave (${S.mode.toUpperCase()}, Courant ${S.courant})` });
    plot.setSeries([
      { x: N, y: U.map((r) => r.l1_root), label: 'uniform N (no refinement)', points: true, color: '#999' },
      { x: N, y: I.map((r) => r.l1_root), label: 'nested, θ-interp: root', points: true, color: PALETTE[0] },
      { x: N, y: I.map((r) => r.l1_child), label: 'nested, θ-interp: child', points: true, color: PALETTE[0], dash: '5 3' },
      { x: N, y: St.map((r) => r.l1_root), label: 'nested, stale: root', points: true, color: PALETTE[1] },
      { x: N, y: St.map((r) => r.l1_child), label: 'nested, stale: child', points: true, color: PALETTE[1], dash: '5 3' },
    ]);
    plot.draw();
    tab.innerHTML = '';
    table(tab, ['N', 'uniform', 'θ root', 'θ child', 'stale root', 'stale child', 'steps (θ)', '|Δm/m|', 'max ∇·B'],
      I.map((r, k) => [r.N, fmtE(U[k].l1_root), fmtE(r.l1_root), fmtE(r.l1_child), fmtE(St[k].l1_root), fmtE(St[k].l1_child), r.steps.join(' / '), fmtE(r.dmass), fmtE(r.divb)]));
    onChange(S);
  };
  select(bar, { label: 'divergence control', options: MODE_OPTS, value: S.mode, onChange: (v) => { S.mode = v; draw(); } });
  select(bar, { label: 'Courant number', options: [{ value: '0.4', label: '0.4' }, { value: '0.8', label: '0.8' }], value: String(S.courant), onChange: (v) => { S.courant = +v; draw(); } });
  draw(); metaLine(host, meta);
  return { draw, plot };
}

// ---------------------------------------------------------------- adaptive vs fixed ratio-2 subcycling
export async function mountAdapt(host, S, onChange = () => {}) {
  const { data, meta } = await loadData('adaptive');
  const bar = div(host, 'qd-controls'); const ph = div(host); const tab = div(host);
  const plot = new LinePlot(ph, { width: 620, height: 340 });
  const draw = () => {
    const a = data.find((r) => r.problem === S.problem && r.mode === S.mode && r.adaptive);
    const f = data.find((r) => r.problem === S.problem && r.mode === S.mode && !r.adaptive);
    const L = a.steps.map((_, k) => k + 1);
    plot.setAxes({ ylog: true, xlabel: 'level', ylabel: 'steps to t = 0.05', title: `Steps per level, 5-level nest (${S.problem}, ${S.mode.toUpperCase()})` });
    plot.setSeries([
      { x: L, y: a.steps, label: 'adaptive (per-level CFL)', points: true },
      { x: L, y: f.steps, label: 'fixed ratio 2', points: true },
    ]);
    plot.draw();
    tab.innerHTML = '';
    table(tab, ['', 'steps per level', 'cell updates', '|Δm/m|', 'max ∇·B (rel.)'], [
      ['adaptive', a.steps.join(' / '), a.cell_updates.toLocaleString(), fmtE(a.dmass), fmtE(a.divb)],
      ['fixed ratio 2', f.steps.join(' / '), f.cell_updates.toLocaleString(), fmtE(f.dmass), fmtE(f.divb)],
    ]);
    note(tab, `Adaptive / fixed cell updates = ${(a.cell_updates / f.cell_updates).toFixed(2)}.`);
    onChange(S);
  };
  select(bar, { label: 'problem', options: [{ value: "cold-core", label: "cold dense core (c_s,core = c_s/10)" }, { value: 'orszag-tang', label: 'Orszag–Tang (uniform speeds)' }], value: S.problem, onChange: (v) => { S.problem = v; draw(); } });
  select(bar, { label: 'divergence control', options: MODE_OPTS, value: S.mode, onChange: (v) => { S.mode = v; draw(); } });
  draw(); metaLine(host, meta);
  return { draw, plot };
}

// ---------------------------------------------------------------- covered-region skipping
export async function mountSkip(host, S, onChange = () => {}) {
  const { data, meta } = await loadData('skip');
  const tab = div(host);
  const rows = [];
  for (const mode of ['ct', 'glm']) {
    const on = data.find((r) => r.mode === mode && r.skip), off = data.find((r) => r.mode === mode && !r.skip);
    rows.push([mode.toUpperCase(), on.steps.join(' / '), on.wall.toFixed(2), off.steps.join(' / '), off.wall.toFixed(2),
               ((1 - on.wall / off.wall) * 100).toFixed(0) + ' %']);
  }
  table(tab, ['mode', 'steps (skip)', 'wall s (skip)', 'steps (no skip)', 'wall s (no skip)', 'saved'], rows);
  note(tab, '3-level nest, 32³ cells per level, Float32, Orszag–Tang (3-D perturbed) to t = 0.03, after a warm-up run.');
  metaLine(host, meta); onChange(S);
  return { draw() {} };
}

// ---------------------------------------------------------------- throughput
export async function mountTput(host, S, onChange = () => {}) {
  const { data, meta } = await loadData('throughput');
  table(div(host), ['mode', 'precision', 'Riemann', 'grid', 'Mcell/s'],
    data.map((r) => [r.mode.toUpperCase(), r.T, r.riemann.toUpperCase(), `${r.N}³`, r.mcells.toFixed(3)]));
  note(host, 'Single periodic patch, cell updates per second of wall time, portable KernelAbstractions kernels on the CPU backend. This is not a GPU figure.');
  metaLine(host, meta); onChange(S);
  return { draw() {} };
}

// ---------------------------------------------------------------- Orszag-Tang reference check
export async function mountOtref(host, S, onChange = () => {}) {
  const { data, meta } = await loadData('otref');
  const ref = { 64: 0.7887, 128: 0.7906 };
  table(div(host), ['mode', 'Riemann', 'N', 'v_rms(t=0.2)', 'GLMMHDTurb (GLM, HLLD)', 'max ∇·B (rel.)'],
    data.map((r) => [r.mode.toUpperCase(), r.riemann.toUpperCase(), r.N, r.vrms.toFixed(4), ref[r.N] ? ref[r.N].toFixed(4) : '—', fmtE(r.divb)]));
  note(host, 'Quasi-2-D Orszag–Tang (N × N × 4), γ = 5/3, Courant 0.7, c_h scale 0.25 — the GLMMHDTurb defaults. Reference values from the GLMMHDTurb README (64³ and 128³ runs).');
  metaLine(host, meta); onChange(S);
  return { draw() {} };
}

// ---------------------------------------------------------------- nested Orszag-Tang field viewer
export async function mountOT(host, S, onChange = () => {}) {
  const { data, meta } = await loadData('otnested');
  const bar = div(host, 'qd-controls');
  const grid = div(host, 'qd-grid2');
  const left = div(grid, 'qd-panel'); const right = div(grid, 'qd-panel');
  div(left, 'qd-title', 'z-midplane slice, all levels composited (finest data wins)');
  const wrap = div(left, 'qd-with-bar'); const stage = div(wrap, 'qd-stage');
  const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 512; stage.appendChild(canvas);
  canvas.setAttribute('aria-label', 'Composited field slice of the nested hierarchy');
  const cbar = document.createElement('canvas'); cbar.className = 'qd-colorbar'; wrap.appendChild(cbar);
  div(right, 'qd-title', 'conservation and ∇·B during the run');
  const ph1 = div(right); const ph2 = div(right);
  const p1 = new LinePlot(ph1, { width: 460, height: 230 });
  const p2 = new LinePlot(ph2, { width: 460, height: 230 });
  const zeroNote = div(right, 'qd-note');
  const info = readout(bar, { label: 't =' });
  const draw = () => {
    const run = data[S.mode]; const fr = run.frames[Math.min(S.frame, run.frames.length - 1)];
    info.set(`${fr.t.toFixed(2)}   steps ${run.series[Math.min(S.frame, run.series.length - 1)].steps.join(' / ')}`);
    // value range over all levels of this frame
    let lo = Infinity, hi = -Infinity;
    for (const L of fr.levels) for (const col of L[S.field]) for (const v of col) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const tf = S.log ? (v) => Math.log10(Math.max(v, 1e-12)) : (v) => v;
    const a = tf(Math.max(lo, S.log ? 1e-12 : lo)), b = tf(hi);
    const lut = cmapLUT(S.cmap);
    const ctx = canvas.getContext('2d'); ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, 512, 512);
    const box = fr.levels[0].n[0] * fr.levels[0].dx;       // box length (root covers it)
    for (const L of fr.levels) {
      const nx = L.n[0], ny = L.n[1], arr = L[S.field];
      const img = new ImageData(nx, ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const t = Math.min(1, Math.max(0, (tf(arr[j][i]) - a) / (b - a || 1)));
        const c = Math.round(t * 255) * 4, p = ((ny - 1 - j) * nx + i) * 4;     // y up
        img.data[p] = lut[c]; img.data[p + 1] = lut[c + 1]; img.data[p + 2] = lut[c + 2]; img.data[p + 3] = 255;
      }
      const off = document.createElement('canvas'); off.width = nx; off.height = ny; off.getContext('2d').putImageData(img, 0, 0);
      const x0 = (L.lo[0] - 1) * L.dx / box * 512, y0 = (L.lo[1] - 1) * L.dx / box * 512, w = nx * L.dx / box * 512, h = ny * L.dx / box * 512;
      ctx.drawImage(off, x0, 512 - y0 - h, w, h);
      if (S.outline && L !== fr.levels[0]) { ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1.5; ctx.strokeRect(x0 + 0.5, 512 - y0 - h + 0.5, w - 1, h - 1); }
    }
    const labels = { rho: 'density ρ', pmag: 'magnetic pressure |B|²/2', speed: 'speed |v_xy|' };
    renderColorbar(cbar, S.cmap, S.log ? a : lo, S.log ? b : hi, { label: (S.log ? 'log₁₀ ' : '') + labels[S.field] });
    const T = run.series.map((s) => s.t);
    const nl = run.series[0].divb.length;
    p1.setAxes({ ylog: true, xlabel: 't', ylabel: 'max |∇·B| dx / max|B|', title: `∇·B per level (${S.mode.toUpperCase()})` });
    p1.setSeries(Array.from({ length: nl }, (_, l) => ({ x: T, y: run.series.map((s) => Math.max(s.divb[l], 1e-17)), label: `level ${l + 1}`, points: true })));
    p1.draw();
    p2.setAxes({ ylog: true, xlabel: 't', ylabel: 'relative change', title: 'Total mass and energy (whole box)' });
    p2.setSeries([
      { x: T, y: run.series.map((s) => Math.max(Math.abs(s.dmass), 1e-17)), label: '|Δ mass| / mass', points: true },
      { x: T, y: run.series.map((s) => Math.max(Math.abs(s.denergy), 1e-17)), label: '|Δ energy| / energy', points: true },
    ]);
    p2.draw();
    zeroNote.textContent = 'Log axes: values that are exactly 0 are drawn at 1e-17.';
    onChange(S);
  };
  select(bar, { label: 'divergence control', options: MODE_OPTS, value: S.mode, onChange: (v) => { S.mode = v; draw(); } });
  select(bar, { label: 'field', options: [{ value: 'rho', label: 'density' }, { value: 'pmag', label: 'magnetic pressure' }, { value: 'speed', label: 'in-plane speed' }], value: S.field, onChange: (v) => { S.field = v; draw(); } });
  slider(bar, { label: 'frame', min: 0, max: data[S.mode].frames.length - 1, step: 1, value: S.frame, format: (v) => String(Math.round(v)), onInput: (v) => { S.frame = Math.round(v); draw(); } });
  select(bar, { label: 'colours', options: ['viridis', 'magma', 'inferno', 'gray', 'coolwarm'].map((c) => ({ value: c, label: c })), value: S.cmap, onChange: (v) => { S.cmap = v; draw(); } });
  checkbox(bar, { label: 'log', value: S.log, onChange: (v) => { S.log = v; draw(); } });
  checkbox(bar, { label: 'patch outlines', value: S.outline, onChange: (v) => { S.outline = v; draw(); } });
  draw(); metaLine(host, meta);
  return { draw };
}

export const MOUNT = { ot: mountOT, ghost: mountGhost, conv: mountConv, adapt: mountAdapt, skip: mountSkip, tput: mountTput, otref: mountOtref, ergo: mountErgo };
