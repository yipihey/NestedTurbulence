// state.js - the complete control state of the laboratory and its URL-hash encoding.
// S uses readable names; the hash uses the short keys of SCHEMA. Only values that differ from the defaults are
// written, so a bare URL is the default view and any hash is a permalink, a preset and a regression fixture at once.

export const EXPERIMENTS = [
  { value: 'ot', label: 'Nested Orszag–Tang: fields' },
  { value: 'ghost', label: 'Ghost-zone time accuracy' },
  { value: 'conv', label: 'Convergence (Alfvén wave)' },
  { value: 'adapt', label: 'Adaptive vs fixed subcycling' },
  { value: 'skip', label: 'Covered-region skipping' },
  { value: 'tput', label: 'CPU throughput' },
  { value: 'otref', label: 'Orszag–Tang reference check' },
];

export function makeDefaults() {
  return { exp: 'ot', mode: 'ct', courant: 0.8, field: 'rho', frame: 10, outline: true, log: false,
           cmap: 'viridis', problem: 'cold-core' };
}

// [stateKey, hashKey, type, range-or-choices]
const SCHEMA = [
  ['exp', 'ex', 'str', ['ot', 'ghost', 'conv', 'adapt', 'skip', 'tput', 'otref']],
  ['mode', 'm', 'str', ['ct', 'glm']],
  ['courant', 'c', 'num', [0.4, 0.8]],
  ['field', 'f', 'str', ['rho', 'pmag', 'speed']],
  ['frame', 't', 'int', [0, 10]],
  ['outline', 'ol', 'bool'],
  ['log', 'lg', 'bool'],
  ['cmap', 'cm', 'str', ['viridis', 'magma', 'inferno', 'gray', 'coolwarm']],
  ['problem', 'pr', 'str', ['orszag-tang', 'cold-core']],
];

const num = (v) => String(+Number(v).toPrecision(6));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Hash string (without '#') for state S. */
export function encodeHash(S) {
  const D = makeDefaults();
  const p = [];
  for (const [sk, hk, type] of SCHEMA) {
    const v = S[sk];
    if (JSON.stringify(v) === JSON.stringify(D[sk])) continue;
    p.push(`${hk}=${type === 'bool' ? (v ? '1' : '0') : type === 'str' ? v : num(v)}`);
  }
  return p.join('&');
}

/** State from a hash string (unknown or invalid entries are ignored). */
export function decodeHash(hash) {
  const S = makeDefaults();
  for (const kv of (hash || '').replace(/^#/, '').split('&')) {
    const [hk, raw] = kv.split('=');
    const ent = SCHEMA.find((e) => e[1] === hk);
    if (!ent || raw === undefined) continue;
    const [sk, , type, range] = ent;
    if (type === 'bool') S[sk] = raw === '1';
    else if (type === 'str') { if (!range || range.includes(raw)) S[sk] = raw; }
    else {
      let v = parseFloat(raw);
      if (!Number.isFinite(v)) continue;
      if (type === 'int') v = Math.round(v);
      S[sk] = range ? clamp(v, range[0], range[1]) : v;
    }
  }
  return S;
}
