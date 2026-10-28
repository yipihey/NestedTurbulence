// app.js - the standalone laboratory: experiment chooser, state in the URL hash, copy-link button.
import { makeDefaults, encodeHash, decodeHash, EXPERIMENTS } from './state.js';
import { MOUNT } from './explore.js';
import { select } from './viz/ui.js';
import { copyText, flash } from './viz/download.js';

const S = decodeHash(location.hash);
const host = document.getElementById('lab');
const bar = document.getElementById('lab-bar');
select(bar, { label: 'experiment', options: EXPERIMENTS, value: S.exp, onChange: (v) => { S.exp = v; sync(); mount(); } });
const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'qd-btn'; btn.textContent = 'copy link';
btn.title = 'Copy the address of this exact view (the part after # encodes every control)';
btn.addEventListener('click', async () => { await copyText(location.href); flash(btn, 'copied'); });
bar.appendChild(btn);

const sync = () => history.replaceState(null, '', '#' + encodeHash(S));
function mount() {
  host.innerHTML = '';
  const cap = document.createElement('p'); cap.className = 'qd-note';
  const ex = EXPERIMENTS.find((e) => e.value === S.exp);
  const book = { ot: 'chapters/05-orszag-tang.html', ghost: 'chapters/03-validation.html#sec-ghost', conv: 'chapters/03-validation.html#sec-convergence',
                 adapt: 'chapters/04-performance.html#sec-adaptive', skip: 'chapters/04-performance.html#sec-skip',
                 tput: 'chapters/04-performance.html#sec-throughput', otref: 'chapters/03-validation.html#sec-otref' }[S.exp];
  cap.innerHTML = `${ex.label} — <a href="../${book}">§ book</a>`;
  host.appendChild(cap);
  const panel = document.createElement('div'); host.appendChild(panel);
  MOUNT[S.exp](panel, S, sync).catch((e) => { panel.textContent = 'failed to load: ' + e; });
}
window.addEventListener('hashchange', () => { if (location.hash.replace(/^#/, '') !== encodeHash(S)) location.reload(); });
mount();
