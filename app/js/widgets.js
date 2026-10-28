// widgets.js - chapter widgets: <div class="qd-widget" data-widget="ghost" data-state="m=glm&c=0.4"></div>
// plus this module script on the page. data-state is an optional lab hash that sets the widget's initial controls.
import { decodeHash } from './state.js';
import { MOUNT } from './explore.js';

for (const el of document.querySelectorAll('.qd-widget')) {
  const m = MOUNT[el.dataset.widget];
  if (m) m(el, decodeHash(el.dataset.state || '')).catch((e) => { el.textContent = 'widget failed: ' + e; });
}
