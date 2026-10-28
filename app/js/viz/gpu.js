// gpu.js (kit subset) - only the canvas helper the colorbar needs; the full WebGPU helpers live in the reference report.
export function pinCssSize(canvas) {
  if (canvas._hccPinned) return;
  const w0 = canvas.width, h0 = canvas.height;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return; // hidden: cannot tell; try again later
  canvas._hccPinned = true;
  canvas.width = w0 + 7; canvas.height = h0 + 7;
  const intrinsicW = canvas.clientWidth !== cw, intrinsicH = canvas.clientHeight !== ch;
  canvas.width = w0; canvas.height = h0;
  if (intrinsicW && !canvas.style.width) canvas.style.width = cw + 'px';
  if (intrinsicH && !canvas.style.height) canvas.style.height = ch + 'px';
}
