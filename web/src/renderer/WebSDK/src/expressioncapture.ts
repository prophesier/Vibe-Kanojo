/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Expression capture — renders each Live2D expression to a transparent PNG so
 * the backend can cache "face" images and the Discord bridge can attach the
 * matching one to each reply. Driven from the running frontend (no headless
 * browser). Kept out of the core Live2D sample files; wired up from main.ts.
 *
 * Images are keyed by expression *index* (the same index the backend sends in
 * actions.expressions), so the bridge can look them up directly.
 */
import { LAppLive2DManager } from './lapplive2dmanager';
import { canvas } from './lappglmanager';

// Fraction of the model's full height (head→feet) that counts as the
// head-to-chest portrait. The capture zooms this region to fill the canvas.
const PORTRAIT_TOP_FRACTION = 0.30;
const ALPHA_THRESHOLD = 10; // pixels above this alpha count as "model"
// Cap the output image's longest side (px). Tunable for size vs. WS payload.
const MAX_OUTPUT_DIM = 1080;

// Expression the model is reset to after capturing, so it isn't left stuck on
// the last captured one.
const RESET_EXPRESSION_INDEX = 0;

// Capture canvas height (px). The width matches the live aspect (so the model
// isn't stretched). Independent of the window size, so it never exceeds the GPU
// limit. The head-to-chest region is zoomed to fill this, so every pixel goes to
// the face — a modest value already yields a high-res portrait.
const CAPTURE_HEIGHT = 1600;
// Fraction of the canvas height the head-to-chest region is zoomed to fill.
const FILL_FRACTION = 0.94;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getModel(): any {
  return LAppLive2DManager.getInstance()?.getModel(0);
}

/** Reset to a neutral resting expression so the model isn't left on the last capture. */
function resetExpression(): void {
  const model = getModel();
  const name: string | undefined = model?._expressions?._keyValues?.[RESET_EXPRESSION_INDEX]?.first;
  if (name != null) model.setExpression(name);
}

/** Find the model's alpha bounding box on the current canvas (coarse scan). */
function measureBBox(): { minX: number; minY: number; maxX: number; maxY: number } | null {
  const src = canvas!;
  const w = src.width;
  const h = src.height;
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext('2d')!;
  tctx.drawImage(src, 0, 0);
  const px = tctx.getImageData(0, 0, w, h).data;
  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < h; y += 2) {
    const row = y * w;
    for (let x = 0; x < w; x += 2) {
      if (px[(row + x) * 4 + 3] > ALPHA_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { minX, minY, maxX, maxY };
}

/**
 * Frame the head-to-chest region to fill the canvas, centred, so the capture
 * spends every pixel on the face (independent of the user's scroll-zoom / drag /
 * window mode). Returns the live matrix to restore afterwards.
 *
 * Steps: probe at a small scale until the whole model is un-clipped and measure
 * its bbox; the head-to-chest region is the top PORTRAIT_TOP_FRACTION of that.
 * Scale so that region fills FILL_FRACTION of the canvas height and shift so its
 * centre is at the canvas centre.
 */
async function applyCaptureFraming(model: any): Promise<number[]> {
  const saved: number[] = model._modelMatrix.getArray().slice();
  const setMat = (s: number, ty: number): void =>
    model._modelMatrix.setMatrix([s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1, 0, 0, ty, 0, 1]);
  const H = canvas!.height;

  // 1. Probe: shrink until the whole model is un-clipped, measure its bbox.
  let probe = 0.5;
  let bbox: ReturnType<typeof measureBBox> = null;
  for (let i = 0; i < 6; i += 1) {
    setMat(probe, 0);
    // eslint-disable-next-line no-await-in-loop
    await wait(120);
    bbox = measureBBox();
    if (bbox) {
      const W = canvas!.width;
      const clipped =
        bbox.minX <= 1 || bbox.minY <= 1 || bbox.maxX >= W - 2 || bbox.maxY >= H - 2;
      if (!clipped) break;
    }
    probe *= 0.6;
  }
  if (!bbox) return saved;

  // 2. Learn the ty→pixel relationship (translation is scale-independent).
  const minYProbe = bbox.minY;
  setMat(probe, 0.1);
  await wait(120);
  const nudged = measureBBox();
  const pxPerTy = nudged ? (nudged.minY - minYProbe) / 0.1 : 0;

  // 3. Head-to-chest region (top fraction of the full model), then scale it to
  //    fill the canvas and centre it.
  const hcHeight = PORTRAIT_TOP_FRACTION * (bbox.maxY - bbox.minY + 1);
  const hcCenterY = bbox.minY + hcHeight / 2;
  const fit = probe * ((FILL_FRACTION * H) / hcHeight);
  // Scaling happens around the rig origin (canvas centre at ty=0); track where
  // hcCenterY lands, then shift it to the canvas centre.
  const hcCenterYAtFit = H / 2 + (hcCenterY - H / 2) * (fit / probe);
  const ty = pxPerTy !== 0 ? (H / 2 - hcCenterYAtFit) / pxPerTy : 0;
  setMat(fit, ty);
  await wait(120);
  return saved;
}

/**
 * Read the current rendered frame and tightly crop the visible model (the
 * framing already zoomed the head-to-chest region to fill the canvas), drop the
 * surrounding transparent space, downscale to MAX_OUTPUT_DIM, return a PNG URL.
 */
function cropPortrait(): string {
  const src = canvas!;
  const w = src.width;
  const h = src.height;

  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext('2d')!;
  tctx.drawImage(src, 0, 0);
  const px = tctx.getImageData(0, 0, w, h).data;

  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < h; y += 2) {
    const row = y * w;
    for (let x = 0; x < w; x += 2) {
      if (px[(row + x) * 4 + 3] > ALPHA_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return src.toDataURL('image/png'); // nothing drawn

  const ow = maxX - minX + 1;
  const oh = maxY - minY + 1;
  // Downscale to MAX_OUTPUT_DIM on the longest side.
  const scale = Math.min(1, MAX_OUTPUT_DIM / Math.max(ow, oh));
  const outW = Math.max(1, Math.round(ow * scale));
  const outH = Math.max(1, Math.round(oh * scale));
  const out = document.createElement('canvas');
  out.width = outW;
  out.height = outH;
  out.getContext('2d')!.drawImage(tmp, minX, minY, ow, oh, 0, 0, outW, outH);
  return out.toDataURL('image/png');
}

/** Number of expressions the current model has. */
export function expressionCount(): number {
  const model = getModel();
  return model?._expressions?.getSize?.() ?? 0;
}

/**
 * Capture one expression (by index) as a transparent PNG data URL.
 * settleMs lets the expression fade-in finish before the frame is read.
 */
export async function captureExpression(index: number, settleMs = 1000): Promise<string> {
  const model = getModel();
  if (!model || !canvas) throw new Error('model/canvas not ready');

  const exprs = model._expressions;
  const name: string | undefined = exprs?._keyValues?.[index]?.first;
  if (name == null) throw new Error(`no expression at index ${index}`);

  // Hold the eyes open so we never catch a mid-blink frame; render at higher
  // resolution for a sharper crop. Both restored in finally.
  const savedEyeBlink = model._eyeBlink;
  const savedW = canvas.width;
  const savedH = canvas.height;
  model._eyeBlink = null;
  canvas.height = CAPTURE_HEIGHT;
  canvas.width = Math.round(CAPTURE_HEIGHT * (savedW / savedH));
  // Pause the resize hook's per-frame scale enforcement so our capture framing
  // isn't overwritten (see use-live2d-resize.animateEase).
  (window as any).__capturing = true;
  let savedMatrix: number[] = model._modelMatrix.getArray().slice();
  try {
    savedMatrix = await applyCaptureFraming(model);
    model.setExpression(name);
    await wait(settleMs);
    return cropPortrait();
  } finally {
    (window as any).__capturing = false;
    model._modelMatrix.setMatrix(savedMatrix);
    canvas.width = savedW;
    canvas.height = savedH;
    model._eyeBlink = savedEyeBlink;
    resetExpression();
  }
}

/** Capture every expression. Returns { index: dataUrl }. */
export async function captureAllFaces(settleMs = 1000): Promise<Record<number, string>> {
  const model = getModel();
  if (!model || !canvas) throw new Error('model/canvas not ready');

  const count = expressionCount();
  const out: Record<number, string> = {};
  const savedEyeBlink = model._eyeBlink;
  const savedW = canvas.width;
  const savedH = canvas.height;
  model._eyeBlink = null;
  canvas.height = CAPTURE_HEIGHT;
  canvas.width = Math.round(CAPTURE_HEIGHT * (savedW / savedH));
  // Pause the resize hook's per-frame scale enforcement (see captureExpression).
  (window as any).__capturing = true;
  let savedMatrix: number[] = model._modelMatrix.getArray().slice();
  try {
    savedMatrix = await applyCaptureFraming(model);
    for (let i = 0; i < count; i += 1) {
      const name = model._expressions._keyValues[i].first;
      model.setExpression(name);
      // eslint-disable-next-line no-await-in-loop
      await wait(settleMs);
      out[i] = cropPortrait();
    }
  } finally {
    (window as any).__capturing = false;
    model._modelMatrix.setMatrix(savedMatrix);
    canvas.width = savedW;
    canvas.height = savedH;
    model._eyeBlink = savedEyeBlink;
    resetExpression();
  }
  return out;
}

/** Dev helper: capture all faces and show them as a grid overlay (click to close). */
async function previewAllFaces(): Promise<void> {
  const faces = await captureAllFaces();
  const overlay = document.createElement('div');
  overlay.style.cssText =
    'position:fixed;inset:0;z-index:99999;background:#1a1a1a;overflow:auto;'
    + 'display:flex;flex-wrap:wrap;gap:8px;padding:12px;align-content:flex-start';
  overlay.title = 'click to close';
  overlay.onclick = () => overlay.remove();
  Object.entries(faces).forEach(([i, url]) => {
    const wrap = document.createElement('div');
    wrap.style.cssText =
      'background:#333;padding:4px;text-align:center;color:#fff;font:12px sans-serif';
    wrap.innerHTML =
      `<div>${i}</div><img src="${url}" `
      + 'style="width:300px;height:300px;object-fit:contain;'
      + 'background:repeating-conic-gradient(#555 0 25%,#666 0 50%) 0/20px 20px"/>';
    overlay.appendChild(wrap);
  });
  document.body.appendChild(overlay);
}

/** Expose the capture API on window (read by main.ts / future WS handler). */
export function setupExpressionCapture(): void {
  (window as any).__captureExpression = captureExpression;
  (window as any).__captureAllFaces = captureAllFaces;
  // Dev-only: the preview overlay must be clicked to close, which could lock the
  // UI if it were triggered while nobody is at the PC. Not exposed in production
  // builds. Set window.__enableFacePreview = true before init to force it on.
  if ((import.meta as any).env?.DEV || (window as any).__enableFacePreview) {
    (window as any).__previewAllFaces = previewAllFaces;
  }
}

// Bind immediately on module (re)load so Vite HMR re-binds the window helpers
// without a full page reload. main.ts also calls this after Live2D init.
if (typeof window !== 'undefined') {
  setupExpressionCapture();
}
