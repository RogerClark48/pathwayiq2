/* FutureFinder v3 — view router */

const _registry = new Map();
let _current = null;

// Views that live in the right panel on desktop (≥768px).
// Left panel (#view-root) holds Finn/chat; right panel (#desktop-right) holds results.
const RIGHT_PANEL_VIEWS = new Set([
  'course-list',
  'course-carousel',
  'course-detail',
  'career-view',
  'job-detail',
  'saved-list',
  'pathway-map',
]);

function root()       { return document.getElementById('view-root'); }
function rightPanel() { return document.getElementById('desktop-right'); }
function isDesktop()  { return window.matchMedia('(min-width: 768px)').matches; }

export function register(name, fn) {
  _registry.set(name, fn);
}

export function go(name, slices = {}) {
  const fn = _registry.get(name);
  if (!fn) throw new Error(`Router: no view registered as "${name}"`);

  const useRight = isDesktop() && RIGHT_PANEL_VIEWS.has(name);
  const el       = useRight ? rightPanel() : root();

  document.body.dataset.view = name;
  el.scrollTop = 0;
  el.innerHTML = '';

  // Clear the right panel before rendering so left-panel views (e.g. welcome)
  // can populate it as part of their own render without being clobbered.
  // Exception: chat-first keeps whatever is showing on the right — the map hero
  // stays live while Finn thinks, then gets replaced when courses arrive.
  if (!useRight && isDesktop() && name !== 'chat-first') {
    const rp = rightPanel();
    if (rp) rp.innerHTML = '';
  }

  el.appendChild(fn(slices));
  _current = name;
}

export function currentView() {
  return _current;
}
