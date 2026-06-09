/* FutureFinder v3 — view router */

const _registry = new Map();
let _current = null;

// Views that live in the right panel on desktop (≥768px).
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

function clearSplit() {
  const rp = rightPanel();
  if (rp) rp.classList.remove('split');
}

// Async: fetch course detail and render CareerView into container.
function loadCareerBottom(courseId, container) {
  container.innerHTML = '<div class="split-loading">Loading career pathways…</div>';
  fetch(`/courses/${courseId}/detail`)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(d => {
      if (!container.isConnected) return;
      const fn = _registry.get('career-view');
      if (!fn) return;
      container.innerHTML = '';
      container.appendChild(fn({
        courseId,
        courseTitle: d.course_title,
        ssa:         d.ssa_code,
        pathways:    d.pathways,
        backRoute:   null,   // no back button needed in split bottom zone
        backSlices:  null,
      }));
    })
    .catch(() => {
      if (container.isConnected)
        container.innerHTML = '<div class="split-loading split-loading--err">Career pathways unavailable.</div>';
    });
}

export function register(name, fn) {
  _registry.set(name, fn);
}

export function go(name, slices = {}) {
  const fn = _registry.get(name);
  if (!fn) throw new Error(`Router: no view registered as "${name}"`);

  const desktop  = isDesktop();
  const useRight = desktop && RIGHT_PANEL_VIEWS.has(name);
  const rp       = rightPanel();

  // ── Desktop course-detail: split right panel top/bottom ──────────────────
  if (desktop && name === 'course-detail') {
    rp.classList.add('split');
    rp.innerHTML = '';

    const top    = document.createElement('div');
    const bottom = document.createElement('div');
    top.id    = 'right-top';
    bottom.id = 'right-bottom';
    rp.appendChild(top);
    rp.appendChild(bottom);

    top.appendChild(fn(slices));
    if (slices.courseId) loadCareerBottom(slices.courseId, bottom);

    document.body.dataset.view = name;
    _current = name;
    return;
  }

  // ── Desktop career-view within an active split: update bottom zone only ──
  if (desktop && name === 'career-view' && rp && rp.classList.contains('split')) {
    const bottom = document.getElementById('right-bottom');
    if (bottom) {
      bottom.scrollTop = 0;
      bottom.innerHTML = '';
      bottom.appendChild(fn(slices));
      document.body.dataset.view = name;
      _current = name;
      return;
    }
  }

  // ── Normal routing ────────────────────────────────────────────────────────
  const el = useRight ? rp : root();

  if (useRight) clearSplit();

  document.body.dataset.view = name;
  el.scrollTop = 0;
  el.innerHTML = '';
  el.appendChild(fn(slices));
  _current = name;

  // Navigating to a left-panel view clears the right panel
  if (!useRight && desktop && rp) {
    clearSplit();
    rp.innerHTML = '';
  }
}

export function currentView() {
  return _current;
}
