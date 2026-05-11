/* FutureFinder v3 — view router */

const _registry = new Map();
let _current = null;

function root() {
  return document.getElementById('view-root');
}

export function register(name, fn) {
  _registry.set(name, fn);
}

export function go(name, slices = {}) {
  const fn = _registry.get(name);
  if (!fn) throw new Error(`Router: no view registered as "${name}"`);

  const el = root();
  el.scrollTop = 0;
  el.innerHTML = '';
  el.appendChild(fn(slices));
  _current = name;
}

export function currentView() {
  return _current;
}
