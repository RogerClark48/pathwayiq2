/* FutureFinder v3 — API fetch helpers */

// Memoised welcome-data fetch. Callers can await this as many times as they
// like; the HTTP request fires exactly once regardless of how many views call it.
let _welcomePromise = null;
let _welcomeData    = null;

export function loadWelcomeData() {
  if (!_welcomePromise) {
    _welcomePromise = fetch('/api/welcome-data')
      .then(r => {
        if (!r.ok) throw new Error(`/api/welcome-data ${r.status}`);
        return r.json();
      })
      .then(data => {
        _welcomeData = data;
        return data;
      });
  }
  return _welcomePromise;
}

// Synchronous accessor — returns null until the promise has resolved.
// Subject and edit views use this; they always render after welcome so
// data is guaranteed to be present.
export function getWelcomeData() {
  return _welcomeData;
}

// CARTO Basemaps raster tiles. Since Aug 2026 CARTO requires an API key on the
// tile URL — keyless requests get an "API KEY REQUIRED" watermark. The key is
// served (referrer-restricted, safe to expose) from /api/welcome-data → map.carto_key.
// Callers should `await loadWelcomeData()` before invoking so the key is present.
export function cartoTileLayer(opts = {}) {
  const key = _welcomeData?.map?.carto_key || '';
  const url = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
            + (key ? `?key=${encodeURIComponent(key)}` : '');
  return window.L.tileLayer(url, {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>' +
                 ' &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20,
    ...opts,
  });
}

export async function postWelcomeChat(sessionId, message, savedItems) {
  const resp = await fetch('/chat/welcome', {
    method:  'POST',
    headers: { 'content-type': 'application/json' },
    body:    JSON.stringify({ session_id: sessionId, message, saved_items: savedItems || [] }),
  });
  if (!resp.ok) throw new Error(`/chat/welcome ${resp.status}`);
  return resp.json(); // { session_id, bot_response, pivot_to_courses }
}
