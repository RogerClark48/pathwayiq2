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
