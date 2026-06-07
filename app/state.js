/* FutureFinder v3 — application state */

const SAVED_LS_KEY = 'ff_v3_saved';

function getOrCreateSessionId() {
  let id = sessionStorage.getItem('ff_session');
  if (!id) { id = crypto.randomUUID(); sessionStorage.setItem('ff_session', id); }
  return id;
}

function loadSaved() {
  try {
    const raw = localStorage.getItem(SAVED_LS_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr.filter(i => i && i.id && i.type && i.title) : [];
  } catch { return []; }
}

function persistSaved() {
  try { localStorage.setItem(SAVED_LS_KEY, JSON.stringify(state.saved.items)); } catch {}
}

export const state = {
  filter:     { subjects: new Set() },
  session:    { id: getOrCreateSessionId(), interactionCount: 0, advisoryGap: 0 },
  chat:       { messages: [], history: [], lastContext: null, isWaitingForResponse: false },
  saved:      { items: loadSaved() },
  courseList: null,
};

/* ── filter ──────────────────────────────────────────────────────────────── */

export function toggleSubject(subject) {
  if (state.filter.subjects.has(subject)) {
    state.filter.subjects.delete(subject);
  } else {
    state.filter.subjects.add(subject);
  }
}

export function clearFilter() {
  state.filter.subjects.clear();
}

/* ── session ─────────────────────────────────────────────────────────────── */

export function incrementInteraction() {
  state.session.interactionCount++;
  state.session.advisoryGap++;
}

export function resetAdvisoryGap() {
  state.session.advisoryGap = 0;
}

/* ── chat ────────────────────────────────────────────────────────────────── */

export function submitMessage(text) {
  state.chat.messages.push({ role: 'user', text });
}

export function setWaiting(val) {
  state.chat.isWaitingForResponse = val;
}

export function setCourseList(data) {
  state.courseList = data;
}

export function pushChatTurn(turn) {
  state.chat.history.push(turn);
  state.chat.lastContext = turn.context ?? null;
}

/* ── saved ───────────────────────────────────────────────────────────────── */

export function saveItem(item) {
  const already = state.saved.items.some(i => i.id === item.id && i.type === item.type);
  if (!already) { state.saved.items.push(item); persistSaved(); }
}

export function unsaveItem(id, type) {
  state.saved.items = state.saved.items.filter(i => !(i.id === id && i.type === type));
  persistSaved();
}

export function isSaved(id, type) {
  return state.saved.items.some(i => i.id === id && i.type === type);
}

/** Toggle save for an enriched item { id, type, title, ssa?, subtitle? }. */
export function toggleSave(item) {
  const on = isSaved(item.id, item.type);
  if (on) unsaveItem(item.id, item.type); else saveItem(item);
  refreshSavedBadges();
  return !on;
}

/** Update every [data-saved-count] badge in the document. */
export function refreshSavedBadges() {
  const n = state.saved.items.length;
  document.querySelectorAll('[data-saved-count]').forEach(b => {
    b.textContent = n;
    b.hidden = n === 0;
  });
}

/* ── reset ───────────────────────────────────────────────────────────────── */

export function resetSession() {
  sessionStorage.removeItem('ff_session');
  const newId = crypto.randomUUID();
  sessionStorage.setItem('ff_session', newId);
  state.session.id = newId;
  state.session.interactionCount = 0;
  state.session.advisoryGap = 0;
  state.chat.messages = [];
  state.chat.history = [];
  state.chat.lastContext = null;
  state.chat.isWaitingForResponse = false;
  state.saved.items = [];
  persistSaved();
  state.courseList = null;
  state.filter.subjects.clear();
}
