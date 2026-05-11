/* FutureFinder v3 — application state */

function getOrCreateSessionId() {
  let id = sessionStorage.getItem('ff_session');
  if (!id) { id = crypto.randomUUID(); sessionStorage.setItem('ff_session', id); }
  return id;
}

export const state = {
  filter:  { subjects: new Set() },
  session: { id: getOrCreateSessionId(), interactionCount: 0, advisoryGap: 0 },
  chat:    { messages: [], history: [], lastContext: null },
  saved:   { items: [] },
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

export function pushChatTurn(turn) {
  state.chat.history.push(turn);
  state.chat.lastContext = turn.context ?? null;
}

/* ── saved ───────────────────────────────────────────────────────────────── */

export function saveItem(item) {
  const already = state.saved.items.some(i => i.id === item.id && i.type === item.type);
  if (!already) state.saved.items.push(item);
}

export function unsaveItem(id, type) {
  state.saved.items = state.saved.items.filter(i => !(i.id === id && i.type === type));
}

export function isSaved(id, type) {
  return state.saved.items.some(i => i.id === id && i.type === type);
}
