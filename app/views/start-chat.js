/* Chat-first start screen */

import { state, submitMessage } from '../state.js';

const WELCOME_TEXT =
  "Hi — I'm here to help you find a course at the Greater Manchester " +
  "Institute of Technology (GMIoT), and to think about where it could lead. " +
  "Tell me what you're interested in, or what kind of work appeals to you, " +
  "and we'll go from there.";

const STARTER_CHIPS = [
  "I like working with my hands",
  "I want a job outdoors",
  "Something with computers",
  "I want to be a nurse",
  "I'm not sure where to start",
  "Show me what's popular",
];

function makeBubble(role, text) {
  const wrap  = document.createElement('div');
  wrap.className = `chat-bubble chat-bubble--${role}`;
  const inner = document.createElement('div');
  inner.className = 'bubble-text';
  inner.textContent = text;
  wrap.appendChild(inner);
  return wrap;
}

function makeStarterChips() {
  const wrap = document.createElement('div');
  wrap.className = 'starter-chips';
  STARTER_CHIPS.forEach(text => {
    const btn = document.createElement('button');
    btn.className = 'starter-chip';
    btn.dataset.text = text;
    btn.textContent = text;
    wrap.appendChild(btn);
  });
  return wrap;
}

export function StartChatView() {
  // Seed welcome message on first load; preserved across in-session navigations.
  if (state.chat.messages.length === 0) {
    state.chat.messages.push({ role: 'bot', text: WELCOME_TEXT });
  }

  const el = document.createElement('div');
  el.className = 'view view-chat';

  el.innerHTML = `
    <div class="chat-messages" id="chat-messages"></div>
    <div class="chat-input-area">
      <div class="chat-input-box">
        <input type="text" class="chat-input-field" id="chat-input"
               placeholder="Type your message…"
               autocomplete="off" autocorrect="off" spellcheck="false">
        <button class="chat-send-btn" id="chat-send" disabled aria-label="Send message">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"
               aria-hidden="true">
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
  `;

  const messagesEl = el.querySelector('#chat-messages');
  const chatInput  = el.querySelector('#chat-input');
  const sendBtn    = el.querySelector('#chat-send');

  // Render existing messages.
  state.chat.messages.forEach(msg => {
    messagesEl.appendChild(makeBubble(msg.role, msg.text));
  });

  // Starter chips — visible only while no user turn exists.
  const hasUserTurn = state.chat.messages.some(m => m.role === 'user');
  let starterEl = null;
  if (!hasUserTurn) {
    starterEl = makeStarterChips();
    messagesEl.appendChild(starterEl);
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;

  // Enable send when input has content.
  chatInput.addEventListener('input', () => {
    sendBtn.disabled = chatInput.value.trim() === '';
  });

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function submit() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Remove starter chips permanently on first user message.
    if (starterEl) { starterEl.remove(); starterEl = null; }

    submitMessage(text);
    messagesEl.appendChild(makeBubble('user', text));
    chatInput.value = '';
    sendBtn.disabled = true;
    scrollBottom();

    // Phase 1 stub — placeholder bot response.
    setTimeout(() => {
      const reply = 'Chat backend coming soon — thanks for exploring!';
      state.chat.messages.push({ role: 'bot', text: reply });
      messagesEl.appendChild(makeBubble('bot', reply));
      scrollBottom();
    }, 400);
  }

  sendBtn.addEventListener('click', submit);
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  });

  // Chip tap — drop text into input, do not submit.
  messagesEl.addEventListener('click', e => {
    const chip = e.target.closest('.starter-chip');
    if (!chip) return;
    chatInput.value = chip.dataset.text;
    sendBtn.disabled = false;
    chatInput.focus();
  });

  return el;
}
