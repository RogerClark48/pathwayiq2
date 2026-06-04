/* Chat-first start screen */

import { state, submitMessage, setWaiting, setCourseList } from '../state.js';
import { postWelcomeChat } from '../api.js';
import { go } from '../router.js';
import { logEvent } from '../analytics.js';

const WELCOME_TEXT =
  "Hi — I'm here to help you find a course at the Greater Manchester " +
  "Institute of Technology (GMIoT), and to think about where it could lead. " +
  "Tell me what you're interested in, or what kind of work appeals to you, " +
  "and we'll go from there.";

const STARTER_CHIP_POOL = [
  "I like working with my hands",
  "I want a job outdoors",
  "Something with computers",
  "I want to work in the creative industries",
  "I want to change career",
  "I'm good at problem-solving",
  "I want to work in healthcare",
  "I like building and making things",
  "Something to do with engineering",
  "I want a career that pays well",
  "I'm not sure what I want yet",
  "I like helping people",
  "I want to work with technology",
  "Something hands-on, not a desk job",
  "I'm interested in construction",
  "I want to work in TV or media",
  "What can I do after my GCSEs?",
  "I like science and how things work",
  "Something creative with design",
  "I want a practical, skilled trade",
  "Where could an apprenticeship take me?",
  "I want to work with my local community",
  "I'm into gaming and software",
  "What jobs are growing around here?",
];

function pickStarterChips() {
  const fixed = "Show me some ideas";
  const pool  = STARTER_CHIP_POOL.slice();
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return [...pool.slice(0, 4), fixed];
}

function linkify(text) {
  const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // single pass: markdown [label](url) takes priority, bare URLs as fallback
  return escaped.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)]+)\)|(https?:\/\/[^\s<]+)/g,
    (_, label, mdUrl, bareUrl) => {
      const url = mdUrl || bareUrl;
      const display = label || bareUrl;
      return `<a href="${url}" target="_blank" rel="noopener noreferrer">${display}</a>`;
    }
  );
}

function makeBubble(role, text) {
  const wrap  = document.createElement('div');
  wrap.className = `chat-bubble chat-bubble--${role}`;
  const inner = document.createElement('div');
  inner.className = 'bubble-text';
  if (role === 'bot') {
    inner.innerHTML = linkify(text);
  } else {
    inner.textContent = text;
  }
  wrap.appendChild(inner);
  return wrap;
}

function makeCtaButton(text, courseList) {
  const btn = document.createElement('button');
  btn.className = 'chat-cta-btn';
  btn.textContent = text;
  btn.addEventListener('click', () => go('course-list', { courseList }));
  return btn;
}

function makeQualMapButton() {
  const btn = document.createElement('button');
  btn.className = 'chat-cta-btn';
  btn.textContent = 'View qualification map →';
  btn.addEventListener('click', () => go('pathway-map', { backRoute: 'chat-first' }));
  return btn;
}

function makeStarterChips() {
  const wrap = document.createElement('div');
  wrap.className = 'starter-chips';
  pickStarterChips().forEach(text => {
    const btn = document.createElement('button');
    btn.className = 'starter-chip';
    btn.dataset.text = text;
    btn.textContent = text;
    wrap.appendChild(btn);
  });
  return wrap;
}

function makeSuggestionChips(suggestions) {
  const wrap = document.createElement('div');
  wrap.className = 'suggestion-chips';
  suggestions.forEach(text => {
    const btn = document.createElement('button');
    btn.className = 'starter-chip suggestion-chip';
    btn.dataset.text = text;
    btn.textContent = text;
    wrap.appendChild(btn);
  });
  return wrap;
}

export function StartChatView({ prefill } = {}) {
  // Seed welcome message on first load; preserved across in-session navigations.
  if (state.chat.messages.length === 0) {
    state.chat.messages.push({ role: 'bot', text: WELCOME_TEXT });
  }

  const el = document.createElement('div');
  el.className = 'view view-chat';

  el.innerHTML = `
    <div class="chat-top-bar">
      <button class="chat-restart-btn" id="chat-restart">Start again</button>
    </div>
    <div class="chat-body">
      <div class="chat-messages" id="chat-messages"><div class="chat-spacer"></div></div>
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
    </div>
  `;

  const messagesEl  = el.querySelector('#chat-messages');
  const chatInput   = el.querySelector('#chat-input');
  const sendBtn     = el.querySelector('#chat-send');
  const restartBtn  = el.querySelector('#chat-restart');

  restartBtn.addEventListener('click', () => {
    const overlay = document.createElement('div');
    overlay.className = 'confirm-overlay';
    overlay.innerHTML = `
      <div class="confirm-panel">
        <p class="confirm-message">This will clear your conversation and any saved items.</p>
        <div class="confirm-actions">
          <button class="confirm-btn-cancel">Cancel</button>
          <button class="confirm-btn-ok">Start again</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    overlay.querySelector('.confirm-btn-cancel').addEventListener('click', () => overlay.remove());
    overlay.querySelector('.confirm-btn-ok').addEventListener('click', () => {
      overlay.remove();
      sessionStorage.removeItem('ff_session');
      window.location.reload();
    });
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  });

  // Render existing messages.
  state.chat.messages.forEach(msg => {
    if (msg.role === 'cta') {
      messagesEl.appendChild(makeCtaButton(msg.text, msg.courseList));
    } else {
      messagesEl.appendChild(makeBubble(msg.role, msg.text));
    }
  });

  // Starter chips — visible only while no user turn exists.
  const hasUserTurn = state.chat.messages.some(m => m.role === 'user');
  let starterEl = null;
  if (!hasUserTurn) {
    starterEl = makeStarterChips();
    messagesEl.appendChild(starterEl);
  }

  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
    if (prefill) {
      chatInput.value = prefill;
      sendBtn.disabled = false;
      chatInput.focus();
    }
  });

  // Enable send when input has content.
  chatInput.addEventListener('input', () => {
    sendBtn.disabled = chatInput.value.trim() === '';
  });

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function makeThinkingBubble() {
    const wrap  = document.createElement('div');
    wrap.className = 'chat-bubble chat-bubble--bot chat-bubble--thinking';
    const inner = document.createElement('div');
    inner.className = 'bubble-text bubble-text--thinking';
    inner.textContent = '···';
    wrap.appendChild(inner);
    return wrap;
  }

  let suggestionEl = null;

  function clearSuggestions() {
    if (suggestionEl) { suggestionEl.remove(); suggestionEl = null; }
  }

  async function submit() {
    const text = chatInput.value.trim();
    if (!text || state.chat.isWaitingForResponse) return;

    // Remove starter chips permanently on first user message.
    if (starterEl) { starterEl.remove(); starterEl = null; }

    // Remove any suggestion chips from the previous turn.
    clearSuggestions();

    logEvent('chat_submit', null, null, null, { query: text });
    submitMessage(text);
    messagesEl.appendChild(makeBubble('user', text));
    chatInput.value = '';
    sendBtn.disabled = true;
    scrollBottom();

    // Show thinking indicator and lock input.
    const thinkingEl = makeThinkingBubble();
    messagesEl.appendChild(thinkingEl);
    scrollBottom();
    setWaiting(true);
    chatInput.disabled = true;

    try {
      const data = await postWelcomeChat(state.session.id, text, state.saved.items);
      thinkingEl.remove();
      const reply = data.bot_response;
      state.chat.messages.push({ role: 'bot', text: reply });
      messagesEl.appendChild(makeBubble('bot', reply));

      if (data.suggestions?.length && !data.pivot_to_courses) {
        suggestionEl = makeSuggestionChips(data.suggestions);
        messagesEl.appendChild(suggestionEl);
      }

      if (data.show_qual_map) {
        messagesEl.appendChild(makeQualMapButton());
      }

      if (data.pivot_to_courses && data.course_list) {
        setCourseList(data.course_list);
        state.chat.messages.push({ role: 'cta', text: 'See courses →', courseList: data.course_list });
        messagesEl.appendChild(makeCtaButton('See courses →', data.course_list));
      }
      scrollBottom();
    } catch (err) {
      console.error('[chat/welcome]', err);
      thinkingEl.remove();
      const reply = 'Something went wrong — please try again.';
      state.chat.messages.push({ role: 'bot', text: reply });
      messagesEl.appendChild(makeBubble('bot', reply));
    } finally {
      setWaiting(false);
      chatInput.disabled = false;
      scrollBottom();
    }
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
