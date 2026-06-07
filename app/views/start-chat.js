/* Chat thread — Finn header, branded bubbles, in-chat course card pivot */

import { state, submitMessage, setWaiting, setCourseList, resetSession } from '../state.js';
import { bookmarkButton } from '../bookmarks.js';
import { splitProse } from '../dom.js';
import { postWelcomeChat } from '../api.js';
import { go } from '../router.js';
import { logEvent } from '../analytics.js';
import { subject, subjectIconSvg } from '../subjects.js';

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

// ── Helpers ───────────────────────────────────────────────────────────────────

const MODE_LABELS = { FT: 'Full-time', PT: 'Part-time', 'FT/PT': 'Full or Part-time' };
function modeLabel(mode) { return mode ? (MODE_LABELS[mode] || mode) : null; }

function splitTitle(raw) {
  const m = (raw || '').match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  return m ? { title: m[1].trim(), specialism: m[2].trim() } : { title: raw || '', specialism: null };
}

function linkify(text) {
  const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return escaped.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)]+)\)|(https?:\/\/[^\s<]+)/g,
    (_, label, mdUrl, bareUrl) => {
      const url = mdUrl || bareUrl;
      const display = label || bareUrl;
      return `<a href="${url}" target="_blank" rel="noopener noreferrer">${display}</a>`;
    }
  );
}

function pickStarterChips() {
  const fixed = "Show me some ideas";
  const pool  = STARTER_CHIP_POOL.slice();
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return [...pool.slice(0, 4), fixed];
}

// ── DOM builders ──────────────────────────────────────────────────────────────

function makeBubble(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role === 'bot' ? 'ai' : 'me'}`;

  if (role === 'bot') {
    const av = document.createElement('span');
    av.className = 'av';
    av.innerHTML = `<img src="assets/brand/logo-mark.png" alt="">`;

    const stack = document.createElement('div');
    stack.className = 'stack';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    const paras = splitProse(text.trim()).filter(Boolean);
    if (paras.length > 1) {
      const prose = document.createElement('div');
      prose.className = 'prose';
      paras.forEach(par => {
        const p = document.createElement('p');
        p.innerHTML = linkify(par);  // linkify HTML-escapes before injecting links
        prose.appendChild(p);
      });
      bubble.appendChild(prose);
    } else {
      bubble.innerHTML = linkify(text);
    }
    stack.appendChild(bubble);

    wrap.appendChild(av);
    wrap.appendChild(stack);
  } else {
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
  }

  return wrap;
}

function makeThinkingBubble() {
  const wrap = document.createElement('div');
  wrap.className = 'msg ai';
  wrap.innerHTML = `
    <span class="av"><img src="assets/brand/logo-mark.png" alt=""></span>
    <div class="stack"><div class="bubble bubble--thinking">···</div></div>`;
  return wrap;
}

function makeCourseCard(course) {
  const s = subject(course.ssa_code);
  const rawTitle = course.course_title || course.title || '';
  const { title, specialism } = splitTitle(rawTitle);
  const lvlQual  = course.qual_type || (course.level != null ? `L${course.level}` : '');
  const provider = course.campus_name || course.provider_name || course.provider || '';

  const chipsHtml = [
    course.qual_type          ? `<span>${course.qual_type}</span>`           : '',
    modeLabel(course.mode)    ? `<span>${modeLabel(course.mode)}</span>`     : '',
  ].filter(Boolean).join('');

  const card = document.createElement('div');
  card.className = 'ck ck-course';
  card.style.setProperty('--sub', s.colour);

  card.innerHTML = `
    <div class="cc-top">
      <span class="wm">${subjectIconSvg(course.ssa_code)}</span>
      ${lvlQual ? `<span class="lvl">${lvlQual}</span>` : ''}
      <div class="cc-tt">
        <b>${title}</b>
        <span>${provider || specialism || ''}</span>
      </div>
    </div>
    ${chipsHtml ? `<div class="cc-bd"><div class="cc-chips">${chipsHtml}</div></div>` : ''}
    <div class="ck-more">Open course  ›</div>`;

  const openCourse = () => go('course-detail', {
    courseId:    course.course_id,
    courseTitle: rawTitle,
    backRoute:   'chat-first',
  });

  card.querySelector('.ck-more').addEventListener('click', e => { e.stopPropagation(); openCourse(); });
  card.addEventListener('click', openCourse);

  // Bookmark in card body — stopPropagation handled inside bookmarkButton
  card.querySelector('.cc-bd')?.appendChild(bookmarkButton({
    id:       course.course_id,
    type:     'course',
    title:    rawTitle,
    ssa:      course.ssa_code,
    subtitle: [course.qual_type, provider].filter(Boolean).join(' · '),
  }));

  return card;
}

// Renders as a .msg.ai: Finn avatar + stack containing the lead course card + see-all button.
function makePivotBlock(courseList) {
  const courses = (courseList && courseList.courses) || [];
  const lead = courses[0];
  if (!lead) return null;

  const total = courses.length;

  const wrap = document.createElement('div');
  wrap.className = 'msg ai';

  const av = document.createElement('span');
  av.className = 'av';
  av.innerHTML = `<img src="assets/brand/logo-mark.png" alt="">`;

  const stack = document.createElement('div');
  stack.className = 'stack';

  stack.appendChild(makeCourseCard(lead));

  const seeAll = document.createElement('button');
  seeAll.className = 'see-all';
  seeAll.textContent = `See all ${total} courses on the map →`;
  seeAll.addEventListener('click', () => go('course-list', { courseList, backRoute: 'chat-first' }));
  stack.appendChild(seeAll);

  wrap.appendChild(av);
  wrap.appendChild(stack);
  return wrap;
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

// ── Main view ─────────────────────────────────────────────────────────────────

export function StartChatView({ prefill, autosend } = {}) {
  if (state.chat.messages.length === 0) {
    state.chat.messages.push({ role: 'bot', text: WELCOME_TEXT });
  }

  const savedCount = state.saved.items.length;

  const el = document.createElement('div');
  el.className = 'view view-chat';

  el.innerHTML = `
    <div class="chat-head">
      <div class="id">
        <img src="assets/brand/logo-mark.png" alt="Finn">
        <div>
          <div class="nm">Finn</div>
          <div class="st">Your guide · online</div>
        </div>
      </div>
      <div class="ch-actions">
        <button class="ch-btn" id="chat-new" aria-label="New conversation">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 20h9"/>
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
        </button>
        <button class="ch-btn" id="chat-saved" aria-label="Saved items">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/>
          </svg>
          <span class="ct" data-saved-count${savedCount === 0 ? ' hidden' : ''}>${savedCount}</span>
        </button>
      </div>
    </div>

    <div class="chat-body">
      <div class="chat-messages" id="chat-messages"></div>
      <div class="chat-input-wrap">
        <div class="chat-sugs" id="chat-sugs"></div>
        <div class="chat-bar">
          <input type="text" class="chat-input-field" id="chat-input"
                 placeholder="Type your message…"
                 autocomplete="off" autocorrect="off" spellcheck="false">
          <button class="send chat-send-btn" id="chat-send" disabled aria-label="Send message">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  `;

  const messagesEl = el.querySelector('#chat-messages');
  const chatInput  = el.querySelector('#chat-input');
  const sendBtn    = el.querySelector('#chat-send');
  const sugsEl     = el.querySelector('#chat-sugs');

  // Home = new conversation (confirm before clearing)
  el.querySelector('#chat-new').addEventListener('click', () => {
    const overlay = document.createElement('div');
    overlay.className = 'confirm-overlay';
    overlay.innerHTML = `
      <div class="confirm-panel">
        <p class="confirm-message">This will clear your conversation and any saved items.</p>
        <div class="confirm-actions">
          <button class="confirm-btn-cancel">Cancel</button>
          <button class="confirm-btn-ok">Start again</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    overlay.querySelector('.confirm-btn-cancel').addEventListener('click', () => overlay.remove());
    overlay.querySelector('.confirm-btn-ok').addEventListener('click', () => {
      overlay.remove();
      resetSession();
      go('welcome');
    });
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  });

  el.querySelector('#chat-saved').addEventListener('click', () => go('saved-list', { backRoute: 'chat-first' }));

  // Render persisted messages
  state.chat.messages.forEach(msg => {
    if (msg.role === 'cta') {
      const block = makePivotBlock(msg.courseList);
      if (block) messagesEl.appendChild(block);
    } else {
      messagesEl.appendChild(makeBubble(msg.role, msg.text));
    }
  });

  // Starter chips — visible only while no user turn exists
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
      if (autosend) {
        submit();
      } else {
        chatInput.focus();
      }
    }
  });

  chatInput.addEventListener('input', () => {
    sendBtn.disabled = chatInput.value.trim() === '';
  });

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function submit() {
    const text = chatInput.value.trim();
    if (!text || state.chat.isWaitingForResponse) return;

    if (starterEl) { starterEl.remove(); starterEl = null; }
    sugsEl.innerHTML = '';

    logEvent('chat_submit', null, null, null, { query: text });
    submitMessage(text);
    messagesEl.appendChild(makeBubble('user', text));
    chatInput.value = '';
    sendBtn.disabled = true;
    scrollBottom();

    const thinkingEl = makeThinkingBubble();
    messagesEl.appendChild(thinkingEl);
    scrollBottom();
    setWaiting(true);
    chatInput.disabled = true;

    try {
      const data = await postWelcomeChat(state.session.id, text, state.saved.items);
      thinkingEl.remove();
      const reply = data.bot_response;

      if (data.pivot_to_courses && data.course_list) {
        // Finn's sentence bubble, then pivot block (card + see-all) as a second .msg.ai
        state.chat.messages.push({ role: 'bot', text: reply });
        messagesEl.appendChild(makeBubble('bot', reply));

        setCourseList(data.course_list);
        state.chat.messages.push({ role: 'cta', courseList: data.course_list });
        const block = makePivotBlock(data.course_list);
        if (block) messagesEl.appendChild(block);
      } else {
        state.chat.messages.push({ role: 'bot', text: reply });
        messagesEl.appendChild(makeBubble('bot', reply));

        if (data.suggestions?.length) {
          data.suggestions.forEach(sug => {
            const btn = document.createElement('button');
            btn.textContent = sug;
            btn.dataset.text = sug;
            sugsEl.appendChild(btn);
          });
        }

        if (data.show_qual_map) {
          messagesEl.appendChild(makeQualMapButton());
        }
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

  // Starter chip tap — auto-send
  messagesEl.addEventListener('click', e => {
    const chip = e.target.closest('.starter-chip');
    if (!chip || !chip.dataset.text) return;
    chatInput.value = chip.dataset.text;
    sendBtn.disabled = false;
    submit();
  });

  // Suggestion chip tap — auto-send
  sugsEl.addEventListener('click', e => {
    const btn = e.target.closest('button');
    if (!btn || !btn.dataset.text) return;
    chatInput.value = btn.dataset.text;
    sendBtn.disabled = false;
    submit();
  });

  return el;
}
