/* Landing / chat-home — Finn greeting, ask-me input, subject tiles, starter chips */

import { state }          from '../state.js';
import { go }             from '../router.js';
import { loadWelcomeData } from '../api.js';
import { subject, subjectIconSvg } from '../subjects.js';

const STARTER_CHIPS = [
  "I'm good with my hands",
  "I'm interested in something with computers",
  "I want a job that pays well",
  "I'm not sure what I want yet",
];

function escAttr(s) {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}

function tileHtml(ssa) {
  const s = subject(ssa);
  return `<button class="mtile" data-seed="${escAttr(`I'd like to explore ${s.label} courses`)}" style="--sub:${s.colour}">
    <span class="wm">${subjectIconSvg(ssa)}</span>
    <span class="ic">${subjectIconSvg(ssa)}</span>
    <b>${s.label}</b>
  </button>`;
}

export function WelcomeView() {
  const savedCount = state.saved.items.length;

  const el = document.createElement('div');
  el.className = 'view view-landing';

  el.innerHTML = `
    <header class="la-top">
      <div class="brandlock-a">
        <img src="assets/brand/logo-mark.png" alt="">
        <div class="wm">
          <span class="nm">FutureFinder</span>
          <span class="inst">Greater Manchester IoT</span>
        </div>
      </div>
      <button class="la-saved" aria-label="Saved items">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/>
        </svg>
        <span class="ct" data-saved-count${savedCount === 0 ? ' hidden' : ''}>${savedCount}</span>
      </button>
    </header>

    <div class="la-body">
      <div class="la-hello">
        <h3>Hi, I'm Finn.</h3>
        <p>Tell me what you enjoy — or where you'd like to go next — and I'll find GMIoT courses and show where they lead.</p>
      </div>

      <div class="la-input">
        <input type="text" id="la-text" placeholder="Ask me anything…"
               autocomplete="off" autocorrect="off" spellcheck="false">
        <button class="send" id="la-send" aria-label="Send">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>

      <div class="la-divider">Explore by subject</div>
      <div class="la-tiles" id="la-tiles">
        <div class="la-tile-skeleton"></div>
        <div class="la-tile-skeleton"></div>
        <div class="la-tile-skeleton"></div>
        <div class="la-tile-skeleton"></div>
        <div class="la-tile-skeleton"></div>
        <div class="la-tile-skeleton"></div>
      </div>

      <div class="la-divider">Or just say…</div>
      <div class="la-chips">
        ${STARTER_CHIPS.map(t => `<button class="la-chip" data-seed="${escAttr(t)}">${t}</button>`).join('')}
      </div>
    </div>
  `;

  const textInput = el.querySelector('#la-text');
  const sendBtn   = el.querySelector('#la-send');
  const tilesEl   = el.querySelector('#la-tiles');

  function seedChat(text) {
    go('chat-first', { prefill: text, autosend: true });
  }

  sendBtn.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (text) seedChat(text);
  });

  textInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const text = textInput.value.trim();
      if (text) seedChat(text);
    }
  });

  // Subject tiles and institution strings — built from /api/welcome-data
  loadWelcomeData().then(data => {
    const inst     = data.institution || {};
    const fullName = inst.full_name || 'Greater Manchester Institute of Technology';
    const abbrev   = inst.abbrev   || 'GMIoT';

    el.querySelector('.inst').textContent = fullName;
    el.querySelector('.la-hello p').textContent =
      `Tell me what you enjoy — or where you'd like to go next — and I'll find ${abbrev} courses and show where they lead.`;

    const ssaCodes = (data.ssa_codes || []).slice().sort((a, b) => Number(a) - Number(b));
    tilesEl.innerHTML = ssaCodes.map(tileHtml).join('');
    tilesEl.querySelectorAll('.mtile').forEach(btn => {
      btn.addEventListener('click', () => seedChat(btn.dataset.seed));
    });
  });

  // Starter chips
  el.querySelectorAll('.la-chip').forEach(btn => {
    btn.addEventListener('click', () => seedChat(btn.dataset.seed));
  });

  // Saved-items button
  el.querySelector('.la-saved').addEventListener('click', () => go('saved-list', { backRoute: 'welcome' }));

  return el;
}
