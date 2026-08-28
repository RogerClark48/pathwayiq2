/* Landing / chat-home — Finn greeting, ask-me input, subject tiles, starter chips */

import { state, resetSession } from '../state.js';
import { go }             from '../router.js';
import { loadWelcomeData, cartoTileLayer } from '../api.js';
import { subject, subjectHex, subjectIconSvg } from '../subjects.js';

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

// ── Module-level hero map (cleaned up on re-entry) ────────────────────────────
let _heroMap = null;

function cleanupHero() {
  if (_heroMap) {
    try { _heroMap.remove(); } catch (_) {}
    _heroMap = null;
  }
}

// ── Landing hero — populates #desktop-right at ≥1200px ───────────────────────
function buildHero(data, seedChat) {
  cleanupHero();

  const ssaCodes      = (data.ssa_codes || []).slice().sort((a, b) => Number(a) - Number(b));
  const campusPins    = data.campus_pins    || [];
  const totalCourses  = data.total_courses  || 0;
  const totalCampuses = data.total_campuses || campusPins.length;

  const hero = document.createElement('div');
  hero.className = 'la-hero';
  hero.innerHTML = `
    <div id="la-hero-map" class="la-hero-map"></div>
    <div class="la-hero-overlay">
      <div class="la-hero-hl">
        <span>${totalCourses} courses across ${totalCampuses} campuses</span>
        <em>Ask Finn where to start</em>
      </div>
      <div class="la-hero-tiles" id="la-hero-tiles">
        ${ssaCodes.map(tileHtml).join('')}
      </div>
    </div>
`;

  hero.querySelectorAll('.mtile').forEach(btn => {
    btn.addEventListener('click', () => seedChat(btn.dataset.seed));
  });

  // Leaflet init deferred so the element is in the DOM first
  requestAnimationFrame(() => {
    const mapEl = hero.querySelector('#la-hero-map');
    if (!mapEl || typeof L === 'undefined') return;

    _heroMap = L.map(mapEl, {
      center:             [53.48, -2.24],
      zoom:               10,
      zoomControl:        false,
      attributionControl: true,
      dragging:           false,
      touchZoom:          false,
      scrollWheelZoom:    false,
      doubleClickZoom:    false,
      boxZoom:            false,
      keyboard:           false,
      tap:                false,
    });

    cartoTileLayer({ maxZoom: 19 }).addTo(_heroMap);

    const bounds = [];
    campusPins.forEach((pin, i) => {
      if (pin.lat == null || pin.lng == null) return;
      const hex  = subjectHex(pin.ssa_code);
      const icon = subjectIconSvg(pin.ssa_code);
      const html = `<div class="ffpin la-hero-pin" style="--c:${hex};animation-delay:${i * 180}ms"><span class="drop">${icon}</span></div>`;
      const marker = L.marker([pin.lat, pin.lng], {
        icon: L.divIcon({ html, iconSize: [30, 38], iconAnchor: [15, 34], className: 'ffpin-host' }),
        interactive: false,
      });
      marker.addTo(_heroMap);
      bounds.push([pin.lat, pin.lng]);
    });

    if (bounds.length > 1) {
      _heroMap.fitBounds(bounds, { padding: [60, 60] });
    } else if (bounds.length === 1) {
      _heroMap.setView(bounds[0], 12);
    }

    setTimeout(() => _heroMap?.invalidateSize(), 60);
  });

  return hero;
}

// ── Shared video modal (same pattern as job-detail) ───────────────────────────

function openVideoModal(videoId, title) {
  const modal = document.createElement('div');
  modal.className = 'video-modal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-label', title);

  const panel = document.createElement('div');
  panel.className = 'video-modal-panel';

  const header = document.createElement('div');
  header.className = 'video-modal-header';

  const titleEl = document.createElement('span');
  titleEl.className = 'video-modal-title';
  titleEl.textContent = title;

  const closeBtn = document.createElement('button');
  closeBtn.className = 'video-modal-close';
  closeBtn.setAttribute('aria-label', 'Close');
  closeBtn.textContent = '✕';

  header.appendChild(titleEl);
  header.appendChild(closeBtn);

  const frame = document.createElement('iframe');
  frame.className = 'video-modal-frame';
  frame.src = `https://www.youtube.com/embed/${videoId}?autoplay=1`;
  frame.allow = 'autoplay; encrypted-media; fullscreen';
  frame.setAttribute('allowfullscreen', '');

  panel.appendChild(header);
  panel.appendChild(frame);
  modal.appendChild(panel);

  function close() {
    frame.src = '';
    modal.remove();
    document.removeEventListener('keydown', onKey);
  }
  function onKey(e) { if (e.key === 'Escape') close(); }

  closeBtn.addEventListener('click', close);
  modal.addEventListener('click', e => { if (e.target === modal) close(); });
  document.addEventListener('keydown', onKey);
  document.body.appendChild(modal);
}

export function WelcomeView() {
  cleanupHero();

  const savedCount = state.saved.items.length;
  const isHeroBreakpoint = () => window.innerWidth >= 1200;

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
      <div class="la-top-actions">
        <button class="la-icon-btn" id="la-reset" aria-label="Start again">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 20h9"/>
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
        </button>
        <button class="la-saved" aria-label="Saved items">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/>
          </svg>
          <span class="ct" data-saved-count${savedCount === 0 ? ' hidden' : ''}>${savedCount}</span>
        </button>
      </div>
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

      <div class="la-divider la-divider-subjects">Explore by subject</div>
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

      <button class="la-qual-video" aria-label="Play: Understanding Level 4 &amp; 5 qualifications">
        <div class="la-qual-video-thumb"
             style="background-image:url('https://img.youtube.com/vi/bUegAJvKB6A/hqdefault.jpg')">
          <div class="la-qual-video-play">
            <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
              <circle cx="14" cy="14" r="14" fill="rgba(0,0,0,.55)"/>
              <polygon points="11,8 22,14 11,20" fill="white"/>
            </svg>
          </div>
        </div>
        <div class="la-qual-video-text">
          <span class="la-qual-video-eyebrow">Video guide</span>
          <span class="la-qual-video-title">Understanding Level 4 &amp; 5 qualifications</span>
        </div>
      </button>
    </div>
  `;

  const textInput = el.querySelector('#la-text');
  const sendBtn   = el.querySelector('#la-send');
  const tilesEl   = el.querySelector('#la-tiles');
  const tilesDivider = el.querySelector('.la-divider-subjects');

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

  const rightPanel = document.getElementById('desktop-right');

  function applyLayout(data) {
    const inst     = data.institution || {};
    const fullName = inst.full_name || 'Greater Manchester Institute of Technology';
    const abbrev   = inst.abbrev   || 'GMIoT';
    const ssaCodes = (data.ssa_codes || []).slice().sort((a, b) => Number(a) - Number(b));

    el.querySelector('.inst').textContent = fullName;
    el.querySelector('.la-hello p').textContent =
      `Tell me what you enjoy — or where you'd like to go next — and I'll find ${abbrev} courses and show where they lead.`;

    if (isHeroBreakpoint() && rightPanel) {
      // Desktop: tiles go into the right-panel hero; hide them from the left rail
      tilesEl.style.display    = 'none';
      tilesDivider.style.display = 'none';
      rightPanel.innerHTML = '';
      rightPanel.appendChild(buildHero(data, seedChat));
    } else {
      // Mobile/tablet: tiles stay inline in the left rail
      tilesEl.style.display    = '';
      tilesDivider.style.display = '';
      tilesEl.innerHTML = ssaCodes.map(tileHtml).join('');
      tilesEl.querySelectorAll('.mtile').forEach(btn => {
        btn.addEventListener('click', () => seedChat(btn.dataset.seed));
      });
    }
  }

  loadWelcomeData().then(data => {
    if (!el.isConnected) return;
    applyLayout(data);

    // Live reflow across the 1200px boundary
    const mq = window.matchMedia('(min-width: 1200px)');
    function onBreakpoint() {
      if (!el.isConnected) { mq.removeEventListener('change', onBreakpoint); return; }
      applyLayout(data);
    }
    mq.addEventListener('change', onBreakpoint);
  });

  // Starter chips
  el.querySelectorAll('.la-chip').forEach(btn => {
    btn.addEventListener('click', () => seedChat(btn.dataset.seed));
  });

  // Qualifications video card
  el.querySelector('.la-qual-video').addEventListener('click', () =>
    openVideoModal('bUegAJvKB6A', 'Understanding Level 4 & 5 qualifications'));

  // Saved-items button
  el.querySelector('.la-saved').addEventListener('click', () => go('saved-list', { backRoute: 'welcome' }));

  // Reset button — same confirm flow as chat-head #chat-new
  el.querySelector('#la-reset').addEventListener('click', () => {
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

  return el;
}
