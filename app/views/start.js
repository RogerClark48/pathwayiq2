/* Start screen — hub view */

import { state, clearFilter } from '../state.js';
import { go } from '../router.js';
import { getWelcomeData } from '../api.js';
import { SSA_LABELS } from '../ssa.js';

// Suggestions keyed by SSA code (string). Two types per code:
// navigational (narrow the list) and exploratory (fact-finding).
const SUGGESTIONS_BY_SSA = {
  '1': [ // Health & Care
    'Show me Level 4 Health & Care courses',
    'Which are available part-time?',
    'What clinical roles does this lead to?',
    'What salary range can I expect?',
    'Are there apprenticeship routes in health?',
  ],
  '4': [ // Engineering & Manufacturing
    'Show me HNCs in Engineering',
    'Which involve hands-on workshop time?',
    'Degree apprenticeships in Engineering',
    'What jobs pay well after this?',
    'Which lead to engineering design roles?',
    'Which courses are part-time?',
  ],
  '5': [ // Construction & Built Environment
    'Show me Level 4 Construction courses',
    'Is there an apprenticeship route?',
    'Which involve site visits or placements?',
    'What project management roles does this lead to?',
    'What salary range can I expect?',
  ],
  '6': [ // Computing & IT
    'Show me HNCs in Digital & Tech',
    'Degree apprenticeships in tech',
    'Which cover cybersecurity or networking?',
    'What are the graduate job options?',
    'Which lead to software engineering roles?',
    'Which are available part-time?',
  ],
  '9': [ // Creative & Media
    'Show me Level 4 Creative courses',
    'Which focus on digital production?',
    'What careers does Creative Media open up?',
    'Which involve industry placements?',
    'What salary range can I expect?',
  ],
  '15': [ // Business & Law
    'Show me HNCs in Business',
    'Apprenticeships in Business & Law',
    'What management roles does this lead to?',
    'Which include professional qualifications?',
    'Which are available part-time?',
  ],
  '99': [ // Sustainability
    'Show me all Sustainability courses',
    'What level are these courses?',
    'What careers does Sustainability lead to?',
    'How does this link with Engineering or Construction?',
    'Which involve practical project work?',
  ],
};

// Deterministic top-margin offsets per chip position — creates scattered feel.
const CHIP_OFFSETS = [0, 8, 0, 12, 4, 6];

function getChips(subjects) {
  const seen  = new Set();
  const chips = [];
  for (const ssa of subjects) {
    for (const text of (SUGGESTIONS_BY_SSA[ssa] || [])) {
      if (!seen.has(text)) { seen.add(text); chips.push(text); }
    }
  }
  return chips.slice(0, 6);
}

function courseCount(data, subjects) {
  if (!data || subjects.size === 0) return 0;
  let total = 0;
  subjects.forEach(ssa => {
    total += Object.values(data.counts).reduce((sum, row) => sum + (row[ssa] || 0), 0);
  });
  return total;
}

function savedDescription() {
  const items   = state.saved.items;
  if (items.length === 0) return 'You have no saved items';
  const courses = items.filter(i => i.type === 'course').length;
  const careers = items.filter(i => i.type === 'career').length;
  const parts   = [];
  if (courses > 0) parts.push(`${courses} course${courses !== 1 ? 's' : ''}`);
  if (careers > 0) parts.push(`${careers} career${careers !== 1 ? 's' : ''}`);
  return parts.join(' · ');
}

export function StartView() {
  const data     = getWelcomeData();
  const subjects = state.filter.subjects;
  const desc     = [...subjects].map(ssa => SSA_LABELS[ssa] || ssa).join(', ');
  const count    = courseCount(data, subjects);
  const chips    = getChips(subjects);

  const el = document.createElement('div');
  el.className = 'view view-start';

  const chipsHtml = chips.map((text, i) => {
    const offset = CHIP_OFFSETS[i % CHIP_OFFSETS.length];
    // Encode for safe attribute; chip text has no double-quotes so this is safe.
    return `<button class="suggestion-chip" data-text="${text}" style="margin-top:${offset}px">${text}</button>`;
  }).join('') + `<button class="suggestion-chip suggestion-chip--more">More…</button>`;

  el.innerHTML = `
    <div class="start-status" id="start-status">
      <div class="status-row status-row--top">
        <span class="status-desc">${desc || 'No subjects selected'}</span>
        <button class="btn-icon status-pencil" id="btn-edit" aria-label="Edit subjects">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
        </button>
      </div>
      <div class="status-row status-row--count">
        <span class="status-count">${count} course${count !== 1 ? 's' : ''}</span>
        <svg class="status-chevron" width="18" height="18" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
      </div>
    </div>

    <div class="start-divider"></div>

    <div class="start-chat">
      <p class="chat-prompt">Or ask anything about courses or careers</p>
      <input type="text" class="chat-input" id="chat-input"
             placeholder="Type a question…"
             autocomplete="off" autocorrect="off" spellcheck="false">
      <div class="suggestion-chips" id="suggestion-chips">
        ${chipsHtml}
      </div>
    </div>

    <div class="start-divider"></div>

    <div class="start-saved" id="start-saved">
      <span class="saved-label">Saved</span>
      <span class="saved-detail">${savedDescription()}</span>
      <svg class="saved-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
    </div>

    <div class="start-footer">
      <a href="#" id="link-start-again" class="start-again">Start again</a>
    </div>
  `;

  // Status block — whole block tappable for browse, pencil routes to edit.
  el.querySelector('#start-status').addEventListener('click', () => {
    alert('Course list — coming soon');
  });
  el.querySelector('#btn-edit').addEventListener('click', e => {
    e.stopPropagation();
    go('edit');
  });

  // Chat input — phase 1 no-op on submit.
  const chatInput = el.querySelector('#chat-input');
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') e.preventDefault(); // submission is wired in phase 2
  });

  // Suggestion chips — drop text into input, do not auto-submit.
  el.querySelector('#suggestion-chips').addEventListener('click', e => {
    const chip = e.target.closest('.suggestion-chip');
    if (!chip) return;
    if (chip.classList.contains('suggestion-chip--more')) {
      alert('More suggestions — coming soon');
      return;
    }
    chatInput.value = chip.dataset.text;
    chatInput.focus();
  });

  // Saved row stub.
  el.querySelector('#start-saved').addEventListener('click', () => {
    alert('Saved items — coming soon');
  });

  // Start again — wipe filter and restart welcome.
  el.querySelector('#link-start-again').addEventListener('click', e => {
    e.preventDefault();
    clearFilter();
    go('welcome');
  });

  return el;
}
