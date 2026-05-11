/* Screen 2 — subject multi-select */

import { state, toggleSubject } from '../state.js';
import { go } from '../router.js';
import { getWelcomeData } from '../api.js';
import { SSA_LABELS } from '../ssa.js';

function courseCount(data, qual, subjects) {
  if (!qual || subjects.size === 0) return 0;
  const row = (data.counts[qual] || {});
  let total = 0;
  subjects.forEach(ssa => { total += row[ssa] || 0; });
  return total;
}

export function SubjectView() {
  const data = getWelcomeData();
  if (!data) {
    // Should not happen in normal flow; guard only.
    const el = document.createElement('div');
    el.className = 'view';
    el.innerHTML = '<div class="view-body"><p class="prompt-label">Loading…</p></div>';
    return el;
  }

  // Resolve SSA codes → labels, warn on unknowns.
  const subjects = data.ssa_codes
    .map(ssa => {
      const label = SSA_LABELS[ssa];
      if (!label) { console.warn(`SubjectView: unknown SSA code "${ssa}" — skipped`); return null; }
      return { ssa, label };
    })
    .filter(Boolean)
    .sort((a, b) => Number(a.ssa) - Number(b.ssa));

  const el = document.createElement('div');
  el.className = 'view view-subject';

  function totalCount() {
    return courseCount(data, state.filter.qual, state.filter.subjects);
  }

  function chipHtml({ ssa, label }) {
    const n   = (data.counts[state.filter.qual] || {})[ssa] || 0;
    const sel = state.filter.subjects.has(ssa);
    const dis = n === 0;
    return `
      <button class="subject-chip${sel ? ' selected' : ''}${dis ? ' disabled' : ''}"
              data-ssa="${ssa}"
              ${dis ? 'disabled aria-disabled="true"' : ''}
              aria-pressed="${sel}">
        <span class="chip-tick" aria-hidden="true"></span>
        <span class="chip-content">
          <span class="chip-label">${label}</span>
          <span class="chip-count">${n} course${n !== 1 ? 's' : ''}</span>
        </span>
      </button>
    `;
  }

  el.innerHTML = `
    <div class="view-body">
      <div class="qual-strip">
        <div class="qual-strip-left">
          <span class="qual-strip-hint">Looking for</span>
          <span class="qual-strip-value">${state.filter.qual}</span>
        </div>
        <button class="qual-strip-edit" aria-label="Change qualification">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
        </button>
      </div>

      <p class="prompt-label">Which subjects interest you?</p>
      <p class="prompt-sub">Pick one or more</p>

      <div class="subject-grid">
        ${subjects.map(chipHtml).join('')}
      </div>

      <p class="course-count" id="course-count">${totalCount()} courses available</p>
    </div>

    <div class="view-footer">
      <button class="btn-primary${state.filter.subjects.size === 0 ? ' btn-disabled' : ''}"
              id="btn-show"
              ${state.filter.subjects.size === 0 ? 'disabled' : ''}>
        Show me →
      </button>
    </div>
  `;

  const countEl = el.querySelector('#course-count');
  const btnShow = el.querySelector('#btn-show');

  el.querySelectorAll('.subject-chip:not([disabled])').forEach(chip => {
    chip.addEventListener('click', () => {
      toggleSubject(chip.dataset.ssa);
      const sel = state.filter.subjects.has(chip.dataset.ssa);
      chip.classList.toggle('selected', sel);
      chip.setAttribute('aria-pressed', String(sel));
      countEl.textContent = `${totalCount()} courses available`;
      const hasAny = state.filter.subjects.size > 0;
      btnShow.disabled = !hasAny;
      btnShow.classList.toggle('btn-disabled', !hasAny);
    });
  });

  el.querySelector('.qual-strip-edit').addEventListener('click', () => go('welcome'));

  btnShow.addEventListener('click', () => {
    if (state.filter.subjects.size > 0) go('start');
  });

  return el;
}
