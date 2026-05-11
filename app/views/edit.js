/* Edit screen — subject filter only */

import { state } from '../state.js';
import { go } from '../router.js';
import { getWelcomeData } from '../api.js';
import { SSA_LABELS } from '../ssa.js';

function ssaCount(data, ssa) {
  return Object.values(data.counts).reduce((sum, row) => sum + (row[ssa] || 0), 0);
}

function totalCount(data, subjects) {
  if (subjects.size === 0) return 0;
  let total = 0;
  subjects.forEach(ssa => { total += ssaCount(data, ssa); });
  return total;
}

export function EditView() {
  const data = getWelcomeData();
  if (!data) {
    const el = document.createElement('div');
    el.className = 'view';
    el.innerHTML = '<div class="view-body"><p class="prompt-label">Loading…</p></div>';
    return el;
  }

  const snapSubjects = new Set(state.filter.subjects);

  const subjects = data.ssa_codes
    .map(ssa => {
      const label = SSA_LABELS[ssa];
      if (!label) { console.warn(`EditView: unknown SSA code "${ssa}" — skipped`); return null; }
      return { ssa, label };
    })
    .filter(Boolean)
    .sort((a, b) => Number(a.ssa) - Number(b.ssa));

  const el = document.createElement('div');
  el.className = 'view view-edit';

  function subjectChipsHtml() {
    return subjects.map(({ ssa, label }) => {
      const n   = ssaCount(data, ssa);
      const sel = state.filter.subjects.has(ssa);
      return `
        <button class="subject-chip${sel ? ' selected' : ''}"
                data-ssa="${ssa}"
                aria-pressed="${sel}">
          <span class="chip-tick" aria-hidden="true"></span>
          <span class="chip-content">
            <span class="chip-label">${label}</span>
            <span class="chip-count">${n} course${n !== 1 ? 's' : ''}</span>
          </span>
        </button>
      `;
    }).join('');
  }

  el.innerHTML = `
    <div class="view-body">
      <div class="view-top-nav">
        <button class="btn-icon btn-cancel" aria-label="Cancel">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
        <span class="view-title">Edit subjects</span>
      </div>

      <p class="prompt-label">Which subjects interest you?</p>
      <p class="prompt-sub">Pick one or more</p>
      <div class="subject-grid" id="edit-subject-grid">
        ${subjectChipsHtml()}
      </div>

      <p class="course-count" id="edit-count">${totalCount(data, state.filter.subjects)} courses available</p>
    </div>

    <div class="view-footer">
      <button class="btn-primary${totalCount(data, state.filter.subjects) === 0 ? ' btn-disabled' : ''}"
              id="btn-done"
              ${totalCount(data, state.filter.subjects) === 0 ? 'disabled' : ''}>
        Done
      </button>
    </div>
  `;

  const subjectGrid = el.querySelector('#edit-subject-grid');
  const countEl     = el.querySelector('#edit-count');
  const btnDone     = el.querySelector('#btn-done');

  function refreshCount() {
    const n = totalCount(data, state.filter.subjects);
    countEl.textContent = `${n} courses available`;
    btnDone.disabled = n === 0;
    btnDone.classList.toggle('btn-disabled', n === 0);
  }

  subjectGrid.querySelectorAll('.subject-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const sel = state.filter.subjects.has(chip.dataset.ssa);
      if (sel) {
        state.filter.subjects.delete(chip.dataset.ssa);
      } else {
        state.filter.subjects.add(chip.dataset.ssa);
      }
      chip.classList.toggle('selected', !sel);
      chip.setAttribute('aria-pressed', String(!sel));
      refreshCount();
    });
  });

  btnDone.addEventListener('click', () => go('start'));

  el.querySelector('.btn-cancel').addEventListener('click', () => {
    state.filter.subjects.clear();
    snapSubjects.forEach(s => state.filter.subjects.add(s));
    go('start');
  });

  return el;
}
