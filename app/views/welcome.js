/* Screen 1 — subject multi-select (only welcome screen) */

import { state, toggleSubject } from '../state.js';
import { go } from '../router.js';
import { loadWelcomeData } from '../api.js';
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

export function WelcomeView() {
  const el = document.createElement('div');
  el.className = 'view view-welcome';

  el.innerHTML = `
    <div class="view-body">
      <p class="welcome-intro">
        FutureFinder helps you find out what course at the Greater Manchester
        Institute of Technology (GMIoT) you might want to study, and where it
        could take you. Start by telling us which subjects interest you.
      </p>

      <p class="prompt-label">Which subjects interest you?</p>
      <p class="prompt-sub">Pick one or more</p>

      <div class="subject-grid" id="subject-grid">
        <div class="chip-skeleton"></div>
        <div class="chip-skeleton"></div>
        <div class="chip-skeleton"></div>
        <div class="chip-skeleton"></div>
        <div class="chip-skeleton"></div>
        <div class="chip-skeleton"></div>
      </div>

      <p class="course-count" id="course-count"></p>
    </div>

    <div class="view-footer">
      <button class="btn-primary btn-disabled" id="btn-show" disabled>Show me →</button>
    </div>
  `;

  const subjectGrid = el.querySelector('#subject-grid');
  const countEl     = el.querySelector('#course-count');
  const btnShow     = el.querySelector('#btn-show');

  loadWelcomeData().then(data => {
    const subjects = data.ssa_codes
      .map(ssa => {
        const label = SSA_LABELS[ssa];
        if (!label) { console.warn(`WelcomeView: unknown SSA code "${ssa}" — skipped`); return null; }
        return { ssa, label };
      })
      .filter(Boolean)
      .sort((a, b) => Number(a.ssa) - Number(b.ssa));

    function chipHtml({ ssa, label }) {
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
    }

    subjectGrid.innerHTML = subjects.map(chipHtml).join('');

    const n = totalCount(data, state.filter.subjects);
    countEl.textContent = state.filter.subjects.size > 0 ? `${n} courses available` : '';

    if (state.filter.subjects.size > 0) {
      btnShow.disabled = false;
      btnShow.classList.remove('btn-disabled');
    }

    subjectGrid.querySelectorAll('.subject-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        toggleSubject(chip.dataset.ssa);
        const sel = state.filter.subjects.has(chip.dataset.ssa);
        chip.classList.toggle('selected', sel);
        chip.setAttribute('aria-pressed', String(sel));
        const count = totalCount(data, state.filter.subjects);
        const hasAny = state.filter.subjects.size > 0;
        countEl.textContent = hasAny ? `${count} courses available` : '';
        btnShow.disabled = !hasAny;
        btnShow.classList.toggle('btn-disabled', !hasAny);
      });
    });
  });

  btnShow.addEventListener('click', () => {
    if (state.filter.subjects.size > 0) go('start');
  });

  return el;
}
