/* Career view — subject-hue lanes, tinted down the list */

import { go }       from '../router.js';
import { logEvent } from '../analytics.js';
import { subject, subjectIconSvg } from '../subjects.js';

// Signpost icon for the header watermark (route / destination metaphor)
const SIGNPOST_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
  stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M3 3v18"/>
  <path d="M17.5 9H3l5-6h9.5a1 1 0 0 1 1 1v4a1 1 0 0 1-1 1Z"/>
  <path d="M3 15h13.5a1 1 0 0 0 1-1v-4a1 1 0 0 0-1-1H3"/>
</svg>`;

export function CareerView(slices = {}) {
  const { courseId, courseTitle, ssa, pathways, backRoute, backSlices } = slices;
  return buildCareerEl({
    courseId, courseTitle, ssa, pathways,
    onBack:     () => go(backRoute, backSlices || {}),
    onJobClick: (job) => go('job-detail', {
      jobId:      job.job_id,
      jobTitle:   job.title,
      ssa,
      backRoute:  'career-view',
      backSlices: { courseId, courseTitle, ssa, pathways, backRoute, backSlices },
    }),
  });
}

// ── Element builder — used by both CareerView (routing) and desktop split ─────

export function buildCareerEl({ courseId, courseTitle, ssa, pathways, onBack, onJobClick }) {
  const s        = subject(ssa);
  const p        = pathways || {};
  const jobById  = Object.fromEntries((p.curated_jobs || []).map(j => [j.job_id, j]));
  const groups   = (p.node_groups || [])
    .map(g => ({
      label: g.label,
      jobs:  (g.job_ids || []).map(id => jobById[id]).filter(Boolean),
    }))
    .filter(g => g.jobs.length);
  const totalRoles = groups.reduce((n, g) => n + g.jobs.length, 0);

  const el = document.createElement('div');
  el.className = 'view view-career';
  el.style.setProperty('--sub', s.colour);

  // ── Coloured header ───────────────────────────────────────────────────────
  const head = document.createElement('div');
  head.className = 'cv-head';
  head.style.setProperty('--sub', s.colour);

  head.innerHTML = `
    <span class="wm" aria-hidden="true">${SIGNPOST_SVG}</span>
    <div class="cv-bar">
      <button class="ic cv-back" aria-label="Back">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="15 18 9 12 15 6"/>
        </svg>
      </button>
      <button class="ic cv-finn" aria-label="Back to chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>
    </div>
    <h4>Where this could lead</h4>
    <div class="sub">From ${courseTitle} · ${totalRoles} role${totalRoles !== 1 ? 's' : ''}</div>`;

  head.querySelector('.cv-back').addEventListener('click', onBack);
  head.querySelector('.cv-finn').addEventListener('click', () => go('chat-first'));

  el.appendChild(head);

  // ── Scrolling body ────────────────────────────────────────────────────────
  const body = document.createElement('div');
  body.className = 'cv-body';

  if (p.narrative_short) {
    const intro = document.createElement('p');
    intro.className = 'cv-intro';
    intro.textContent = p.narrative_short;
    body.appendChild(intro);
  }

  // Course trunk chip
  const trunk = document.createElement('div');
  trunk.className = 'cv-course';
  trunk.textContent = courseTitle;
  body.appendChild(trunk);

  body.appendChild(Object.assign(document.createElement('div'), { className: 'cv-spine' }));

  // Lanes
  const lanesEl = document.createElement('div');
  lanesEl.className = 'cv-lanes';

  if (groups.length) {
    groups.forEach((group, i) => {
      const lane = document.createElement('div');
      lane.className = 'cv-lane';
      // Tint ramp: 100% → 84% → 68% … floored at 50% so faintest lane stays legible
      const pct = Math.max(50, 100 - i * 16);
      lane.style.setProperty('--cl', `color-mix(in srgb, var(--sub) ${pct}%, var(--bg))`);

      const ln = document.createElement('div');
      ln.className = 'ln';
      ln.innerHTML = `
        <span class="d"></span>
        <b>${group.label}</b>
        <span class="ct">${group.jobs.length} role${group.jobs.length !== 1 ? 's' : ''}</span>`;
      lane.appendChild(ln);

      const rolesEl = document.createElement('div');
      rolesEl.className = 'roles';

      group.jobs.forEach((job, ji) => {
        logEvent('career_impression', 'job', job.job_id, job.title);

        const chip = document.createElement('span');
        chip.textContent = job.title;
        if (ji === 0) chip.className = 'lead';

        chip.addEventListener('click', () => onJobClick(job));

        rolesEl.appendChild(chip);
      });

      lane.appendChild(rolesEl);
      lanesEl.appendChild(lane);
    });
  } else {
    const empty = document.createElement('p');
    empty.className = 'cv-empty';
    empty.textContent = 'Career destinations are not available for this course yet.';
    lanesEl.appendChild(empty);
  }

  body.appendChild(lanesEl);
  el.appendChild(body);

  return el;
}
