/* Course detail screen */

import { go }                       from '../router.js';
import { state, saveItem, unsaveItem, isSaved } from '../state.js';

const BOOKMARK_EMPTY = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

const BOOKMARK_SAVED = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

function section(className, ...children) {
  const el = document.createElement('div');
  el.className = className;
  children.forEach(c => c && el.appendChild(c));
  return el;
}

function para(className, text) {
  if (!text) return null;
  const el = document.createElement('p');
  el.className = className;
  el.textContent = text;
  return el;
}

function heading(className, text) {
  const el = document.createElement('h3');
  el.className = className;
  el.textContent = text;
  return el;
}

function jobPill(job) {
  const el = document.createElement('span');
  el.className = 'cd-job-pill';
  el.textContent = job.title;
  return el;
}

function jobCard(job) {
  const el = document.createElement('div');
  el.className = 'cd-job-card';
  const title = document.createElement('div');
  title.className = 'cd-job-card-title';
  title.textContent = job.title;
  el.appendChild(title);
  if (job.reasoning) {
    const reason = document.createElement('div');
    reason.className = 'cd-job-card-reason';
    reason.textContent = job.reasoning;
    el.appendChild(reason);
  }
  return el;
}

export function CourseDetailView(slices = {}) {
  const courseId    = slices.courseId;
  const backRoute   = slices.backRoute || 'course-list';
  const courseTitle = slices.courseTitle || '';

  const el = document.createElement('div');
  el.className = 'view view-course-detail';

  // ── Top bar ──────────────────────────────────────────────────────────────
  const topBar = document.createElement('div');
  topBar.className = 'cd-topbar';

  const backBtn = document.createElement('button');
  backBtn.className = 'cd-back';
  backBtn.setAttribute('aria-label', 'Back');
  backBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
    aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg>`;
  backBtn.addEventListener('click', () => go(backRoute));

  const saveBtn = document.createElement('button');
  saveBtn.className = 'cd-save';
  saveBtn.setAttribute('aria-label', 'Save course');
  const updateSaveBtn = () => {
    const saved = isSaved(courseId, 'course');
    saveBtn.innerHTML = saved ? BOOKMARK_SAVED : BOOKMARK_EMPTY;
    saveBtn.setAttribute('aria-pressed', String(saved));
  };
  updateSaveBtn();
  saveBtn.addEventListener('click', () => {
    if (isSaved(courseId, 'course')) {
      unsaveItem(courseId, 'course');
    } else {
      saveItem({ id: courseId, type: 'course', title: courseTitle });
    }
    updateSaveBtn();
    const badge = document.getElementById('saved-count');
    if (badge) badge.textContent = state.saved.items.length;
  });

  topBar.appendChild(backBtn);
  topBar.appendChild(saveBtn);
  el.appendChild(topBar);

  // ── Loading state — replaced once fetch returns ───────────────────────────
  const content = document.createElement('div');
  content.className = 'cd-content';
  content.innerHTML = '<p class="cd-loading">Loading…</p>';
  el.appendChild(content);

  fetch(`/courses/${courseId}/detail`)
    .then(r => {
      if (!r.ok) throw new Error(`/courses/${courseId}/detail ${r.status}`);
      return r.json();
    })
    .then(d => renderDetail(content, d))
    .catch(err => {
      console.error('[course-detail]', err);
      content.innerHTML = '<p class="cd-error">Could not load course details.</p>';
    });

  return el;
}

function renderDetail(content, d) {
  content.innerHTML = '';

  // ── Title block ───────────────────────────────────────────────────────────
  const titleBlock = document.createElement('div');
  titleBlock.className = 'cd-title-block';

  const titleEl = document.createElement('h2');
  titleEl.className = 'cd-title';
  titleEl.textContent = d.course_title;
  titleBlock.appendChild(titleEl);

  const meta = document.createElement('div');
  meta.className = 'cd-meta';
  const metaParts = [d.qual_type, d.level ? `Level ${d.level}` : '', d.provider].filter(Boolean);
  meta.textContent = metaParts.join(' · ');
  titleBlock.appendChild(meta);

  if (d.mode || d.duration || d.campus) {
    const sub = document.createElement('div');
    sub.className = 'cd-sub';
    sub.textContent = [d.mode, d.duration, d.campus].filter(Boolean).join(' · ');
    titleBlock.appendChild(sub);
  }

  content.appendChild(titleBlock);

  // ── Preview ───────────────────────────────────────────────────────────────
  if (d.preview) content.appendChild(para('cd-preview', d.preview));

  // ── Overview ──────────────────────────────────────────────────────────────
  if (d.overview) {
    content.appendChild(heading('cd-section-heading', 'Overview'));
    content.appendChild(para('cd-body', d.overview));
  }

  // ── "Course details" expansion (entry_requirements + progression) ─────────
  if (d.entry_requirements || d.progression || d.content) {
    const detailsToggle = document.createElement('button');
    detailsToggle.className = 'cd-expand-btn';
    detailsToggle.textContent = 'Course details';
    detailsToggle.dataset.open = 'false';

    const detailsPanel = document.createElement('div');
    detailsPanel.className = 'cd-expand-panel';
    detailsPanel.hidden = true;

    if (d.content) {
      detailsPanel.appendChild(heading('cd-section-heading', 'What you'll learn'));
      detailsPanel.appendChild(para('cd-body', d.content));
    }
    if (d.entry_requirements) {
      detailsPanel.appendChild(heading('cd-section-heading', 'Entry requirements'));
      detailsPanel.appendChild(para('cd-body', d.entry_requirements));
    }
    if (d.progression) {
      detailsPanel.appendChild(heading('cd-section-heading', 'Progression'));
      detailsPanel.appendChild(para('cd-body', d.progression));
    }

    detailsToggle.addEventListener('click', () => {
      const open = detailsToggle.dataset.open === 'true';
      detailsToggle.dataset.open = String(!open);
      detailsToggle.textContent = open ? 'Course details' : 'Less';
      detailsPanel.hidden = open;
    });

    content.appendChild(detailsToggle);
    content.appendChild(detailsPanel);
  }

  // ── Course URL ────────────────────────────────────────────────────────────
  if (d.course_url) {
    const urlEl = document.createElement('a');
    urlEl.className = 'cd-course-url';
    urlEl.href = d.course_url;
    urlEl.target = '_blank';
    urlEl.rel = 'noopener noreferrer';
    urlEl.textContent = 'View course on GMIoT website →';
    content.appendChild(urlEl);
  }

  // ── Career pathways ───────────────────────────────────────────────────────
  if (d.pathways) {
    const pathwaysSection = document.createElement('div');
    pathwaysSection.className = 'cd-pathways';

    content.appendChild(document.createElement('hr'));
    content.lastChild.className = 'cd-divider';

    const pathHeading = heading('cd-section-heading', 'Where this could lead');
    content.appendChild(pathHeading);

    // Brief state: narrative_short + card_jobs as pills
    const briefEl = document.createElement('div');
    briefEl.className = 'cd-pathways-brief';

    if (d.pathways.narrative_short) {
      briefEl.appendChild(para('cd-body', d.pathways.narrative_short));
    }
    if (d.pathways.card_jobs?.length) {
      const pillsEl = document.createElement('div');
      pillsEl.className = 'cd-job-pills';
      d.pathways.card_jobs.forEach(j => pillsEl.appendChild(jobPill(j)));
      briefEl.appendChild(pillsEl);
    }

    // Full state: narrative + curated_jobs as cards
    const fullEl = document.createElement('div');
    fullEl.className = 'cd-pathways-full';
    fullEl.hidden = true;

    if (d.pathways.narrative) {
      fullEl.appendChild(para('cd-body', d.pathways.narrative));
    }
    if (d.pathways.curated_jobs?.length) {
      d.pathways.curated_jobs.forEach(j => fullEl.appendChild(jobCard(j)));
    }

    // Toggle button
    const pathToggle = document.createElement('button');
    pathToggle.className = 'cd-expand-btn cd-expand-btn--pathways';
    pathToggle.textContent = 'Full career view';
    pathToggle.dataset.open = 'false';
    pathToggle.addEventListener('click', () => {
      const open = pathToggle.dataset.open === 'true';
      pathToggle.dataset.open = String(!open);
      pathToggle.textContent = open ? 'Full career view' : 'Brief view';
      briefEl.hidden = !open;
      fullEl.hidden = open;
    });

    pathwaysSection.appendChild(briefEl);
    pathwaysSection.appendChild(fullEl);
    pathwaysSection.appendChild(pathToggle);
    content.appendChild(pathwaysSection);
  }
}
