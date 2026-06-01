/* Job detail screen */

import { go }                       from '../router.js';
import { state, saveItem, unsaveItem, isSaved } from '../state.js';
import { logEvent } from '../analytics.js';

const BOOKMARK_EMPTY = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

const BOOKMARK_SAVED = `<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

function heading(className, text) {
  const el = document.createElement('h3');
  el.className = className;
  el.textContent = text;
  return el;
}

function para(className, text) {
  if (!text) return null;
  const el = document.createElement('p');
  el.className = className;
  el.textContent = text;
  return el;
}

function divider() {
  const hr = document.createElement('hr');
  hr.className = 'cd-divider';
  return hr;
}

export function JobDetailView(slices = {}) {
  const jobId      = slices.jobId;
  const jobTitle   = slices.jobTitle || '';
  const backRoute  = slices.backRoute  || 'course-detail';
  const backSlices = slices.backSlices || {};

  const el = document.createElement('div');
  el.className = 'view view-job-detail';

  // ── Top bar ────────────────────────────────────────────────────────────────
  const topBar = document.createElement('div');
  topBar.className = 'cd-topbar';

  const backBtn = document.createElement('button');
  backBtn.className = 'cd-back';
  backBtn.setAttribute('aria-label', 'Back');
  backBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
    aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg>`;
  backBtn.addEventListener('click', () => go(backRoute, backSlices));

  const saveBtn = document.createElement('button');
  saveBtn.className = 'cd-save';
  saveBtn.setAttribute('aria-label', 'Save job');
  const updateSaveBtn = () => {
    const saved = isSaved(jobId, 'job');
    saveBtn.innerHTML = saved ? BOOKMARK_SAVED : BOOKMARK_EMPTY;
    saveBtn.setAttribute('aria-pressed', String(saved));
  };
  updateSaveBtn();
  saveBtn.addEventListener('click', () => {
    if (isSaved(jobId, 'job')) {
      unsaveItem(jobId, 'job');
    } else {
      saveItem({ id: jobId, type: 'job', title: jobTitle });
    }
    updateSaveBtn();
    const badge = document.getElementById('saved-count');
    if (badge) badge.textContent = state.saved.items.length;
  });

  topBar.appendChild(backBtn);
  topBar.appendChild(saveBtn);
  el.appendChild(topBar);

  // ── Content ────────────────────────────────────────────────────────────────
  const content = document.createElement('div');
  content.className = 'cd-content';
  content.innerHTML = '<p class="cd-loading">Loading…</p>';
  el.appendChild(content);

  logEvent('career_detail_open', 'job', jobId, jobTitle);

  fetch(`/jobs/${jobId}`)
    .then(r => {
      if (!r.ok) throw new Error(`/jobs/${jobId} ${r.status}`);
      return r.json();
    })
    .then(d => {
      renderMain(content, d);
      wireRevealButton(
        content.querySelector('#jd-progression-section'),
        'Career path',
        (panel) => loadExplain(panel, jobId),
      );
      wireRevealButton(
        content.querySelector('#jd-courses-section'),
        'Courses that lead here',
        (panel) => loadCourses(panel, jobId),
      );
    })
    .catch(err => {
      console.error('[job-detail]', err);
      content.innerHTML = '<p class="cd-error">Could not load job details.</p>';
    });

  return el;
}

function renderMain(content, d) {
  content.innerHTML = '';

  // ── Title block ─────────────────────────────────────────────────────────
  const titleBlock = document.createElement('div');
  titleBlock.className = 'cd-title-block';

  const titleEl = document.createElement('h2');
  titleEl.className = 'cd-title';
  titleEl.textContent = d.title;
  titleBlock.appendChild(titleEl);

  const metaParts = [];
  if (d.source) {
    const badge = document.createElement('span');
    badge.className = `jd-source-badge jd-source-badge--${d.source.toLowerCase()}`;
    badge.textContent = d.source;
    titleBlock.appendChild(badge);
  }

  if (d.salary_display) {
    const sal = document.createElement('div');
    sal.className = 'jd-salary';
    sal.textContent = d.salary_display;
    titleBlock.appendChild(sal);
  }

  content.appendChild(titleBlock);

  // ── Overview ─────────────────────────────────────────────────────────────
  if (d.overview) {
    content.appendChild(heading('cd-section-heading', 'Overview'));
    content.appendChild(para('cd-body', d.overview));
  }

  // ── Expandable detail ────────────────────────────────────────────────────
  if (d.typical_duties || d.skills_required || d.entry_routes) {
    const toggle = document.createElement('button');
    toggle.className = 'cd-expand-btn';
    toggle.textContent = 'Role details';
    toggle.dataset.open = 'false';

    const panel = document.createElement('div');
    panel.className = 'cd-expand-panel';
    panel.hidden = true;

    if (d.typical_duties) {
      panel.appendChild(heading('cd-section-heading', 'Typical duties'));
      panel.appendChild(para('cd-body', d.typical_duties));
    }
    if (d.skills_required) {
      panel.appendChild(heading('cd-section-heading', 'Skills required'));
      panel.appendChild(para('cd-body', d.skills_required));
    }
    if (d.entry_routes) {
      panel.appendChild(heading('cd-section-heading', 'Entry routes'));
      panel.appendChild(para('cd-body', d.entry_routes));
    }

    toggle.addEventListener('click', () => {
      const open = toggle.dataset.open === 'true';
      toggle.dataset.open = String(!open);
      toggle.textContent = open ? 'Role details' : 'Less';
      panel.hidden = open;
    });

    content.appendChild(toggle);
    content.appendChild(panel);
  }

  // ── External link ────────────────────────────────────────────────────────
  if (d.source_url) {
    const link = document.createElement('a');
    link.className = 'cd-course-url';
    link.href = d.source_url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    const sourceName = d.source === 'NCS' ? 'National Careers Service'
                     : d.source === 'PROSPECTS' ? 'Prospects'
                     : d.source || 'source';
    link.textContent = `View on ${sourceName} →`;
    content.appendChild(link);
  }

  // ── Working in Greater Manchester ────────────────────────────────────────
  if (d.employer_text) {
    content.appendChild(divider());
    content.appendChild(heading('cd-section-heading', 'Working in Greater Manchester'));
    content.appendChild(para('cd-body', d.employer_text));
  }

  // ── On-demand sections ───────────────────────────────────────────────────
  const progSection = document.createElement('div');
  progSection.id = 'jd-progression-section';
  content.appendChild(progSection);

  const coursesSection = document.createElement('div');
  coursesSection.id = 'jd-courses-section';
  content.appendChild(coursesSection);
}

function wireRevealButton(section, label, onReveal) {
  if (!section) return;

  const hr = document.createElement('hr');
  hr.className = 'cd-divider';
  section.appendChild(hr);

  const btn = document.createElement('button');
  btn.className = 'jd-reveal-btn';
  btn.textContent = label;
  btn.dataset.open = 'false';
  section.appendChild(btn);

  const panel = document.createElement('div');
  panel.hidden = true;
  section.appendChild(panel);
  section.appendChild(btn);  // button always lives after the panel

  let loaded = false;

  btn.addEventListener('click', () => {
    const open = btn.dataset.open === 'true';
    if (!open) {
      btn.dataset.open = 'true';
      btn.textContent = 'Less';
      panel.hidden = false;
      if (!loaded) {
        loaded = true;
        onReveal(panel);
      }
    } else {
      btn.dataset.open = 'false';
      btn.textContent = label;
      panel.hidden = true;
    }
  });
}

function loadExplain(panel, jobId) {
  panel.appendChild(heading('cd-section-heading', 'Career path'));

  const loading = document.createElement('p');
  loading.className = 'cd-loading';
  loading.textContent = 'Loading career path…';
  panel.appendChild(loading);

  fetch(`/jobs/${jobId}/explain`)
    .then(r => r.json())
    .then(d => {
      loading.remove();
      const text = d.text || '';
      if (!text) {
        const p = document.createElement('p');
        p.className = 'cd-body cd-muted';
        p.textContent = 'No career pathway information available.';
        panel.appendChild(p);
        return;
      }
      logEvent('progression_open', 'job', jobId, null);
      const narr = document.createElement('p');
      narr.className = 'cd-body';
      narr.textContent = text;
      panel.appendChild(narr);
    })
    .catch(() => {
      loading.remove();
    });
}

function loadCourses(panel, jobId) {
  panel.appendChild(heading('cd-section-heading', 'Courses that lead here'));

  const loading = document.createElement('p');
  loading.className = 'cd-loading';
  loading.textContent = 'Finding relevant courses…';
  panel.appendChild(loading);

  fetch(`/jobs/${jobId}/courses?limit=5`)
    .then(r => r.json())
    .then(d => {
      loading.remove();
      const results = d.results || [];
      if (!results.length) {
        const p = document.createElement('p');
        p.className = 'cd-body cd-muted';
        p.textContent = 'No matching courses found.';
        panel.appendChild(p);
        return;
      }

      // Preview panel — shown when a course is tapped
      const preview = document.createElement('div');
      preview.className = 'jd-course-preview';
      preview.hidden = true;

      const previewTitle = document.createElement('div');
      previewTitle.className = 'jd-course-preview-title';

      const previewBody = document.createElement('p');
      previewBody.className = 'cd-body';

      const previewSave = document.createElement('button');
      previewSave.className = 'jd-course-preview-save';

      preview.appendChild(previewTitle);
      preview.appendChild(previewBody);
      preview.appendChild(previewSave);

      let _activeCourseId = null;

      const updatePreviewSave = (course) => {
        const saved = isSaved(course.id, 'course');
        previewSave.textContent = saved ? 'Saved' : 'Save course';
        previewSave.className = `jd-course-preview-save${saved ? ' jd-course-preview-save--saved' : ''}`;
      };

      previewSave.addEventListener('click', () => {
        if (!_activeCourseId) return;
        const course = results.find(c => c.id === _activeCourseId);
        if (!course) return;
        if (isSaved(course.id, 'course')) {
          unsaveItem(course.id, 'course');
        } else {
          saveItem({ id: course.id, type: 'course', title: course.title });
        }
        updatePreviewSave(course);
        const badge = document.getElementById('saved-count');
        if (badge) badge.textContent = state.saved.items.length;
      });

      const list = document.createElement('div');
      list.className = 'jd-course-list';
      const allItems = [];

      results.forEach(course => {
        const item = document.createElement('button');
        item.className = 'jd-course-item';
        allItems.push(item);

        item.addEventListener('click', () => {
          const alreadyActive = _activeCourseId === course.id;
          allItems.forEach(i => i.classList.remove('jd-course-item--active'));
          if (alreadyActive) {
            _activeCourseId = null;
            preview.hidden = true;
          } else {
            _activeCourseId = course.id;
            item.classList.add('jd-course-item--active');
            previewTitle.textContent = course.title;
            previewBody.textContent  = course.overview || '';
            updatePreviewSave(course);
            preview.hidden = false;
          }
        });

        const title = document.createElement('div');
        title.className = 'jd-course-title';
        title.textContent = course.title;

        const meta = document.createElement('div');
        meta.className = 'jd-course-meta';
        meta.textContent = [course.qualification_type, course.provider].filter(Boolean).join(' · ');

        item.appendChild(title);
        item.appendChild(meta);
        list.appendChild(item);
      });

      panel.appendChild(list);
      panel.appendChild(preview);
    })
    .catch(() => {
      loading.remove();
    });
}

