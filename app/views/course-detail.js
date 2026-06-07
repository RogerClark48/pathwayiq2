/* Course detail — notebook */

import { go }                                        from '../router.js';
import { state, isSaved, toggleSave, refreshSavedBadges } from '../state.js';
import { logEvent }                                  from '../analytics.js';
import { subject, subjectIconSvg }                   from '../subjects.js';

const MODE_LABELS = { FT: 'Full-time', PT: 'Part-time', 'FT/PT': 'Full or Part-time' };
function modeLabel(m) { return m ? (MODE_LABELS[m] || m) : null; }

const BOOKMARK_EMPTY = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

const BOOKMARK_SAVED = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

// ── Main view ─────────────────────────────────────────────────────────────────

export function CourseDetailView(slices = {}) {
  const courseId    = slices.courseId;
  const backRoute   = slices.backRoute || 'course-list';
  const courseTitle = slices.courseTitle || '';

  const el = document.createElement('div');
  el.className = 'view view-course-detail';

  // Minimal back available during load
  el.innerHTML = `
    <div class="cd-loading-screen">
      <button class="cd-load-back" aria-label="Back">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="15 18 9 12 15 6"/>
        </svg>
      </button>
      <span>Loading…</span>
    </div>`;

  el.querySelector('.cd-load-back').addEventListener('click', () =>
    go(backRoute, backRoute === 'course-list' ? { courseList: state.courseList } : {}));

  logEvent('course_detail_open', 'course', courseId, courseTitle);

  fetch(`/courses/${courseId}/detail`)
    .then(r => {
      if (!r.ok) throw new Error(`/courses/${courseId}/detail ${r.status}`);
      return r.json();
    })
    .then(d => {
      el.innerHTML = '';
      renderDetail(el, d, courseId, courseTitle, backRoute);
    })
    .catch(err => {
      console.error('[course-detail]', err);
      el.innerHTML = '<p class="cd-error">Could not load course details.</p>';
    });

  return el;
}

// ── Render (called after fetch) ───────────────────────────────────────────────

function renderDetail(el, d, courseId, courseTitle, backRoute) {
  const s = subject(d.ssa_code);

  // ── Coloured header ───────────────────────────────────────────────────────
  const head = document.createElement('div');
  head.className = 'cd-head';
  head.style.setProperty('--sub', s.colour);

  const campus    = d.campus && d.campus !== d.provider ? d.campus : null;
  const metaLine2 = [d.provider, modeLabel(d.mode), d.duration, campus]
    .filter(Boolean).join(' · ');

  head.innerHTML = `
    <span class="wm" aria-hidden="true">${subjectIconSvg(d.ssa_code)}</span>
    <div class="cd-bar">
      <button class="ic cd-back" aria-label="Back">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="15 18 9 12 15 6"/>
        </svg>
      </button>
      <button class="ic cd-finn" aria-label="Back to Finn">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>
      <button class="ic cd-save" aria-label="Save course"></button>
    </div>
    <h4>${d.course_title}</h4>
    <div class="cd-meta">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
        <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
      </svg>
      ${s.label}${d.qual_type ? ` · ${d.qual_type}` : ''}
    </div>
    ${metaLine2 ? `
    <div class="cd-meta">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z"/>
        <circle cx="12" cy="10" r="3"/>
      </svg>
      ${metaLine2}
    </div>` : ''}`;

  head.querySelector('.cd-back').addEventListener('click', () =>
    go(backRoute, backRoute === 'course-list' ? { courseList: state.courseList } : {}));

  head.querySelector('.cd-finn').addEventListener('click', () => go('chat-first'));

  const saveBtn = head.querySelector('.cd-save');
  const updateSaveBtn = () => {
    const saved = isSaved(courseId, 'course');
    saveBtn.innerHTML = saved ? BOOKMARK_SAVED : BOOKMARK_EMPTY;
    saveBtn.classList.toggle('on', saved);
    saveBtn.setAttribute('aria-pressed', String(saved));
  };
  updateSaveBtn();
  saveBtn.addEventListener('click', () => {
    const subtitle = [d.qual_type, d.provider].filter(Boolean).join(' · ');
    toggleSave({ id: courseId, type: 'course', title: courseTitle, ssa: d.ssa_code, subtitle });
    updateSaveBtn();
  });

  el.appendChild(head);

  // ── Notebook ──────────────────────────────────────────────────────────────
  const tabDefs = [
    d.overview           && { key: 'Overview', render: () => overviewPane(d) },
    d.content            && { key: 'Learn',    render: () => learnPane(d.content) },
    d.entry_requirements && { key: 'Entry',    render: () => textPane(d.entry_requirements) },
    d.progression        && { key: 'Next',     render: () => textPane(d.progression) },
  ].filter(Boolean);

  const notebook  = document.createElement('div');
  notebook.className = 'notebook';

  const tabStrip = document.createElement('div');
  tabStrip.className = 'nb-tabs';

  const sheet = document.createElement('div');
  sheet.className = 'nb-sheet';

  tabDefs.forEach((def, i) => {
    const tab = document.createElement('button');
    tab.className = `nb-tab${i === 0 ? ' on' : ''}`;
    tab.textContent = def.key;

    const pane = document.createElement('div');
    pane.className = `nb-pane${i === 0 ? ' on' : ''}`;
    pane.appendChild(def.render());
    sheet.appendChild(pane);

    tab.addEventListener('click', () => {
      tabStrip.querySelectorAll('.nb-tab').forEach(t => t.classList.remove('on'));
      sheet.querySelectorAll('.nb-pane').forEach(p => p.classList.remove('on'));
      tab.classList.add('on');
      pane.classList.add('on');
      sheet.scrollTop = 0;
    });

    tabStrip.appendChild(tab);
  });

  notebook.appendChild(tabStrip);
  notebook.appendChild(sheet);
  el.appendChild(notebook);

  // ── Pinned career bar ─────────────────────────────────────────────────────
  if (d.pathways) {
    const cta = document.createElement('div');
    cta.className = 'cd-cta';

    const cardJobs = d.pathways.card_jobs || [];
    const shown    = cardJobs.slice(0, 2);
    const more     = cardJobs.length - shown.length;

    shown.forEach(j => logEvent('career_impression', 'job', j.job_id, j.title));

    const pillsHtml = shown.map(j => `<span>${j.title}</span>`).join('')
      + (more > 0 ? `<span class="more">+${more} roles</span>` : '');

    cta.innerHTML = `
      <span class="lbl">Where this could lead</span>
      ${pillsHtml ? `<div class="pills">${pillsHtml}</div>` : ''}
      <button class="btn btn-blue cd-career">Full career view →</button>`;

    cta.querySelector('.cd-career').addEventListener('click', () =>
      go('career-view', {
        courseId,
        courseTitle: d.course_title,
        ssa:         d.ssa_code,
        pathways:    d.pathways,
        backRoute:   'course-detail',
        backSlices:  { courseId, courseTitle: d.course_title, backRoute },
      })
    );

    el.appendChild(cta);
  }
}

// ── Pane builders ─────────────────────────────────────────────────────────────

function overviewPane(d) {
  const frag = document.createDocumentFragment();

  // Fact grid
  const facts = [
    d.qual_type        ? { label: 'Award',  value: d.qual_type }          : null,
    modeLabel(d.mode)  ? { label: 'Study',  value: modeLabel(d.mode) }    : null,
    d.duration         ? { label: 'Length', value: d.duration }           : null,
    d.campus           ? { label: 'Campus', value: d.campus }             : null,
  ].filter(Boolean);

  if (facts.length) {
    const grid = document.createElement('div');
    grid.className = 'nb-facts';
    facts.forEach(({ label, value }) => {
      const cell = document.createElement('div');
      cell.innerHTML = `<span class="nb-fact-label">${label}</span><span class="nb-fact-value">${value}</span>`;
      grid.appendChild(cell);
    });
    frag.appendChild(grid);
  }

  // Apprenticeship notice
  if (d.qual_type?.includes('Apprenticeship')) {
    const notice = document.createElement('p');
    notice.className = 'nb-notice';
    notice.textContent =
      'Apprenticeships are studied while you work and are offered through an employer. ' +
      'Check the provider’s webpage for details on how to apply.';
    frag.appendChild(notice);
  }

  // Overview prose
  if (d.overview) {
    const p = document.createElement('p');
    p.className = 'nb-body';
    p.textContent = d.overview;
    frag.appendChild(p);
  }

  // Course URL — the primary apply action, must survive redesign
  if (d.course_url) {
    const a = document.createElement('a');
    a.className = 'nb-url';
    a.href      = d.course_url;
    a.target    = '_blank';
    a.rel       = 'noopener noreferrer';
    a.textContent = `View course on ${d.provider || 'provider'} website →`;
    frag.appendChild(a);
  }

  return frag;
}

function learnPane(content) {
  const frag  = document.createDocumentFragment();
  const lines = (content || '').split('\n').map(l => l.trim()).filter(Boolean);
  const isList = lines.some(l => /^[•\-\d]/.test(l));

  if (isList) {
    const ol = document.createElement('ol');
    ol.className = 'nb-mods';
    lines.forEach(l => {
      const text = l.replace(/^[•\-\d]+[.)]*\s*/, '').trim();
      if (!text) return;
      const li = document.createElement('li');
      li.textContent = text;
      ol.appendChild(li);
    });
    frag.appendChild(ol);
  } else {
    const p = document.createElement('p');
    p.className = 'nb-body';
    p.textContent = content;
    frag.appendChild(p);
  }
  return frag;
}

function textPane(text) {
  const frag = document.createDocumentFragment();
  const p    = document.createElement('p');
  p.className = 'nb-body';
  p.textContent = text;
  frag.appendChild(p);
  return frag;
}
