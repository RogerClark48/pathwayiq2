/* Role profile — notebook */

import { go }                                              from '../router.js';
import { state, isSaved, toggleSave, unsaveItem, saveItem } from '../state.js';
import { logEvent }                                        from '../analytics.js';
import { subject, subjectIconSvg }                         from '../subjects.js';
import { renderProse, renderField, splitProse }            from '../dom.js';
import { getWelcomeData }                                  from '../api.js';

// Generic role icon for the header watermark (intentionally not a subject icon)
const ROLE_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
  stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <rect x="2" y="7" width="20" height="14" rx="2"/>
  <path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/>
  <line x1="12" y1="12" x2="12" y2="16"/>
  <line x1="10" y1="14" x2="14" y2="14"/>
</svg>`;

const BOOKMARK_EMPTY = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

const BOOKMARK_SAVED = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/></svg>`;

function sourceName(src) {
  return src === 'NCS' ? 'National Careers Service'
       : src === 'PROSPECTS' ? 'Prospects'
       : src || '';
}

// ── Main view ─────────────────────────────────────────────────────────────────

export function JobDetailView(slices = {}) {
  const { jobId, jobTitle, backRoute, backSlices, ssa } = slices;
  const accent = slices.ssa ? subject(slices.ssa).colour : 'var(--sub-neutral)';
  return buildJobEl({
    jobId, jobTitle, ssa, accent,
    onBack: () => go(backRoute, backSlices || {}),
  });
}

// ── Element builder — used by both JobDetailView (routing) and desktop split ──

export function buildJobEl({ jobId, jobTitle, ssa, accent, onBack, backLabel, onCoursesClick }) {
  const el = document.createElement('div');
  el.className = 'view view-job-detail';

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

  el.querySelector('.cd-load-back').addEventListener('click', onBack);

  logEvent('career_detail_open', 'job', jobId, jobTitle);

  fetch(`/jobs/${jobId}`)
    .then(r => {
      if (!r.ok) throw new Error(`/jobs/${jobId} ${r.status}`);
      return r.json();
    })
    .then(d => {
      el.innerHTML = '';
      renderDetail(el, d, jobId, jobTitle, onBack, accent, ssa, backLabel, onCoursesClick);
    })
    .catch(err => {
      console.error('[job-detail]', err);
      el.innerHTML = '<p class="cd-error">Could not load role details.</p>';
    });

  return el;
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderDetail(el, d, jobId, jobTitle, onBack, accent, ssa, backLabel, onCoursesClick) {
  // Propagate subject colour to the whole view so notebook panes (Climb) inherit it
  el.style.setProperty('--sub', accent);

  // ── Header ────────────────────────────────────────────────────────────────
  const head = document.createElement('div');
  head.className = 'jd-head';
  head.style.setProperty('--sub', accent);

  const backBtnHtml = backLabel
    ? `<button class="jd-back-pill">${backLabel}<svg viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
         stroke-linejoin="round" width="12" height="12" aria-hidden="true">
         <polyline points="9 18 15 12 9 6"/></svg></button>`
    : `<button class="ic jd-back" aria-label="Back">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
              stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
           <polyline points="15 18 9 12 15 6"/>
         </svg>
       </button>`;

  head.innerHTML = `
    <span class="wm" aria-hidden="true">${subjectIconSvg(ssa)}</span>
    <div class="cd-bar">
      ${backBtnHtml}
      <button class="ic jd-finn" aria-label="Back to chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>
      <button class="ic jd-save" aria-label="Save role"></button>
    </div>
    <div class="jd-kicker">
      ${d.source ? `<span class="jd-source-kicker">${sourceName(d.source)}</span> · ` : ''}Career role
    </div>
    <h4>${d.title}</h4>
    ${d.salary_display ? `<div class="jd-headfacts"><span class="jd-pay">${d.salary_display}</span></div>` : ''}`;

  (head.querySelector('.jd-back') || head.querySelector('.jd-back-pill'))
    .addEventListener('click', onBack);
  head.querySelector('.jd-finn').addEventListener('click', () => go('chat-first'));

  const saveBtn = head.querySelector('.jd-save');
  const updateSaveBtn = () => {
    const saved = isSaved(jobId, 'job');
    saveBtn.innerHTML = saved ? BOOKMARK_SAVED : BOOKMARK_EMPTY;
    saveBtn.classList.toggle('on', saved);
    saveBtn.setAttribute('aria-pressed', String(saved));
  };
  updateSaveBtn();
  saveBtn.addEventListener('click', () => {
    toggleSave({ id: jobId, type: 'job', title: jobTitle, ssa, subtitle: sourceName(d.source) });
    updateSaveBtn();
  });

  el.appendChild(head);

  // Fired as soon as the profile loads, not on tab click — both are slow (~15s) Sonnet
  // calls on a cold cache, so they need a head start before the user reaches the tab.
  const insidePromise = d.overview
    ? fetch(`/jobs/${jobId}/inside`).then(r => r.json()).catch(() => null)
    : Promise.resolve(null);
  const climbPromise = fetch(`/jobs/${jobId}/ladder`).then(r => r.json()).catch(() => null);

  // ── Notebook ──────────────────────────────────────────────────────────────
  const tabDefs = [
    { key: 'Overview', lazy: false, render: () => overviewPane(d) },
    (d.typical_duties || d.skills_required)
      ? { key: 'Duties',  lazy: false, render: () => dutiesPane(d) }
      : null,
    d.entry_routes
      ? { key: 'Get in', lazy: false, render: () => textPane(d.entry_routes) }
      : null,
    d.overview
      ? { key: 'Inside', lazy: true, render: (pane) => loadInside(pane, insidePromise, jobId) }
      : null,
    { key: 'Climb', lazy: true, render: (pane) => loadClimb(pane, climbPromise, jobId) },
  ].filter(Boolean);

  const notebook = document.createElement('div');
  notebook.className = 'notebook';

  const tabStrip = document.createElement('div');
  tabStrip.className = 'nb-tabs';

  const sheet = document.createElement('div');
  sheet.className = 'nb-sheet';

  tabDefs.forEach((def, i) => {
    const tab  = document.createElement('button');
    tab.className = `nb-tab${i === 0 ? ' on' : ''}`;
    tab.textContent = def.key;

    const pane = document.createElement('div');
    pane.className = `nb-pane${i === 0 ? ' on' : ''}`;

    if (!def.lazy) {
      pane.appendChild(def.render());
    }

    let lazyDone = false;
    tab.addEventListener('click', () => {
      tabStrip.querySelectorAll('.nb-tab').forEach(t => t.classList.remove('on'));
      sheet.querySelectorAll('.nb-pane').forEach(p => p.classList.remove('on'));
      tab.classList.add('on');
      pane.classList.add('on');
      sheet.scrollTop = 0;
      if (def.lazy && !lazyDone) {
        lazyDone = true;
        def.render(pane); // render receives the pane to populate into
      }
    });

    tabStrip.appendChild(tab);
    sheet.appendChild(pane);
  });

  notebook.appendChild(tabStrip);
  notebook.appendChild(sheet);
  el.appendChild(notebook);

  // ── Pinned courses bar ────────────────────────────────────────────────────
  const cta = document.createElement('div');
  cta.className = 'jd-cta';

  const ctaHead = document.createElement('button');
  ctaHead.className = 'jd-cta-head';
  cta.appendChild(ctaHead);

  if (onCoursesClick) {
    // Desktop split: tap widens the overlay into role | courses pane
    ctaHead.innerHTML = `Courses that lead here <svg viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
      stroke-linejoin="round" width="16" height="16" aria-hidden="true">
      <polyline points="9 18 15 12 9 6"/></svg>`;
    ctaHead.addEventListener('click', onCoursesClick);
  } else {
    // Mobile / standalone: inline expand below the button
    ctaHead.innerHTML = `Courses that lead here <span class="chev">▾</span>`;

    const ctaBody = document.createElement('div');
    ctaBody.className = 'jd-cta-body';
    ctaBody.hidden = true;
    cta.appendChild(ctaBody);

    let ctaOpen = false;
    let coursesLoaded = false;
    ctaHead.addEventListener('click', () => {
      ctaOpen = !ctaOpen;
      ctaBody.hidden = !ctaOpen;
      ctaHead.querySelector('.chev').textContent = ctaOpen ? '▴' : '▾';
      if (ctaOpen && !coursesLoaded) {
        coursesLoaded = true;
        loadCoursesInto(ctaBody, jobId);
      }
    });
  }

  el.appendChild(cta);
}

// ── Pane builders ─────────────────────────────────────────────────────────────

function overviewPane(d) {
  const frag = document.createDocumentFragment();

  // Fact grid — real fields only, no invented data
  const facts = [
    d.salary_display ? { label: 'Pay',    value: d.salary_display }       : null,
    d.source         ? { label: 'Source', value: sourceName(d.source) }   : null,
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

  if (d.overview) frag.appendChild(renderProse(d.overview));

  if (d.source_url) {
    const a = document.createElement('a');
    a.className = 'nb-url';
    a.href      = d.source_url;
    a.target    = '_blank';
    a.rel       = 'noopener noreferrer';
    a.textContent = `View on ${sourceName(d.source)} →`;
    frag.appendChild(a);
  }

  if (d.icould_videos?.length) {
    frag.appendChild(buildVideoRow(d.icould_videos));
  }

  if (d.employer_text) {
    const gm = document.createElement('div');
    gm.className = 'jd-gm';
    const gmHead = document.createElement('h6');
    gmHead.textContent = `Working in ${getWelcomeData()?.institution?.region || 'Greater Manchester'}`;
    const gmBody = document.createElement('p');
    gmBody.textContent = d.employer_text;
    gm.appendChild(gmHead);
    gm.appendChild(gmBody);
    frag.appendChild(gm);
  }

  return frag;
}

function buildVideoRow(videos) {
  const section = document.createElement('div');
  section.className = 'jd-videos';

  const label = document.createElement('p');
  label.className = 'jd-videos-label';
  label.textContent = 'Hear from people in this kind of work';
  section.appendChild(label);

  const row = document.createElement('div');
  row.className = 'jd-videos-row';

  videos.forEach(v => {
    const card = document.createElement('button');
    card.className = 'jd-video-card';
    card.setAttribute('aria-label', `Play: ${v.title}`);

    const thumb = document.createElement('div');
    thumb.className = 'jd-video-thumb';
    if (v.thumbnail_url) {
      thumb.style.backgroundImage = `url('${v.thumbnail_url}')`;
    }

    const play = document.createElement('div');
    play.className = 'jd-video-play';
    play.innerHTML = `<svg viewBox="0 0 24 24" fill="white" aria-hidden="true" width="28" height="28">
      <circle cx="12" cy="12" r="12" fill="rgba(0,0,0,0.45)"/>
      <polygon points="10,8 18,12 10,16" fill="white"/>
    </svg>`;
    thumb.appendChild(play);

    const info = document.createElement('div');
    info.className = 'jd-video-info';

    const title = document.createElement('span');
    title.className = 'jd-video-title';
    title.textContent = v.title;

    const attr = document.createElement('span');
    attr.className = 'jd-video-attr';
    attr.textContent = 'icould career stories';

    info.appendChild(title);
    info.appendChild(attr);

    card.appendChild(thumb);
    card.appendChild(info);
    card.addEventListener('click', () => openVideoModal(v.video_id, v.title));
    row.appendChild(card);
  });

  section.appendChild(row);
  return section;
}

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
  closeBtn.focus();
}

function dutiesPane(d) {
  const frag = document.createDocumentFragment();

  if (d.typical_duties) frag.appendChild(renderField(d.typical_duties));

  if (d.skills_required) {
    const h = document.createElement('h6');
    h.className = 'nb-label';
    h.textContent = 'Skills';
    frag.appendChild(h);

    const chips = document.createElement('div');
    chips.className = 'jd-skills';
    d.skills_required.split(/[,\n]/).map(s => s.trim()).filter(Boolean).forEach(skill => {
      const chip = document.createElement('span');
      chip.textContent = skill;
      chips.appendChild(chip);
    });
    frag.appendChild(chips);
  }

  return frag;
}

function textPane(text) {
  return renderField(text);
}

// Lazy — receives the pane element and the fetch promise kicked off on profile load.
function loadInside(pane, insidePromise, jobId) {
  const loading = document.createElement('p');
  loading.className = 'nb-body nb-muted';
  loading.textContent = 'Loading…';
  pane.appendChild(loading);

  insidePromise.then(d => {
    loading.remove();
    const sections = Array.isArray(d?.sections) ? d.sections : [];
    if (!sections.length) {
      const p = document.createElement('p');
      p.className = 'nb-body nb-muted';
      p.textContent = 'No inside view for this role yet.';
      pane.appendChild(p);
      return;
    }
    sections.forEach(({ heading, text }) => {
      if (heading) {
        const h = document.createElement('h6');
        h.className = 'nb-label';
        h.textContent = heading;
        pane.appendChild(h);
      }
      if (text) pane.appendChild(renderProse(text));
    });
    logEvent('inside_open', 'job', jobId, null);
  });
}

// Lazy — receives the pane element and the fetch promise kicked off on profile load.
function loadClimb(pane, climbPromise, jobId) {
  const loading = document.createElement('p');
  loading.className = 'nb-body nb-muted';
  loading.textContent = 'Loading…';
  pane.appendChild(loading);

  climbPromise
    .then(data => {
      loading.remove();
      const rungs = Array.isArray(data?.ladder) ? data.ladder : [];
      const paras = Array.isArray(data?.commentary) ? data.commentary : [];
      const hasLadder = rungs.length >= 2 && rungs.some(r => r.marker === 'current');

      if (hasLadder) {
        const lbl = document.createElement('p');
        lbl.className = 'nb-label';
        lbl.textContent = 'Where it leads';
        pane.appendChild(lbl);
        pane.appendChild(renderLadder(rungs));
      }

      if (paras.length) {
        pane.appendChild(renderCommentary(paras));
      }

      if (!hasLadder && !paras.length) {
        renderProseFallback(pane, jobId);
        return;
      }

      logEvent('progression_open', 'job', jobId, null);
    });
}

function renderLadder(rungs) {
  const box = document.createElement('div');
  box.className = 'jd-climb';
  rungs.forEach(r => {
    const cls = 'jd-step'
      + (r.marker === 'goal'    ? ' goal'    : '')
      + (r.marker === 'current' ? ' current' : '');
    const step = document.createElement('div');
    step.className = cls;

    const dot  = document.createElement('span');
    dot.className = 'd';

    const title = document.createElement('b');
    title.textContent = r.role;   // textContent — never innerHTML for model output

    const caption = document.createElement('em');
    caption.textContent = r.stage;

    step.appendChild(dot);
    step.appendChild(title);
    step.appendChild(caption);
    box.appendChild(step);
  });
  return box;
}

function renderCommentary(paras) {
  return renderProse(paras, 'jd-climb-prose');
}

function renderProseFallback(pane, jobId) {
  fetch(`/jobs/${jobId}/explain`)
    .then(r => r.json())
    .then(d => {
      const raw = (d.text || '').trim();
      if (!raw) {
        const p = document.createElement('p');
        p.className = 'nb-body nb-muted';
        p.textContent = 'No career pathway information yet.';
        pane.appendChild(p);
        return;
      }
      logEvent('progression_open', 'job', jobId, null);
      pane.appendChild(renderProse(raw, 'jd-climb-prose'));
    })
    .catch(() => {});
}

// Lazy — called on first expand of the CTA bar (mobile) or first "Courses" tap (desktop).
// Terminal: no navigation. Each row opens an inline preview + save only.
export function loadCoursesInto(panel, jobId) {
  const loading = document.createElement('p');
  loading.className = 'nb-body nb-muted';
  loading.textContent = 'Finding courses…';
  panel.appendChild(loading);

  fetch(`/jobs/${jobId}/courses?limit=5`)
    .then(r => r.json())
    .then(d => {
      loading.remove();
      const results = d.results || [];
      if (!results.length) {
        const p = document.createElement('p');
        p.className = 'nb-body nb-muted';
        p.textContent = `No matching ${getWelcomeData()?.institution?.abbrev || 'GMIoT'} courses found.`;
        panel.appendChild(p);
        return;
      }

      // Inline preview panel — no navigation out of this leaf
      const preview     = document.createElement('div');
      preview.className = 'jd-course-preview';
      preview.hidden    = true;

      const previewTitle = document.createElement('div');
      previewTitle.className = 'jd-course-preview-title';

      const previewBody = document.createElement('p');
      previewBody.className = 'nb-body';

      const previewSave = document.createElement('button');
      previewSave.className = 'jd-course-preview-save';

      preview.appendChild(previewTitle);
      preview.appendChild(previewBody);
      preview.appendChild(previewSave);

      let _activeCourseId = null;

      const updatePreviewSave = course => {
        const saved = isSaved(course.id, 'course');
        previewSave.textContent = saved ? 'Saved ✓' : 'Save course';
        previewSave.className = `jd-course-preview-save${saved ? ' jd-course-preview-save--saved' : ''}`;
      };

      previewSave.addEventListener('click', () => {
        const course = results.find(c => c.id === _activeCourseId);
        if (!course) return;
        toggleSave({
          id:       course.id,
          type:     'course',
          title:    course.title,
          subtitle: [course.qualification_type, course.provider].filter(Boolean).join(' · '),
        });
        updatePreviewSave(course);
      });

      const list     = document.createElement('div');
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
            panel.scrollTop = panel.scrollHeight;
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
    .catch(() => loading.remove());
}
