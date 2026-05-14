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

function jobPill(job, courseId, courseTitle, backRoute) {
  const el = document.createElement('button');
  el.className = 'cd-job-pill';
  el.textContent = job.title;
  el.addEventListener('click', () => go('job-detail', {
    jobId:      job.job_id,
    jobTitle:   job.title,
    backRoute:  'course-detail',
    backSlices: { courseId, courseTitle, backRoute },
  }));
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

const GROUP_COLORS = ['#00897B', '#F57C00', '#1A237E', '#6A1B9A', '#C62828'];

function buildGroupDiagram(courseTitle, jobs, nodeGroups, onSelect) {
  const NS    = 'http://www.w3.org/2000/svg';
  const W     = 340;
  const MARGIN = 50;
  const JR    = 22;
  const JGP   = 52;
  const GRP_Y = 80;
  const JOB_Y0 = 140;

  const jobById = Object.fromEntries(jobs.map(j => [j.job_id, j]));

  // Build group objects; fall back to one group if data missing
  const groups = (nodeGroups && nodeGroups.length)
    ? nodeGroups.map((g, i) => ({
        label: g.label,
        fill:  GROUP_COLORS[i % GROUP_COLORS.length],
        jobs:  (g.job_ids || []).map(id => jobById[id]).filter(Boolean),
      }))
    : [{ label: 'Career roles', fill: GROUP_COLORS[0], jobs: Object.values(jobById) }];

  // Assign x centre for each group column
  const N = groups.length;
  const step = N === 1 ? 0 : (W - 2 * MARGIN) / (N - 1);
  groups.forEach((g, i) => { g.cx = N === 1 ? W / 2 : MARGIN + i * step; });

  const maxRows = Math.max(...groups.map(g => g.jobs.length), 1);
  const H = JOB_Y0 + (maxRows - 1) * JGP + JR + 24;

  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.cssText = 'width:100%;display:block;';

  function svgEl(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return e;
  }

  function svgText(x, y, str, size, fill, weight) {
    const t = svgEl('text', {
      x, y,
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      'font-size': size,
      'font-family': '-apple-system,"Segoe UI",Roboto,sans-serif',
      fill,
      'font-weight': weight || '400',
    });
    t.textContent = str;
    return t;
  }

  function addWrapped(parent, x, y, str, maxCh, lineH, size, fill, weight) {
    const words = str.split(' ');
    const lines = [];
    let cur = '';
    for (const w of words) {
      const test = cur ? `${cur} ${w}` : w;
      if (test.length > maxCh && cur) {
        lines.push(cur); cur = w;
        if (lines.length >= 3) break;
      } else cur = test;
    }
    if (cur && lines.length < 3) lines.push(cur);
    const totalH = (lines.length - 1) * lineH;
    lines.forEach((l, i) =>
      parent.appendChild(svgText(x, y - totalH / 2 + i * lineH, l, size, fill, weight))
    );
  }

  // ── Course rect ───────────────────────────────────────────────────────────
  svg.appendChild(svgEl('rect', { x: 10, y: 6, width: W - 20, height: 32, rx: 8, fill: '#1A237E' }));
  addWrapped(svg, W / 2, 22, courseTitle, 36, 11, 9, '#FFFFFF', '600');

  // Available horizontal space per column
  const colW = N === 1 ? W - 2 * MARGIN : step;

  // ── Lines: course → group headers ─────────────────────────────────────────
  groups.forEach(g => {
    if (!g.jobs.length) return;
    svg.appendChild(svgEl('line', {
      x1: W / 2, y1: 38, x2: g.cx, y2: GRP_Y - 17,
      stroke: '#CCCCCC', 'stroke-width': 1.5,
    }));
  });

  // ── Lines: group header → each job ────────────────────────────────────────
  groups.forEach(g => {
    g.jobs.forEach((_, ji) => {
      const jy = JOB_Y0 + ji * JGP;
      svg.appendChild(svgEl('line', {
        x1: g.cx, y1: GRP_Y + 17, x2: g.cx, y2: jy - JR,
        stroke: '#CCCCCC', 'stroke-width': 1,
      }));
    });
  });

  // ── Group header pills ─────────────────────────────────────────────────────
  groups.forEach(g => {
    if (!g.jobs.length) return;
    const maxPw    = 2 * Math.min(g.cx - 4, W - g.cx - 4);
    const pw       = Math.min(g.label.length * 5.2 + 16, colW - 4, maxPw);
    const pillCh   = Math.max(5, Math.floor((pw - 8) / 4.2));
    svg.appendChild(svgEl('rect', {
      x: g.cx - pw / 2, y: GRP_Y - 17, width: pw, height: 34, rx: 17,
      fill: g.fill,
    }));
    addWrapped(svg, g.cx, GRP_Y, g.label, pillCh, 11, 7, '#FFFFFF', '700');
  });

  // ── Job nodes ─────────────────────────────────────────────────────────────
  const allCircles = [];

  groups.forEach(g => {
    g.jobs.forEach((job, ji) => {
      const jy  = JOB_Y0 + ji * JGP;
      const grp = document.createElementNS(NS, 'g');
      grp.style.cursor = 'pointer';

      const c = svgEl('circle', { cx: g.cx, cy: jy, r: JR, fill: g.fill });
      grp.appendChild(c);
      allCircles.push({ circle: c, defaultFill: g.fill });

      addWrapped(grp, g.cx, jy, job.title, 11, 9, 7, '#FFFFFF', '500');

      grp.addEventListener('click', () => {
        allCircles.forEach(({ circle: jc, defaultFill }) =>
          jc.setAttribute('fill', jc === c ? '#FFD600' : defaultFill)
        );
        onSelect(job);
      });

      svg.appendChild(grp);
    });
  });

  return svg;
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
    .then(d => renderDetail(content, d, backRoute))
    .catch(err => {
      console.error('[course-detail]', err);
      content.innerHTML = '<p class="cd-error">Could not load course details.</p>';
    });

  return el;
}

function renderDetail(content, d, backRoute) {
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

  // ── Apprenticeship notice ─────────────────────────────────────────────────
  if (d.qual_type && d.qual_type.includes('Apprenticeship')) {
    const notice = document.createElement('div');
    notice.className = 'cd-notice cd-notice--apprenticeship';
    notice.textContent = 'Apprenticeships are studied while you work and are offered through an employer.'
      + ' Check the provider’s webpage for details on how to apply.';
    content.appendChild(notice);
  }

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
      detailsPanel.appendChild(heading('cd-section-heading', "What you'll learn"));
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
    urlEl.textContent = `View course on ${d.provider || 'provider'} website →`;
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
      d.pathways.card_jobs.forEach(j =>
        pillsEl.appendChild(jobPill(j, d.course_id, d.course_title, backRoute))
      );
      briefEl.appendChild(pillsEl);
    }

    // Full state: bullet narrative + tiered network diagram
    const fullEl = document.createElement('div');
    fullEl.className = 'cd-pathways-full';
    fullEl.hidden = true;

    // Career narrative — prose paragraphs with optional inline bullet lists
    const narrativeSrc = d.pathways.narrative_bullets || d.pathways.narrative;
    if (narrativeSrc) {
      const narrativeEl = document.createElement('div');
      narrativeEl.className = 'cd-narrative';
      let currentList = null;
      narrativeSrc.split('\n').forEach(line => {
        const trimmed = line.trim();
        if (!trimmed) { currentList = null; return; }
        if (trimmed.startsWith('•')) {
          if (!currentList) {
            currentList = document.createElement('ul');
            currentList.className = 'cd-bullets';
            narrativeEl.appendChild(currentList);
          }
          const text = trimmed.replace(/^•\s*/, '');
          const li = document.createElement('li');
          li.className = 'cd-bullet-item';
          li.textContent = text;
          currentList.appendChild(li);
        } else {
          currentList = null;
          const p = document.createElement('p');
          p.className = 'cd-body';
          p.textContent = trimmed;
          narrativeEl.appendChild(p);
        }
      });
      fullEl.appendChild(narrativeEl);
    }

    if (d.pathways.curated_jobs?.length) {
      const hint = document.createElement('p');
      hint.className = 'cd-network-hint';
      hint.textContent = 'Tap a role to find out more';

      const reasoningPanel = document.createElement('div');
      reasoningPanel.className = 'cd-reasoning-panel';
      reasoningPanel.hidden = true;

      const reasoningTitle = document.createElement('button');
      reasoningTitle.className = 'cd-reasoning-title cd-reasoning-title--link';
      reasoningTitle.setAttribute('aria-label', 'View job details');

      const reasoningText = document.createElement('p');
      reasoningText.className = 'cd-reasoning-text';
      reasoningPanel.appendChild(reasoningTitle);
      reasoningPanel.appendChild(reasoningText);

      let _activeJob = null;
      reasoningTitle.addEventListener('click', () => {
        if (!_activeJob) return;
        go('job-detail', {
          jobId:      _activeJob.job_id,
          jobTitle:   _activeJob.title,
          backRoute:  'course-detail',
          backSlices: { courseId: d.course_id, courseTitle: d.course_title, backRoute },
        });
      });

      const diagram = buildGroupDiagram(
        d.course_title,
        d.pathways.curated_jobs,
        d.pathways.node_groups,
        (job) => {
          _activeJob = job;
          reasoningTitle.textContent = `${job.title} →`;
          reasoningText.textContent  = job.reasoning;
          reasoningPanel.hidden = false;
          hint.hidden = true;
        }
      );

      fullEl.appendChild(diagram);
      fullEl.appendChild(hint);
      fullEl.appendChild(reasoningPanel);
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
