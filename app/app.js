'use strict';

const API_BASE = '';

const SUBJECTS = ['Engineering', 'Digital & Tech', 'Construction', 'Health', 'Arts & Media'];

const QUALS = [
  { label: 'T Level',            filter: 'T Level',            tip: 'A two-year technical qualification equivalent to 3 A Levels, with an industry placement' },
  { label: 'Apprenticeship',     filter: 'Apprenticeship',     tip: 'Earn while you learn — work with an employer while studying for a qualification' },
  { label: 'HNC',                filter: 'HNC',                tip: 'Higher National Certificate — a one-year higher education qualification at Level 4' },
  { label: 'HND',                filter: 'HND',                tip: 'Higher National Diploma — a two-year higher education qualification at Level 5' },
  { label: 'Foundation Degree',  filter: 'Foundation Degree',  tip: 'A two-year higher education qualification, equivalent to the first two years of a degree' },
  { label: "Bachelor's Degree",  filter: "Bachelor's Degree",  tip: 'A full undergraduate degree, typically three years full-time' },
  { label: "Master's Degree",    filter: "Master's Degree",    tip: 'A postgraduate qualification, typically one year full-time after a degree' },
  { label: 'Access to HE',       filter: 'Access to HE',       tip: 'A Level 3 qualification designed to prepare adults for university study' },
  { label: 'Short Course',       filter: 'Short Course',       tip: 'A shorter qualification — duration and level varies' },
];

const LIST_CARD_THRESHOLD = 3;

// ─── Session ID ───────────────────────────────────────────────────────────────
function getOrCreateSessionId() {
  let sid = sessionStorage.getItem('pathwayiq_session_id');
  if (!sid) {
    sid = crypto.randomUUID();
    sessionStorage.setItem('pathwayiq_session_id', sid);
  }
  return sid;
}
const SESSION_ID = getOrCreateSessionId();

// ─── Analytics ────────────────────────────────────────────────────────────────
function logEvent(event, entityType, entityId, entityTitle, meta) {
  try {
    fetch('/analytics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id:   SESSION_ID,
        event,
        entity_type:  entityType  || null,
        entity_id:    entityId    || null,
        entity_title: entityTitle || null,
        meta:         meta ? JSON.stringify(meta) : null,
      }),
    });
  } catch (_) {}
}

logEvent('session_start');

// ─── State ────────────────────────────────────────────────────────────────────
let selectedSubject      = null;
let selectedQualFilter   = null;
let activeTooltipTile    = null;
let advisoryIntroShown   = false;
const sessionContext     = [];
const savedItems         = [];  // { id, type, title, cardElement }
const chatHistory        = [];  // { role: "user"|"assistant", content: string }
const browsingHistory    = [];  // { type: "course"|"career", title: string, id: string }
let   candidateSet       = null; // { course_ids, job_ids, built_from } — active search set

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const subjectGrid  = document.getElementById('subject-grid');
const qualGrid     = document.getElementById('qual-grid');
const thread       = document.getElementById('thread');
const tooltip      = document.getElementById('tooltip');
const centralZone  = document.getElementById('central-zone');
const chatInput    = document.getElementById('chat-input');
const sendBtn      = document.getElementById('send-btn');

// ─── API ──────────────────────────────────────────────────────────────────────
async function apiFetch(path) {
  const resp = await fetch(API_BASE + path);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

async function apiPost(path, body) {
  const resp = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

// ─── Salary helpers ───────────────────────────────────────────────────────────
function shortSalary(salary) {
  if (!salary) return null;
  // "£25,000 – £68,000" → "£25k–£68k"
  return salary
    .replace(/£(\d{1,3}),000/g, '£$1k')
    .replace(' – ', '–');
}

// ─── Thread helpers ───────────────────────────────────────────────────────────
function scrollToCard(el) {
  setTimeout(() => {
    const bottomBarHeight = document.getElementById('bottom-bar').offsetHeight;
    centralZone.scrollTo({ top: el.offsetTop - bottomBarHeight, behavior: 'smooth' });
  }, 50);
}

function addCard(el, title) {
  if (title) sessionContext.push(title);
  thread.appendChild(el);
  scrollToCard(el);
}

function addChatBubble(text) {
  const div = document.createElement('div');
  div.className = 'chat-bubble';
  div.textContent = text;
  thread.appendChild(div);
  scrollToCard(div);
}

function setLlmLine(text) {
  document.querySelector('.llm-line').textContent = text;
}

function addSystemBubble(text) {
  const div = document.createElement('div');
  div.className = 'system-bubble';
  div.textContent = text;
  thread.appendChild(div);
  scrollToCard(div);
}

function addTransitionLabel(text) {
  const div = document.createElement('div');
  div.className = 'transition-label';
  div.textContent = text;
  thread.appendChild(div);
}

// ─── Modal confirm ────────────────────────────────────────────────────────────
function showConfirm(message, confirmLabel, onConfirm) {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';

  const dialog = document.createElement('div');
  dialog.className = 'modal-dialog';

  const msg = document.createElement('p');
  msg.className = 'modal-message';
  msg.textContent = message;

  const btns = document.createElement('div');
  btns.className = 'modal-btns';

  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'modal-btn modal-btn--cancel';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', () => overlay.remove());

  const confirmBtn = document.createElement('button');
  confirmBtn.className = 'modal-btn modal-btn--confirm';
  confirmBtn.textContent = confirmLabel;
  confirmBtn.addEventListener('click', () => { overlay.remove(); onConfirm(); });

  btns.append(cancelBtn, confirmBtn);
  dialog.append(msg, btns);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
}

// ─── Pin button ───────────────────────────────────────────────────────────────
function isPinned(id, type) {
  return savedItems.some(item => item.id === id && item.type === type);
}

function makePinBtn(id, type, title, cardEl) {
  const btn = document.createElement('button');
  btn.className = 'action-btn pin-btn';

  // Reflect current saved state at creation time
  if (isPinned(id, type)) {
    btn.textContent = '📌 Saved';
    btn.classList.add('pin-btn--saved');
  } else {
    btn.textContent = '📌 Save';
  }

  btn.addEventListener('click', () => {
    const alreadySaved = savedItems.some(s => s.id === id && s.type === type);
    if (alreadySaved) {
      const idx = savedItems.findIndex(s => s.id === id && s.type === type);
      savedItems.splice(idx, 1);
      btn.textContent = '📌 Save';
      btn.classList.remove('pin-btn--saved');
    } else {
      savedItems.push({ id, type, title, cardElement: cardEl });
      btn.textContent = '📌 Saved';
      btn.classList.add('pin-btn--saved');
    }
  });

  return btn;
}

// ─── Saved items card ─────────────────────────────────────────────────────────
function buildSavedCard() {
  const card = document.createElement('div');
  card.className = 'card saved-card';

  const header = document.createElement('p');
  header.className = 'saved-card-header';
  header.textContent = 'Saved items';
  card.appendChild(header);

  if (savedItems.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'loading-text';
    empty.textContent = 'No saved items.';
    card.appendChild(empty);
    return card;
  }

  const list = document.createElement('div');
  list.className = 'saved-list';

  savedItems.forEach(item => {
    const row = document.createElement('div');
    row.className = 'saved-row';

    const typePill = pill(
      item.type === 'course' ? 'Course' : 'Career',
      item.type === 'course' ? 'pill-amber' : 'pill-green',
    );
    const titleSpan = document.createElement('span');
    titleSpan.className = 'saved-row-title';
    titleSpan.append(iconSvg(item.type), document.createTextNode(item.title));

    row.append(typePill, titleSpan);

    row.addEventListener('click', async () => {
      try {
        const apiPath = item.type === 'course' ? `/courses/${item.id}` : `/jobs/${item.id}`;
        const data = await apiFetch(apiPath);
        const normalized = { ...data, id: String(data.id) };
        addTransitionLabel(`You tapped ${item.title}`);
        if (item.type === 'course') {
          renderCourseCard(normalized, 0);
        } else {
          renderCareerCard(normalized);
        }
      } catch (err) {
        console.error('Could not load saved item', err);
      }
    });

    list.appendChild(row);
  });

  card.appendChild(list);
  return card;
}

// ─── Detail view ──────────────────────────────────────────────────────────────
function renderBulletField(text) {
  const items = text.split('\n')
    .map(line => line.replace(/^-\s*/, '').trim())
    .filter(line => line.length > 0);
  const ul = document.createElement('ul');
  items.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    ul.appendChild(li);
  });
  return ul;
}

function addDetailSection(panel, label, content, asList) {
  const h = document.createElement('p');
  h.className = 'detail-section-header';
  h.textContent = label;
  panel.appendChild(h);
  if (asList) {
    panel.appendChild(renderBulletField(content));
  } else {
    const p = document.createElement('p');
    p.className = 'detail-body';
    p.textContent = content;
    panel.appendChild(p);
  }
}

function buildDetailPanel(data, type) {
  const panel = document.createElement('div');
  panel.className = 'detail-panel';

  if (type === 'course') {
    if (data.overview)            addDetailSection(panel, 'Overview',             data.overview,            false);
    if (data.what_you_will_learn) addDetailSection(panel, 'What you will learn',  data.what_you_will_learn, true);
    if (data.entry_requirements)  addDetailSection(panel, 'Entry requirements',   data.entry_requirements,  true);
    if (data.progression)         addDetailSection(panel, 'Progression',          data.progression,         true);

    if (!data.overview && !data.what_you_will_learn) {
      const p = document.createElement('p');
      p.className = 'detail-body';
      p.textContent = 'No details available.';
      panel.appendChild(p);
    }

    if (data.course_url) {
      const a = document.createElement('a');
      a.className = 'detail-link';
      a.href = data.course_url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = `View course page at ${data.provider || 'provider'} ↗ (opens in new tab)`;
      panel.appendChild(a);
    }
  } else {
    // Career detail — named fields from extraction pass
    if (!data.overview) {
      // 36 records with no enriched_description have NULL content fields
      const p = document.createElement('p');
      p.className = 'detail-body';
      p.textContent = 'Full details available at the source link below.';
      panel.appendChild(p);
    } else {
      addDetailSection(panel, 'Overview',            data.overview,            false);
      addDetailSection(panel, 'Typical duties',      data.typical_duties,      true);
      addDetailSection(panel, 'Skills required',     data.skills_required,     true);
      addDetailSection(panel, 'Entry routes',        data.entry_routes,        true);
      if (data.salary) addDetailSection(panel, 'Salary', data.salary,          false);
      if (data.career_progression) addDetailSection(panel, 'Career progression', data.career_progression, true);
    }
    if (data.source_url) {
      const a = document.createElement('a');
      a.className = 'detail-link';
      a.href = data.source_url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = `View full profile at ${data.source || 'source'} ↗ (opens in new tab)`;
      panel.appendChild(a);
    }
  }
  return panel;
}

function attachDetailToggle(titleEl, cardEl, apiPath, type, iconType, onOpen) {
  // Prepend icon then wrap text — chevron stays right-aligned via space-between
  const text = titleEl.textContent;
  titleEl.textContent = '';
  if (iconType) titleEl.appendChild(iconSvg(iconType));
  const textSpan = document.createElement('span');
  textSpan.className = 'title-text';
  textSpan.textContent = text;
  titleEl.appendChild(textSpan);

  const chevron = document.createElement('span');
  chevron.className = 'chevron';
  chevron.textContent = 'Details ▾';
  titleEl.appendChild(chevron);

  let panel = null;
  let open  = false;

  async function toggle() {
    if (open) {
      panel.remove();
      panel = null;
      open  = false;
      chevron.textContent = 'Details ▾';
      return;
    }
    open = true;
    chevron.textContent = 'Details ▴';
    if (onOpen) onOpen();
    panel = document.createElement('div');
    panel.className = 'detail-panel';
    const loading = document.createElement('p');
    loading.className = 'loading-text';
    loading.style.fontStyle = 'italic';
    loading.textContent = 'Loading details…';
    panel.appendChild(loading);
    cardEl.appendChild(panel);

    try {
      const data = await apiFetch(apiPath);
      panel.replaceWith(buildDetailPanel(data, type));
      panel = cardEl.querySelector('.detail-panel');
      panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      panel.innerHTML = '<p class="error-text">Could not load details.</p>';
      console.error(err);
    }
  }

  titleEl.addEventListener('click', toggle);
}

// ─── Match dot ────────────────────────────────────────────────────────────────
function matchDot(score) {
  const colour = score >= 80 ? 'green' : 'amber';
  const [lo, hi] = colour === 'green' ? [80, 100] : [65, 79];
  const opacity = 0.45 + 0.55 * ((score - lo) / (hi - lo));

  const dot = document.createElement('span');
  dot.className = `match-dot ${colour}`;
  dot.style.opacity = opacity.toFixed(2);

  const label = document.createElement('span');
  label.className = `match-score-label ${colour}`;
  label.textContent = `${score}%`;
  label.style.opacity = opacity.toFixed(2);

  const wrapper = document.createElement('span');
  wrapper.className = 'match-indicator';
  wrapper.dataset.score = score;
  wrapper.append(dot, label);
  return wrapper;
}

// ─── Pill factory ─────────────────────────────────────────────────────────────
function pill(text, cls) {
  if (!text) return null;
  const span = document.createElement('span');
  span.className = `pill ${cls}`;
  span.textContent = text;
  return span;
}

// ─── Type icons ───────────────────────────────────────────────────────────────
function iconSvg(type) {
  const wrap = document.createElement('span');
  wrap.className = 'type-icon';
  if (type === 'course') {
    wrap.innerHTML = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <polygon points="8,2 15,6 8,10 1,6" fill="#1A1A2E"/>
      <rect x="11" y="6.5" width="2" height="5" rx="1" fill="#1A1A2E"/>
      <path d="M5 8v3c0 1.1 1.3 2 3 2s3-.9 3-2V8" stroke="#1A1A2E" stroke-width="1.5" fill="none" stroke-linecap="round"/>
    </svg>`;
  } else if (type === 'progression') {
    wrap.innerHTML = `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <rect x="2" y="13" width="4" height="4" rx="0.5" fill="#0d9488"/>
      <rect x="8" y="9" width="4" height="8" rx="0.5" fill="#0d9488"/>
      <rect x="14" y="5" width="4" height="12" rx="0.5" fill="#0d9488"/>
      <path d="M11 2l2.5 2.5L11 7" stroke="#0d9488" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
      <line x1="4" y1="4.5" x2="13" y2="4.5" stroke="#0d9488" stroke-width="1.5" stroke-linecap="round"/>
    </svg>`;
  } else {
    wrap.innerHTML = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="2" y="5" width="12" height="9" rx="2" fill="#007B83" stroke="#007B83" stroke-width="1.5"/>
      <path d="M5 5V4a1 1 0 011-1h4a1 1 0 011 1v1" stroke="#007B83" stroke-width="1.5" fill="none" stroke-linecap="round"/>
      <line x1="2" y1="9" x2="14" y2="9" stroke="#007B83" stroke-width="1"/>
    </svg>`;
  }
  return wrap;
}

// ─── Course card ──────────────────────────────────────────────────────────────
function buildCourseCard(course, rank) {
  const card = document.createElement('div');
  card.className = 'card course-card';
  card.dataset.courseId = course.id;

  // Type label
  const typeLabel = document.createElement('p');
  typeLabel.className = 'card-type';
  typeLabel.textContent = rank === 0 ? 'COURSE · BEST MATCH' : 'COURSE · ALSO RELEVANT';

  // Title
  const title = document.createElement('p');
  title.className = 'card-title';
  title.textContent = course.title;

  // Meta pills
  const meta = document.createElement('div');
  meta.className = 'card-meta';
  const pPill = pill(course.provider, 'pill-grey');
  if (pPill) meta.appendChild(pPill);
  const sPill = pill(course.subject_area, 'pill-grey');
  if (sPill) meta.appendChild(sPill);

  // Careers section
  const connLabel = document.createElement('p');
  connLabel.className = 'connections-label';
  connLabel.textContent = 'Career connections';

  const careerRows = document.createElement('div');
  careerRows.className = 'career-rows';
  const loading = document.createElement('p');
  loading.className = 'loading-text';
  loading.textContent = 'Loading careers…';
  careerRows.appendChild(loading);

  // Actions
  const actions = document.createElement('div');
  actions.className = 'card-actions';
  actions.appendChild(makePinBtn(course.id, 'course', course.title, card));

  card.append(typeLabel, title, meta, connLabel, careerRows, actions);
  attachDetailToggle(title, card, `/courses/${course.id}`, 'course', 'course', () => logEvent('course_detail_open', 'course', course.id, course.title));
  return { card, careerRows };
}

function populateCareerRows(careerRows, jobs) {
  careerRows.innerHTML = '';
  if (!jobs || jobs.length === 0) {
    const p = document.createElement('p');
    p.className = 'loading-text';
    p.textContent = 'No strong career connections found for this course.';
    careerRows.appendChild(p);
    return;
  }
  jobs.slice(0, 5).forEach(job => {
    logEvent('career_impression', 'job', job.id, job.title);
    const row = document.createElement('div');
    row.className = 'career-row';

    const name  = document.createElement('span');
    name.className = 'row-title';
    name.textContent = job.title;

    const sal = shortSalary(job.salary);
    const salSpan = document.createElement('span');
    salSpan.className = 'row-salary';
    salSpan.textContent = sal || '';

    const link = document.createElement('a');
    link.className = 'row-link';
    link.textContent = '↗';
    link.href = job.source_url || '#';
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.title = 'Open job profile';

    row.append(matchDot(job.match_score), iconSvg('career'), name, salSpan, link);

    // Caution flag — shown when domain score significantly exceeds skills alignment
    if (job.caution) {
      const warn = document.createElement('span');
      warn.className = 'caution-flag';
      warn.textContent = '⚠';
      warn.title = 'This course is in the same field but may not directly teach the skills required for this role';
      name.after(warn);

      const tooltip = document.createElement('p');
      tooltip.className = 'caution-tooltip';
      tooltip.textContent = 'This course is in the same field but may not directly teach the skills required for this role.';
      tooltip.hidden = true;

      warn.addEventListener('click', e => {
        e.stopPropagation();
        tooltip.hidden = !tooltip.hidden;
      });

      row.appendChild(tooltip);
    }

    // Tapping the row (not the ↗ or ⚠) opens a career card
    row.addEventListener('click', e => {
      if (e.target === link || link.contains(e.target)) return;
      if (e.target.classList.contains('caution-flag')) return;
      addTransitionLabel(`You tapped ${job.title}`);
      renderCareerCard(job);
    });

    careerRows.appendChild(row);
  });
}

async function renderCourseCard(course, rank) {
  const { card, careerRows } = buildCourseCard(course, rank);
  addCard(card, course.title);
  logEvent('course_impression', 'course', course.id, course.title);
  browsingHistory.push({ type: 'course', title: course.title, id: String(course.id) });

  // Load careers asynchronously
  try {
    const data = await apiFetch(`/courses/${course.id}/careers?limit=8`);
    populateCareerRows(careerRows, data.results);
  } catch (err) {
    careerRows.innerHTML = `<p class="error-text">Could not load careers.</p>`;
    console.error(err);
  }
}

// ─── Career card ──────────────────────────────────────────────────────────────
function buildCareerCard(job) {
  const card = document.createElement('div');
  card.className = 'card career-card';
  card.dataset.jobId = job.id;

  const typeLabel = document.createElement('p');
  typeLabel.className = 'card-type';
  typeLabel.textContent = 'CAREER PROFILE';

  const title = document.createElement('p');
  title.className = 'card-title';
  title.textContent = job.title;

  const meta = document.createElement('div');
  meta.className = 'card-meta';
  const salPill = pill(job.salary || null, 'pill-teal');
  const srcPill = pill(job.source || null, 'pill-grey');
  if (srcPill) meta.appendChild(srcPill);
  if (salPill) meta.appendChild(salPill);

  const progBtn = document.createElement('button');
  progBtn.className = 'pill pill-outline-teal';
  progBtn.textContent = 'Where could this role lead?';
  progBtn.addEventListener('click', () => {
    logEvent('progression_open', 'job', job.id, job.title);
    addTransitionLabel(`Progression from ${job.title}`);
    loadProgressionCard(job.id, job.title);
  });
  meta.appendChild(progBtn);

  const adzunaBtn = document.createElement('a');
  adzunaBtn.className = 'adzuna-link';
  adzunaBtn.href = `https://www.adzuna.co.uk/jobs/search?q=${encodeURIComponent('"' + job.title + '"')}&loc=67883`;
  adzunaBtn.target = '_blank';
  adzunaBtn.rel = 'noopener noreferrer';
  adzunaBtn.textContent = 'Find vacancies → (opens in new tab)';
  adzunaBtn.addEventListener('click', () => logEvent('adzuna_click', 'job', job.id, job.title));
  meta.appendChild(adzunaBtn);

  const connLabel = document.createElement('p');
  connLabel.className = 'connections-label';
  connLabel.textContent = 'Courses that lead here';

  const courseRows = document.createElement('div');
  courseRows.className = 'course-rows';
  const loading = document.createElement('p');
  loading.className = 'loading-text';
  loading.textContent = 'Loading courses…';
  courseRows.appendChild(loading);

  const actions = document.createElement('div');
  actions.className = 'card-actions';
  actions.appendChild(makePinBtn(job.id, 'job', job.title, card));

  card.append(typeLabel, title, meta, connLabel, courseRows, actions);
  attachDetailToggle(title, card, `/jobs/${job.id}`, 'career', 'career', () => logEvent('career_detail_open', 'job', job.id, job.title));
  return { card, courseRows };
}

function populateCourseRows(courseRows, courses) {
  courseRows.innerHTML = '';
  if (!courses || courses.length === 0) {
    const p = document.createElement('p');
    p.className = 'loading-text';
    p.textContent = 'No strong course connections found for this career.';
    courseRows.appendChild(p);
    return;
  }
  courses.slice(0, 5).forEach((course, i) => {
    const row = document.createElement('div');
    row.className = 'course-row';

    const name = document.createElement('span');
    name.className = 'row-title';
    name.textContent = course.title;

    row.append(matchDot(course.match_score), iconSvg('course'), name);

    // Caution flag — shown when domain score significantly exceeds skills alignment
    if (course.caution) {
      const warn = document.createElement('span');
      warn.className = 'caution-flag';
      warn.textContent = '⚠';
      warn.title = 'This career is in the same field as this course but may require skills not directly taught by it';

      const tooltip = document.createElement('p');
      tooltip.className = 'caution-tooltip';
      tooltip.textContent = 'This career is in the same field as this course but may require skills not directly taught by it.';
      tooltip.hidden = true;

      warn.addEventListener('click', e => {
        e.stopPropagation();
        tooltip.hidden = !tooltip.hidden;
      });

      name.after(warn);
      row.appendChild(tooltip);
    }

    row.addEventListener('click', e => {
      if (e.target.classList.contains('caution-flag')) return;
      addTransitionLabel(`You tapped ${course.title}`);
      renderCourseCard(course, i);
    });

    courseRows.appendChild(row);
  });
}

async function renderCareerCard(job) {
  const { card, courseRows } = buildCareerCard(job);
  addCard(card, job.title);
  browsingHistory.push({ type: 'career', title: job.title, id: String(job.id) });

  try {
    const data = await apiFetch(`/jobs/${job.id}/courses?limit=8`);
    populateCourseRows(courseRows, data.results);
  } catch (err) {
    courseRows.innerHTML = `<p class="error-text">Could not load courses.</p>`;
    console.error(err);
  }
}

// ─── Progression card ─────────────────────────────────────────────────────────
function progressionRoleRow(job) {
  const row = document.createElement('div');
  row.className = 'progression-role-row';
  const title = document.createElement('span');
  title.className = 'progression-role-title';
  title.textContent = job.title;
  row.appendChild(title);
  row.addEventListener('click', () => {
    addTransitionLabel(`You tapped ${job.title}`);
    renderCareerCard(job);
  });
  return row;
}

function buildProgressionCard(data, currentJobTitle, jobId) {
  const card = document.createElement('div');
  card.className = 'card progression-card';

  const typeLabel = document.createElement('p');
  typeLabel.className = 'card-type';
  typeLabel.textContent = 'CAREER PATHWAY';

  const heading = document.createElement('p');
  heading.className = 'card-title';
  heading.textContent = 'Career pathway';

  card.append(typeLabel, heading);

  if (!data.has_progression) {
    const empty = document.createElement('p');
    empty.className = 'loading-text';
    empty.textContent = 'No progression pathway found for this role.';
    card.appendChild(empty);
    return card;
  }

  // Narrative — most prominent element
  const narrative = document.createElement('p');
  narrative.className = 'progression-narrative';
  narrative.textContent = data.narrative;
  card.appendChild(narrative);

  // Tell me more button + inline expansion
  const moreBtn = document.createElement('button');
  moreBtn.className = 'pill pill-outline-teal';
  moreBtn.textContent = 'Tell me more';
  moreBtn.style.marginTop = '4px';
  moreBtn.style.marginBottom = '16px';

  const morePanel = document.createElement('div');
  morePanel.className = 'progression-more-panel';
  morePanel.style.display = 'none';

  moreBtn.addEventListener('click', async () => {
    if (morePanel.style.display !== 'none') {
      morePanel.style.display = 'none';
      moreBtn.textContent = 'Tell me more';
      return;
    }
    moreBtn.textContent = 'Loading…';
    moreBtn.disabled = true;
    try {
      const data = await apiFetch(`/jobs/${jobId}/explain`);
      morePanel.textContent = data.text;
      morePanel.style.display = 'block';
      moreBtn.textContent = 'Show less';
    } catch (err) {
      morePanel.textContent = 'Could not load additional information.';
      morePanel.style.display = 'block';
      moreBtn.textContent = 'Tell me more';
    }
    moreBtn.disabled = false;
  });

  card.appendChild(moreBtn);
  card.appendChild(morePanel);

  // Ladder: outbound (higher) → current → inbound (lower)
  const ladder = document.createElement('div');
  ladder.className = 'progression-ladder';

  // Outbound section
  const outboundSection = document.createElement('div');
  outboundSection.className = 'progression-section';
  if (data.outbound && data.outbound.length > 0) {
    data.outbound.forEach(j => outboundSection.appendChild(progressionRoleRow(j)));
  } else {
    const empty = document.createElement('p');
    empty.className = 'progression-empty';
    empty.textContent = 'No progression roles found in our database';
    outboundSection.appendChild(empty);
  }

  const arrowTop = document.createElement('div');
  arrowTop.className = 'progression-arrow';
  arrowTop.textContent = '↑';

  // Current role
  const currentRow = document.createElement('div');
  currentRow.className = 'progression-current';
  const currentTitle = document.createElement('span');
  currentTitle.className = 'progression-current-title';
  currentTitle.textContent = currentJobTitle;
  const currentLabel = document.createElement('span');
  currentLabel.className = 'progression-current-label';
  currentLabel.textContent = 'You are here';
  currentRow.append(currentTitle, currentLabel);

  const arrowBottom = document.createElement('div');
  arrowBottom.className = 'progression-arrow';
  arrowBottom.textContent = '↑';

  // Inbound section
  const inboundSection = document.createElement('div');
  inboundSection.className = 'progression-section';
  if (data.inbound && data.inbound.length > 0) {
    data.inbound.forEach(j => inboundSection.appendChild(progressionRoleRow(j)));
  } else {
    const empty = document.createElement('p');
    empty.className = 'progression-empty';
    empty.textContent = 'No entry-level pathways found in our database';
    inboundSection.appendChild(empty);
  }

  ladder.append(outboundSection, arrowTop, currentRow, arrowBottom, inboundSection);
  card.appendChild(ladder);

  const disclaimer = document.createElement('p');
  disclaimer.className = 'ai-disclaimer';
  disclaimer.textContent = 'Pathway suggestions are AI-generated and may not reflect every route into or out of this role.';
  card.appendChild(disclaimer);

  return card;
}

async function loadProgressionCard(jobId, currentJobTitle) {
  // Show loading card immediately
  const loadingCard = document.createElement('div');
  loadingCard.className = 'card progression-card';
  const loadTypeLabel = document.createElement('p');
  loadTypeLabel.className = 'card-type';
  loadTypeLabel.textContent = 'CAREER PATHWAY';
  const loadingText = document.createElement('p');
  loadingText.className = 'loading-text';
  loadingText.style.fontStyle = 'italic';
  loadingText.textContent = 'Preparing career pathway…';
  loadingCard.append(loadTypeLabel, loadingText);
  addCard(loadingCard, 'Career pathway');

  try {
    const data = await apiFetch(`/jobs/${jobId}/progression`);
    const newCard = buildProgressionCard(data, currentJobTitle, jobId);
    loadingCard.replaceWith(newCard);
  } catch (err) {
    loadingText.textContent = 'Could not load career pathway.';
    loadingText.style.fontStyle = '';
    console.error('Progression load failed:', err);
  }
}

// ─── List card ────────────────────────────────────────────────────────────────
const LIST_ROWS_INITIAL = 8;

function buildListCard(results, type, contextLabel) {
  const isCourse = type === 'course';
  const card = document.createElement('div');
  card.className = `card card--list card--list-${isCourse ? 'course' : 'job'}`;

  // Header
  const header = document.createElement('div');
  header.className = 'list-card-header';

  const typeCount = document.createElement('span');
  typeCount.className = 'list-card-type';
  typeCount.textContent = `${isCourse ? 'COURSES' : 'CAREERS'} · ${results.length} RESULTS`;

  header.appendChild(typeCount);
  if (contextLabel) {
    const ctx = pill(contextLabel, 'pill-grey');
    ctx.style.marginLeft = 'auto';
    header.appendChild(ctx);
  }
  card.appendChild(header);

  // Rows container
  const rowsEl = document.createElement('div');
  rowsEl.className = 'list-rows';

  results.forEach((item, idx) => {
    const row = document.createElement('div');
    row.className = 'list-row';
    if (idx >= LIST_ROWS_INITIAL) row.classList.add('list-row--hidden');

    if (isCourse) {
      const titleEl = document.createElement('span');
      titleEl.className = 'list-row-title';
      titleEl.append(iconSvg('course'), document.createTextNode(item.title));

      const subEl = document.createElement('span');
      subEl.className = 'list-row-sub';
      subEl.textContent = item.provider || '';

      const right = document.createElement('div');
      right.className = 'list-row-right';
      const qPill = pill(item.qualification_type || null, 'pill-amber');
      if (qPill) right.appendChild(qPill);

      const left = document.createElement('div');
      left.className = 'list-row-left';
      left.append(titleEl, subEl);

      row.append(left, right);
      row.addEventListener('click', () => {
        addTransitionLabel(item.title);
        renderCourseCard(item, 0);
      });
    } else {
      const titleEl = document.createElement('span');
      titleEl.className = 'list-row-title';
      titleEl.append(iconSvg('career'), document.createTextNode(item.title));

      const subEl = document.createElement('span');
      subEl.className = 'list-row-sub';
      subEl.textContent = item.salary || '';

      const right = document.createElement('div');
      right.className = 'list-row-right';
      const srcPill = pill(item.source || null, 'pill-grey');
      if (srcPill) right.appendChild(srcPill);

      const left = document.createElement('div');
      left.className = 'list-row-left';
      left.append(titleEl, subEl);

      row.append(left, right);
      row.addEventListener('click', () => {
        addTransitionLabel(item.title);
        renderCareerCard(item);
      });
    }

    rowsEl.appendChild(row);
  });

  card.appendChild(rowsEl);

  // Expand button (only if more than LIST_ROWS_INITIAL rows)
  if (results.length > LIST_ROWS_INITIAL) {
    const expandBtn = document.createElement('button');
    expandBtn.className = 'list-expand-btn';
    expandBtn.textContent = `Show all ${results.length}`;
    let expanded = false;
    expandBtn.addEventListener('click', () => {
      expanded = !expanded;
      rowsEl.querySelectorAll('.list-row--hidden').forEach(r => {
        r.style.display = expanded ? 'flex' : '';
      });
      if (expanded) {
        rowsEl.classList.add('list-rows--expanded');
        expandBtn.textContent = 'Show fewer';
      } else {
        rowsEl.classList.remove('list-rows--expanded');
        expandBtn.textContent = `Show all ${results.length}`;
      }
    });
    card.appendChild(expandBtn);
  }

  return card;
}

function renderListCard(results, type, contextLabel) {
  const card = buildListCard(results, type, contextLabel);
  addCard(card, null);
}

// ─── Advisory card ────────────────────────────────────────────────────────────
function buildAdvisoryCard(advisory) {
  const card = document.createElement('div');
  card.className = 'card card--advisory';

  const header = document.createElement('p');
  header.className = 'advisory-header';
  header.textContent = '✶ YOU MIGHT ALSO FIND THIS INTERESTING';

  const typeLabel = document.createElement('p');
  typeLabel.className = 'card-type';
  typeLabel.textContent = advisory.type === 'course' ? 'COURSE' : 'CAREER PROFILE';
  // Inherit colour from .course-card / .career-card not available here — set inline
  typeLabel.style.color = advisory.type === 'course' ? 'var(--teal)' : 'var(--green)';

  const title = document.createElement('p');
  title.className = 'card-title';
  title.textContent = advisory.title;

  const meta = document.createElement('div');
  meta.className = 'card-meta';
  if (advisory.type === 'course') {
    const pPill = pill(advisory.provider, 'pill-grey');
    if (pPill) meta.appendChild(pPill);
  } else {
    const salPill = pill(advisory.salary || null, 'pill-teal');
    const srcPill = pill(advisory.source || null, 'pill-grey');
    if (salPill) meta.appendChild(salPill);
    if (srcPill) meta.appendChild(srcPill);
  }

  const explanation = document.createElement('p');
  explanation.className = 'advisory-explanation';
  explanation.textContent = advisory.explanation;

  const cta = document.createElement('button');
  cta.className = 'advisory-cta';
  cta.textContent = 'Explore this →';
  cta.addEventListener('click', () => {
    addTransitionLabel(`You tapped ${advisory.title}`);
    if (advisory.type === 'course') {
      renderCourseCard(advisory, 0);
    } else {
      renderCareerCard(advisory);
    }
  });

  card.append(header, typeLabel, title, meta, explanation, cta);

  const apiPath = advisory.type === 'course'
    ? `/courses/${advisory.id}`
    : `/jobs/${advisory.id}`;
  attachDetailToggle(title, card, apiPath, advisory.type === 'course' ? 'course' : 'career', advisory.type);

  return card;
}

function renderAdvisoryCard(advisory) {
  if (!advisoryIntroShown) {
    advisoryIntroShown = true;
    addSystemBubble(
      "I'm keeping an eye on what you're exploring and I'll surface things you might not have thought to look for."
    );
  }
  const card = buildAdvisoryCard(advisory);
  addCard(card, advisory.title);
}

// ─── Subject / qualification selection ───────────────────────────────────────
async function loadCourses() {
  if (!selectedSubject) return;

  // Clear thread
  thread.innerHTML = '';
  setLlmLine('');

  const label = selectedQualFilter
    ? `You selected ${selectedSubject} · ${QUALS.find(q => q.filter === selectedQualFilter)?.label}`
    : `You selected ${selectedSubject}`;
  addTransitionLabel(label);

  let url = `/search/courses?subject=${encodeURIComponent(selectedSubject)}`;
  if (selectedQualFilter) url += `&qualification=${encodeURIComponent(selectedQualFilter)}`;

  try {
    const data = await apiFetch(url);
    if (data.results.length === 0) {
      const p = document.createElement('p');
      p.className = 'loading-text';
      p.style.textAlign = 'center';
      p.style.padding = '16px 0';
      p.textContent = 'No courses found — try a different subject or qualification.';
      thread.appendChild(p);
      return;
    }
    if (data.candidate_set) candidateSet = data.candidate_set;
    if (data.results.length >= LIST_CARD_THRESHOLD) {
      renderListCard(data.results, 'course', selectedSubject);
    } else {
      for (let i = 0; i < data.results.length; i++) {
        renderCourseCard(data.results[i], i);
      }
    }
  } catch (err) {
    const p = document.createElement('p');
    p.className = 'error-text';
    p.style.textAlign = 'center';
    p.style.padding = '16px 0';
    p.textContent = 'Could not connect to the API. Is the server running?';
    thread.appendChild(p);
    console.error(err);
  }
}

function selectSubject(subject, tileEl) {
  const proceed = () => {
    logEvent('tile_tap', null, null, null, { tile_type: 'subject', tile_label: subject });
    selectedSubject       = subject;
    selectedQualFilter    = null;
    chatHistory.length    = 0;
    browsingHistory.length = 0;
    candidateSet          = null;

    subjectGrid.querySelectorAll('.subject-tile').forEach(t => t.classList.remove('active'));
    tileEl.classList.add('active');

    qualGrid.querySelectorAll('.qual-tile').forEach(t => {
      t.classList.add('enabled');
      t.classList.remove('active');
    });

    loadCourses();
  };

  if (thread.children.length > 0) {
    showConfirm(
      'Start a new search? Your current results will be cleared.',
      'Start new search',
      proceed,
    );
  } else {
    proceed();
  }
}

function toggleQual(filter, tileEl) {
  if (!selectedSubject) return;

  const proceed = () => {
    logEvent('tile_tap', null, null, null, { tile_type: 'qual', tile_label: filter });
    if (selectedQualFilter === filter) {
      selectedQualFilter = null;
      tileEl.classList.remove('active');
    } else {
      selectedQualFilter = filter;
      qualGrid.querySelectorAll('.qual-tile').forEach(t => t.classList.remove('active'));
      tileEl.classList.add('active');
    }
    loadCourses();
  };

  if (thread.children.length > 0) {
    showConfirm(
      'Start a new search? Your current results will be cleared.',
      'Start new search',
      proceed,
    );
  } else {
    proceed();
  }
}

// ─── Chat input ───────────────────────────────────────────────────────────────
async function handleChatSubmit() {
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = '';
  logEvent('chat_submit', null, null, null, { query: message });

  chatHistory.push({ role: 'user', content: message });
  addChatBubble(message);
  setLlmLine('Thinking…');

  try {
    const data = await apiPost('/chat', {
      message,
      session_id:       SESSION_ID,
      session_context:  sessionContext,
      chat_history:     chatHistory.slice(0, -1),  // prior turns; current already appended
      browsing_history: browsingHistory,
      saved_items: {
        courses: savedItems.filter(i => i.type === 'course').map(i => ({ title: i.title, id: i.id })),
        careers: savedItems.filter(i => i.type === 'job').map(i => ({ title: i.title, id: i.id })),
      },
      candidate_set: candidateSet,
    });

    if (data.candidate_set !== undefined) candidateSet = data.candidate_set;

    setLlmLine(data.acknowledgement);
    if (data.acknowledgement) {
      chatHistory.push({ role: 'assistant', content: data.acknowledgement });
    }

    if (data.response_text) {
      addSystemBubble(data.response_text);
      chatHistory.push({ role: 'assistant', content: data.response_text });
    } else if (data.results && data.results.length > 0) {
      data.results.forEach(r => logEvent('chat_impression', r.type === 'job' ? 'job' : 'course', r.id, r.title));
      addTransitionLabel(`From your query: "${message}"`);
      const courseResults = data.results.filter(r => r.type !== 'job');
      const jobResults    = data.results.filter(r => r.type === 'job');

      if (courseResults.length >= LIST_CARD_THRESHOLD) {
        renderListCard(courseResults, 'course', null);
      } else {
        courseResults.forEach((r, i) => renderCourseCard(r, i));
      }

      if (jobResults.length >= LIST_CARD_THRESHOLD) {
        renderListCard(jobResults, 'job', null);
      } else {
        jobResults.forEach(r => renderCareerCard(r));
      }
    } else {
      addTransitionLabel('No results found for that query.');
    }

    if (data.advisory) {
      renderAdvisoryCard(data.advisory);
    }
  } catch (err) {
    setLlmLine('Sorry, I could not process that query.');
    console.error(err);
  }
}

sendBtn.addEventListener('click', handleChatSubmit);
chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') handleChatSubmit(); });

// ─── Tooltip ──────────────────────────────────────────────────────────────────
function showTooltip(text, anchorEl) {
  if (activeTooltipTile === anchorEl) {
    hideTooltip();
    return;
  }
  activeTooltipTile = anchorEl;
  tooltip.textContent = text;
  tooltip.classList.remove('hidden');

  const rect = anchorEl.getBoundingClientRect();
  const gap = 8;
  let top = rect.top - tooltip.offsetHeight - gap;
  let left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2;

  // Keep inside viewport
  if (top < 8) top = rect.bottom + gap;
  left = Math.max(8, Math.min(left, window.innerWidth - tooltip.offsetWidth - 8));

  tooltip.style.top  = `${top}px`;
  tooltip.style.left = `${left}px`;
}

function hideTooltip() {
  tooltip.classList.add('hidden');
  activeTooltipTile = null;
}

document.addEventListener('click', e => {
  if (!e.target.closest('.info-btn')) hideTooltip();
});

// ─── Saved items / grid icon ──────────────────────────────────────────────────
document.querySelector('.session-btn').addEventListener('click', () => {
  if (savedItems.length === 0) {
    addSystemBubble("There are no entries in your saved list. Tap the pin icon on any card to save it.");
    return;
  }
  showConfirm(
    'View your saved items? Your current results will be cleared.',
    'View saved items',
    () => {
      thread.innerHTML = '';
      setLlmLine('');
      chatHistory.length     = 0;
      browsingHistory.length = 0;
      const card = buildSavedCard();
      thread.appendChild(card);
      scrollToCard(card);
    },
  );
});

// ─── Render start screen ──────────────────────────────────────────────────────
function renderStartScreen() {
  // Subject tiles
  SUBJECTS.forEach(subject => {
    const tile = document.createElement('button');
    tile.className = 'subject-tile';
    tile.textContent = subject;
    tile.addEventListener('click', () => selectSubject(subject, tile));
    subjectGrid.appendChild(tile);
  });

  // Qualification tiles
  QUALS.forEach(({ label, filter, tip }) => {
    const tile = document.createElement('div');
    tile.className = 'qual-tile';

    const labelSpan = document.createElement('span');
    labelSpan.textContent = label;

    const infoBtn = document.createElement('button');
    infoBtn.className = 'info-btn';
    infoBtn.textContent = 'ⓘ';
    infoBtn.title = tip;
    infoBtn.addEventListener('click', e => {
      e.stopPropagation();
      showTooltip(tip, infoBtn);
    });

    tile.append(labelSpan, infoBtn);
    tile.addEventListener('click', () => toggleQual(filter, tile));
    qualGrid.appendChild(tile);
  });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
renderStartScreen();

// ─── Pathway modal ────────────────────────────────────────────────────────────

const PATHWAY_EXPLANATIONS = {
  gcse: {
    heading: 'GCSEs / Level 2',
    body: "GCSEs are the qualifications most students take at the end of secondary school, usually at age 16. They're the starting point for most further study. You don't need specific GCSEs for every path — but having a solid set, especially in English and Maths, opens more doors.",
    leads_to: 'T Levels, A Levels, Apprenticeships',
    gmiot: false,
  },
  t_level: {
    heading: 'T Level',
    body: 'A T Level is a two-year technical qualification equivalent to three A Levels. It combines classroom learning with a substantial industry placement (at least 45 days with an employer), so you graduate with both a qualification and real work experience. Designed specifically for students who want a career in a technical field.',
    leads_to: 'HNC, HND, or degree-level study',
    gmiot: true,
  },
  a_levels: {
    heading: 'A Levels',
    not_gmiot: true,
  },
  apprenticeship: {
    heading: 'Apprenticeship (Level 3)',
    body: "An apprenticeship lets you earn a wage while working towards a qualification. You spend most of your time with an employer and some time in college or training. A Level 3 apprenticeship is equivalent to A Levels. A good route if you know the industry you want to work in and prefer learning on the job.",
    leads_to: 'Higher Apprenticeship or further study',
    gmiot: true,
  },
  hnc_hnd: {
    heading: 'HNC / HND',
    body: "HNCs (Higher National Certificates) and HNDs (Higher National Diplomas) are higher education qualifications delivered at college. An HNC is one year full-time (Level 4, equivalent to the first year of a degree). An HND is two years full-time (Level 5, equivalent to the first two years of a degree). Both can lead directly into employment or onto the second or third year of a Bachelor's degree.",
    leads_to: "Bachelor's degree (often with advanced entry)",
    gmiot: true,
  },
  foundation_degree: {
    heading: 'Foundation Degree',
    body: "A Foundation Degree is a two-year higher education qualification at Level 5 — equivalent to the first two years of a Bachelor's degree. Usually delivered at college in partnership with a university. Can be topped up to a full Bachelor's degree in one further year.",
    leads_to: "Bachelor's degree top-up",
    gmiot: true,
  },
  higher_apprenticeship: {
    heading: 'Higher Apprenticeship',
    body: "A Higher Apprenticeship works the same way as a Level 3 apprenticeship — earning while learning — but at a higher level (Level 4–6). You could come out with an HNC, HND, or even a degree-level qualification, all while being employed. Increasingly common in engineering, digital, and construction sectors.",
    leads_to: "Employment or Bachelor's degree",
    gmiot: false,
  },
  bachelors: {
    heading: "Bachelor's Degree",
    body: "A Bachelor's degree is a full undergraduate degree, typically three years full-time at university (four in Scotland or for some courses with a placement year). It's Level 6 on the qualification framework. The most common route into graduate-level professional roles. Can be entered directly from A Levels, T Levels, or via top-up from an HND or Foundation Degree.",
    leads_to: "Employment or Master's degree",
    gmiot: true,
  },
  masters: {
    heading: "Master's Degree",
    body: "A Master's degree is a postgraduate qualification taken after a Bachelor's degree, usually one year full-time. It allows you to specialise deeply in a subject or field, and is often required or preferred for senior professional or research roles.",
    leads_to: 'Senior professional roles or research',
    gmiot: true,
  },
};

let activePathwayNode = null;

function renderPathwayExplanation(nodeKey) {
  const data = PATHWAY_EXPLANATIONS[nodeKey];
  if (!data) return;

  // Update active-node highlight
  document.querySelectorAll('.pathway-node--active').forEach(el => el.classList.remove('pathway-node--active'));
  const nodeEl = document.querySelector(`[data-node="${nodeKey}"]`);
  if (nodeEl) nodeEl.classList.add('pathway-node--active');
  activePathwayNode = nodeKey;

  const panel = document.getElementById('pathway-explanation');
  panel.hidden = false;
  panel.innerHTML = '';

  const heading = document.createElement('p');
  heading.className = 'pathway-exp-heading';
  heading.textContent = data.heading;
  panel.appendChild(heading);

  if (data.not_gmiot) {
    const note = document.createElement('p');
    note.className = 'pathway-exp-body';
    note.textContent = 'A Levels are a well-established academic route, typically taken in sixth form or college. GMIoT does not offer A Level courses — but A Levels remain a valid path into HNC, HND, Foundation Degree, or university study.';
    panel.appendChild(note);
  } else {
    if (data.body) {
      const body = document.createElement('p');
      body.className = 'pathway-exp-body';
      body.textContent = data.body;
      panel.appendChild(body);
    }
    if (data.leads_to) {
      const leads = document.createElement('p');
      leads.className = 'pathway-exp-leads';
      leads.innerHTML = `<strong>Leads to:</strong> ${data.leads_to}`;
      panel.appendChild(leads);
    }
    if (data.gmiot) {
      const indicator = document.createElement('p');
      indicator.className = 'pathway-exp-gmiot';
      indicator.textContent = 'GMIoT offers courses at this level — explore them in the app.';
      panel.appendChild(indicator);
    }
  }

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function openPathwayModal() {
  const modal = document.getElementById('pathway-modal');
  modal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  // Reset state
  const expPanel = document.getElementById('pathway-explanation');
  expPanel.hidden = true;
  expPanel.innerHTML = '';
  document.querySelectorAll('.pathway-node--active').forEach(el => el.classList.remove('pathway-node--active'));
  activePathwayNode = null;
  modal.querySelector('.pathway-panel').scrollTop = 0;
  document.getElementById('pathway-close').focus();
}

function closePathwayModal() {
  document.getElementById('pathway-modal').classList.add('hidden');
  document.body.style.overflow = '';
}

document.getElementById('pathway-link').addEventListener('click', e => {
  e.preventDefault();
  openPathwayModal();
});

document.getElementById('pathway-close').addEventListener('click', closePathwayModal);

document.getElementById('pathway-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('pathway-modal')) closePathwayModal();
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !document.getElementById('pathway-modal').classList.contains('hidden')) {
    closePathwayModal();
  }
});

document.querySelectorAll('.pathway-node').forEach(node => {
  const key = node.dataset.node;

  node.addEventListener('click', () => {
    if (activePathwayNode === key) {
      node.classList.remove('pathway-node--active');
      activePathwayNode = null;
      const panel = document.getElementById('pathway-explanation');
      panel.hidden = true;
      panel.innerHTML = '';
    } else {
      renderPathwayExplanation(key);
    }
  });

  node.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); node.click(); }
  });
});
