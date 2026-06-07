/* Saved items screen — courses + job roles + campus map */

import { state, unsaveItem, refreshSavedBadges } from '../state.js';
import { go } from '../router.js';
import { subject } from '../subjects.js';

const CLOSE_SVG = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
</svg>`;

const REMOVE_SVG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
</svg>`;

function makeRow(item, onTap, onRemove) {
  const colour = subject(item.ssa).colour;
  const row = document.createElement('div');
  row.className = 'saved-row';
  row.style.setProperty('--sub', colour);

  const titleBtn = document.createElement('button');
  titleBtn.className = 'saved-row-title';

  const titleLine = document.createElement('div');
  titleLine.textContent = item.title;
  titleBtn.appendChild(titleLine);

  if (item.subtitle) {
    const sub = document.createElement('div');
    sub.className = 'saved-row-sub';
    sub.textContent = item.subtitle;
    titleBtn.appendChild(sub);
  }

  titleBtn.addEventListener('click', onTap);

  const removeBtn = document.createElement('button');
  removeBtn.className = 'saved-row-remove';
  removeBtn.setAttribute('aria-label', `Remove ${item.title}`);
  removeBtn.innerHTML = REMOVE_SVG;
  removeBtn.addEventListener('click', () => {
    unsaveItem(item.id, item.type);
    refreshSavedBadges();
    onRemove(row);
  });

  row.appendChild(titleBtn);
  row.appendChild(removeBtn);
  return row;
}

function makeSection(label) {
  const wrap = document.createElement('div');
  wrap.className = 'saved-section';
  const h = document.createElement('p');
  h.className = 'saved-section-title';
  h.textContent = label;
  wrap.appendChild(h);
  return wrap;
}

function buildMap(courseIds) {
  const wrap = document.createElement('div');
  wrap.className = 'saved-map-wrap';
  const mapEl = document.createElement('div');
  mapEl.className = 'saved-map';
  wrap.appendChild(mapEl);

  setTimeout(async () => {
    const L = window.L;
    if (!L) { wrap.style.display = 'none'; return; }

    const map = L.map(mapEl, { zoomControl: true });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> © <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    const markers = [];

    try {
      const resp = await fetch('/saved/campuses', {
        method:  'POST',
        headers: { 'content-type': 'application/json' },
        body:    JSON.stringify({ course_ids: courseIds }),
      });
      const campuses = await resp.json();
      campuses.forEach(campus => {
        const courseList = campus.courses.map(t => `<li>${t}</li>`).join('');
        const popup = `<strong>${campus.provider}</strong><br>${campus.campus_name} · ${campus.postcode}`
          + `<ul style="margin:6px 0 0;padding-left:16px;">${courseList}</ul>`;
        markers.push(L.marker([campus.lat, campus.lng]).addTo(map).bindPopup(popup));
      });
    } catch (e) {
      console.error('[saved-list] campus fetch error', e);
    }

    if (markers.length > 0) {
      map.fitBounds(L.featureGroup(markers).getBounds().pad(0.2));
    } else {
      map.setView([53.483, -2.244], 11);
    }

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(pos => {
        const userMarker = L.circleMarker(
          [pos.coords.latitude, pos.coords.longitude],
          { radius: 8, color: '#0372FF', fillColor: '#0372FF', fillOpacity: 0.9, weight: 2 }
        ).addTo(map).bindPopup('Your location');
        markers.push(userMarker);
        map.fitBounds(L.featureGroup(markers).getBounds().pad(0.2));
      }, () => {});
    }
  }, 0);

  return wrap;
}

export function SavedListView(slices = {}) {
  const backRoute = slices.backRoute || 'chat-first';

  const el = document.createElement('div');
  el.className = 'view view-saved-list';

  // Header
  const header = document.createElement('div');
  header.className = 'saved-head';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'saved-head-close';
  closeBtn.setAttribute('aria-label', 'Close saved items');
  closeBtn.innerHTML = CLOSE_SVG;
  closeBtn.addEventListener('click', () => go(backRoute));

  const titleEl = document.createElement('span');
  titleEl.className = 'saved-head-title';
  titleEl.textContent = 'Saved';

  const countEl = document.createElement('span');
  countEl.className = 'saved-head-count';
  countEl.textContent = state.saved.items.length;

  header.appendChild(closeBtn);
  header.appendChild(titleEl);
  header.appendChild(countEl);
  el.appendChild(header);

  const courses = state.saved.items.filter(i => i.type === 'course');
  const careers = state.saved.items.filter(i => i.type === 'job');

  if (courses.length === 0 && careers.length === 0) {
    el.appendChild(emptyState());
    return el;
  }

  function emptyState() {
    const p = document.createElement('p');
    p.className = 'saved-list-empty';
    p.textContent = 'Nothing saved yet — tap the bookmark on any course or role to save it here.';
    return p;
  }

  function checkEmpty() {
    if (state.saved.items.length === 0) el.appendChild(emptyState());
  }

  // Courses section
  let mapWrap = null;
  if (courses.length > 0) {
    const courseSection = makeSection('Courses');
    const rows = document.createElement('div');
    rows.className = 'saved-rows';

    courses.forEach(item => rows.appendChild(makeRow(
      item,
      () => go('course-detail', {
        courseId:    item.id,
        courseTitle: item.title,
        backRoute:   'saved-list',
        backSlices:  { backRoute },
      }),
      (row) => {
        row.remove();
        if (!rows.querySelector('.saved-row')) {
          courseSection.remove();
          if (mapWrap) { mapWrap.remove(); mapWrap = null; }
        }
        checkEmpty();
      }
    )));

    courseSection.appendChild(rows);
    el.appendChild(courseSection);
    mapWrap = buildMap(courses.map(i => i.id));
    el.appendChild(mapWrap);
  }

  // Job roles section
  if (careers.length > 0) {
    const careerSection = makeSection('Job roles');
    const rows = document.createElement('div');
    rows.className = 'saved-rows';

    careers.forEach(item => rows.appendChild(makeRow(
      item,
      () => go('job-detail', {
        jobId:      item.id,
        jobTitle:   item.title,
        backRoute:  'saved-list',
        backSlices: { backRoute },
      }),
      (row) => {
        row.remove();
        if (!rows.querySelector('.saved-row')) careerSection.remove();
        checkEmpty();
      }
    )));

    careerSection.appendChild(rows);
    el.appendChild(careerSection);
  }

  return el;
}
