/* Saved items screen — courses + job roles + campus map */

import { state, unsaveItem } from '../state.js';
import { go } from '../router.js';

const CLOSE_SVG = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
</svg>`;

const REMOVE_SVG = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
</svg>`;

function makeRow(item, onTap) {
  const row = document.createElement('div');
  row.className = 'saved-row';

  const titleBtn = document.createElement('button');
  titleBtn.className = 'saved-row-title';
  titleBtn.textContent = item.title;
  titleBtn.addEventListener('click', onTap);

  const removeBtn = document.createElement('button');
  removeBtn.className = 'saved-row-remove';
  removeBtn.setAttribute('aria-label', `Remove ${item.title}`);
  removeBtn.innerHTML = REMOVE_SVG;
  removeBtn.addEventListener('click', () => {
    unsaveItem(item.id, item.type);
    row.remove();
    const badge = document.getElementById('saved-count');
    if (badge) badge.textContent = state.saved.items.length;
  });

  row.appendChild(titleBtn);
  row.appendChild(removeBtn);
  return row;
}

function makeSection(label) {
  const wrap = document.createElement('div');
  wrap.className = 'saved-section';
  const h = document.createElement('h3');
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
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 18,
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
          { radius: 8, color: '#0d9488', fillColor: '#0d9488', fillOpacity: 0.9, weight: 2 }
        ).addTo(map).bindPopup('Your location');
        markers.push(userMarker);
        map.fitBounds(L.featureGroup(markers).getBounds().pad(0.2));
      }, () => {});
    }
  }, 0);

  return wrap;
}

export function SavedListView() {
  const el = document.createElement('div');
  el.className = 'view view-saved-list';

  // Header
  const header = document.createElement('div');
  header.className = 'saved-list-header';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'saved-list-close';
  closeBtn.setAttribute('aria-label', 'Close saved items');
  closeBtn.innerHTML = CLOSE_SVG;
  closeBtn.addEventListener('click', () => go('chat-first'));

  const titleEl = document.createElement('span');
  titleEl.className = 'saved-list-heading';
  titleEl.textContent = 'Saved';

  header.appendChild(closeBtn);
  header.appendChild(titleEl);
  el.appendChild(header);

  const courses = state.saved.items.filter(i => i.type === 'course');
  const careers = state.saved.items.filter(i => i.type === 'job');

  if (courses.length === 0 && careers.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'saved-list-empty';
    empty.textContent = 'Nothing saved yet — tap the bookmark on any course or job role to save it here.';
    el.appendChild(empty);
    return el;
  }

  // Courses
  if (courses.length > 0) {
    const section = makeSection('Courses');
    const rows = document.createElement('div');
    rows.className = 'saved-rows';
    courses.forEach(item => rows.appendChild(makeRow(item, () => go('course-detail', {
      courseId:   item.id,
      courseTitle: item.title,
      backRoute:  'saved-list',
    }))));
    section.appendChild(rows);
    el.appendChild(section);
    el.appendChild(buildMap(courses.map(i => i.id)));
  }

  // Job roles
  if (careers.length > 0) {
    const section = makeSection('Job roles');
    const rows = document.createElement('div');
    rows.className = 'saved-rows';
    careers.forEach(item => rows.appendChild(makeRow(item, () => go('job-detail', {
      jobId:     item.id,
      jobTitle:  item.title,
      backRoute: 'saved-list',
      backSlices: {},
    }))));
    section.appendChild(rows);
    el.appendChild(section);
  }

  return el;
}
