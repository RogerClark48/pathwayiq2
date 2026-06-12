/* Course list — Horizon carousel over map backdrop */

import { go }        from '../router.js';
import { logEvent }  from '../analytics.js';
import { subject, subjectHex, subjectIconSvg } from '../subjects.js';
import { bookmarkButton } from '../bookmarks.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

const MODE_LABELS = { FT: 'Full-time', PT: 'Part-time', 'FT/PT': 'Full or Part-time' };
function modeLabel(mode) { return mode ? (MODE_LABELS[mode] || mode) : null; }

function splitTitle(raw) {
  const m = (raw || '').match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  return m ? { title: m[1].trim(), specialism: m[2].trim() } : { title: raw || '', specialism: null };
}

function haversineKm(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function toMiles(km) { return km * 0.621371; }

function getLocation(timeout = 2000) {
  return new Promise(resolve => {
    if (!navigator.geolocation) { resolve(null); return; }
    const timer = setTimeout(() => resolve(null), timeout);
    navigator.geolocation.getCurrentPosition(
      pos => { clearTimeout(timer); resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }); },
      ()   => { clearTimeout(timer); resolve(null); }
    );
  });
}

// ── Branded Leaflet pin ───────────────────────────────────────────────────────
// subjectHex() gives raw hex — required because fill is injected into an off-DOM SVG string.
// The icon stroke (#fff) is set via CSS on a live element, so that's fine.

function makePin(ssa) {
  const hex  = subjectHex(ssa);
  const icon = subjectIconSvg(ssa);
  const html = `<div class="ffpin" style="--c:${hex}"><span class="drop">${icon}</span></div>`;
  return L.divIcon({ html, iconSize: [30, 38], iconAnchor: [15, 34], className: 'ffpin-host' });
}

// ── Module-level map reference (cleaned up on re-entry) ───────────────────────
let _activeMap = null;

// ── Main view ─────────────────────────────────────────────────────────────────
export function CourseCarouselView(slices = {}) {
  if (_activeMap) {
    try { _activeMap.remove(); } catch (_) {}
    _activeMap = null;
  }

  const data      = slices.courseList || {};
  const courses   = data.courses || [];
  const backRoute = slices.backRoute || 'chat-first';

  const el = document.createElement('div');
  el.className = 'view view-course-carousel';

  if (courses.length === 0) {
    el.innerHTML = `
      <div class="carousel-empty">
        <p class="carousel-empty-text">No courses found.</p>
        <button class="btn-back-chat">← Back</button>
      </div>`;
    el.querySelector('.btn-back-chat').addEventListener('click', () => go(backRoute));
    return el;
  }

  // Brief skeleton while geolocation resolves (denied = instant; granted = fast)
  el.innerHTML = `
    <div class="carousel-stage">
      <div class="carousel-map-bg"></div>
      <div class="ccx-locating">
        <span>Locating…</span>
      </div>
    </div>`;

  getLocation(2000).then(userLoc => {
    if (!el.isConnected) return;
    renderFull(el, courses, userLoc, backRoute);
  });

  return el;
}

// ── Shared: distance sort ─────────────────────────────────────────────────────
function sortByDistance(courses, userLoc) {
  const sorted = courses.map(c => ({ ...c }));
  if (userLoc) {
    sorted.forEach(c => {
      c._distMi = (c.lat != null && c.lng != null)
        ? toMiles(haversineKm(userLoc.lat, userLoc.lng, c.lat, c.lng))
        : null;
    });
    sorted.sort((a, b) => (a._distMi ?? Infinity) - (b._distMi ?? Infinity));
  }
  return sorted;
}

// ── Shared: filter ────────────────────────────────────────────────────────────
function applyFilters(sorted, mf, lf) {
  const f = sorted.filter(c => {
    if (mf && modeLabel(c.mode) !== mf) return false;
    if (lf && c.level !== lf) return false;
    return true;
  });
  return f.length ? f : sorted;
}

// ── Dispatcher — decides deck vs split and wires live reflow ─────────────────
function renderFull(el, courses, userLoc, backRoute) {
  const sorted = sortByDistance(courses, userLoc);

  function doLayout() {
    if (_activeMap) { try { _activeMap.remove(); } catch (_) {} _activeMap = null; }
    el.innerHTML = '';
    if (window.innerWidth >= 1200) {
      renderSplit(el, sorted, userLoc, backRoute);
    } else {
      renderDeck(el, sorted, userLoc, backRoute);
    }
  }

  doLayout();

  // Live reflow across the 1200px boundary
  const mq = window.matchMedia('(min-width: 1200px)');
  function onBreakpoint() {
    if (!el.isConnected) { mq.removeEventListener('change', onBreakpoint); return; }
    doLayout();
  }
  mq.addEventListener('change', onBreakpoint);
}

// ── Deck layout (mobile + 768–1199px) ────────────────────────────────────────
function renderDeck(el, sorted, userLoc, backRoute) {
  const hasLoc = !!userLoc;
  const modes  = [...new Set(sorted.map(c => modeLabel(c.mode)).filter(Boolean))];
  const levels = [...new Set(sorted.map(c => c.level).filter(v => v != null))].sort((a, b) => a - b);
  const showRefine = modes.length > 1 || levels.length > 1;

  function subText(count) {
    if (!hasLoc) return `${count} found`;
    return count === sorted.length ? 'Near you · nearest first' : `${count} of ${sorted.length} · nearest first`;
  }

  const modeChipsHtml  = modes.map(m => `<button class="ccx-chip" data-mode="${m}">${m}</button>`).join('');
  const dividerHtml    = modes.length && levels.length ? `<span class="ccx-divider"></span>` : '';
  const levelChipsHtml = levels.map(l => `<button class="ccx-chip" data-level="${l}">Level ${l}</button>`).join('');

  el.innerHTML = `
    <div class="carousel-stage">
      <div class="carousel-map-bg" id="cc-map"></div>
      <div class="carousel-overlay">
        <div class="ccx-head">
          <button class="ccx-back" aria-label="Back">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="15 18 9 12 15 6"/>
            </svg>
          </button>
          <div class="ccx-title-block">
            <span class="ccx-count"><b>${sorted.length}</b> courses</span>
            <span class="ccx-sub" id="ccx-sub">${subText(sorted.length)}</span>
          </div>
          <button class="ccx-finn" aria-label="Back to Finn">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
          </button>
        </div>
        ${showRefine ? `<div class="ccx-refine">${modeChipsHtml}${dividerHtml}${levelChipsHtml}</div>` : ''}
        <div class="carousel-cards-zone">
          <div class="carousel-track" id="cc-track" role="listbox" aria-label="Courses"></div>
          <div class="carousel-dots" id="cc-dots" aria-hidden="true"></div>
        </div>
      </div>
    </div>`;

  el.querySelector('.ccx-back').addEventListener('click', () => go(backRoute));
  el.querySelector('.ccx-finn').addEventListener('click', () => go('chat-first'));

  const track  = el.querySelector('#cc-track');
  const dotsEl = el.querySelector('#cc-dots');
  const subEl  = el.querySelector('#ccx-sub');

  // ── State shared across buildCards calls ──────────────────────────────────
  let cardEls      = [];
  let dotEls       = [];
  let currentIdx   = 0;
  let activeCourses = sorted;
  let currentPin   = null;

  let _scrollAnimId = null;
  let _dragStart    = null;
  let _dragLeft     = 0;
  let _dragDelta    = 0;

  // ── Smooth scroll ─────────────────────────────────────────────────────────
  function smoothScrollTo(targetX, duration = 300) {
    if (_scrollAnimId) { cancelAnimationFrame(_scrollAnimId); _scrollAnimId = null; }
    const startX = track.scrollLeft;
    const delta  = Math.max(0, targetX) - startX;
    if (Math.abs(delta) < 1) return;
    const t0 = performance.now();
    function step(t) {
      const p = Math.min((t - t0) / duration, 1);
      const e = 1 - (1 - p) ** 3;
      track.scrollLeft = startX + delta * e;
      _scrollAnimId = p < 1 ? requestAnimationFrame(step) : null;
    }
    _scrollAnimId = requestAnimationFrame(step);
  }

  // ── Single active pin ─────────────────────────────────────────────────────
  function setActivePin(course) {
    if (!_activeMap || !course || course.lat == null || course.lng == null) return;
    setTimeout(() => {
      if (currentPin) currentPin.remove();
      currentPin = L.marker([course.lat, course.lng], { icon: makePin(course.ssa_code) }).addTo(_activeMap);
      _activeMap.panTo([course.lat, course.lng], { animate: true, duration: 0.65, easeLinearity: 0.5 });
    }, 90);
  }

  // ── Active card tracking ──────────────────────────────────────────────────
  function updateActive(idx) {
    if (idx === currentIdx && cardEls[idx]?.classList.contains('is-active')) return;
    currentIdx = idx;
    cardEls.forEach((c, i) => {
      const on = i === idx;
      c.classList.toggle('is-active', on);
      c.setAttribute('aria-selected', on ? 'true' : 'false');
      c.tabIndex = on ? 0 : -1;
    });
    dotEls.forEach((d, i) => d.classList.toggle('is-active', i === idx));
    setActivePin(activeCourses[idx]);
  }

  // ── Card builder ──────────────────────────────────────────────────────────
  function buildCards(courseSet) {
    activeCourses = courseSet;
    track.innerHTML = '';
    dotsEl.innerHTML = '';
    cardEls = [];
    dotEls  = [];
    currentIdx = 0;

    courseSet.forEach((course, idx) => {
      const s        = subject(course.ssa_code);
      const raw      = course.course_title || '';
      const { title, specialism } = splitTitle(raw);
      const lvlQual  = course.qual_type || (course.level != null ? `L${course.level}` : '');
      const provider = [course.provider_name, course.campus_name].filter(Boolean).join(' · ');
      const hook     = (course.preview_text || '').slice(0, 160);
      const distHtml = (hasLoc && course._distMi != null)
        ? `<span class="dist">${course._distMi.toFixed(1)} mi</span>` : '';

      const chipsHtml = [
        course.qual_type       ? `<span>${course.qual_type}</span>`       : '',
        modeLabel(course.mode) ? `<span>${modeLabel(course.mode)}</span>` : '',
        distHtml,
      ].filter(Boolean).join('');

      // Outer wrapper: drives scroll-snap and inactive dimming
      const wrap = document.createElement('div');
      wrap.className = `carousel-card${idx === 0 ? ' is-active' : ''}`;
      wrap.setAttribute('role', 'option');
      wrap.setAttribute('aria-selected', idx === 0 ? 'true' : 'false');
      wrap.tabIndex = idx === 0 ? 0 : -1;

      // Inner card: shared .ck.ck-course identity (matches chat compact card)
      const card = document.createElement('div');
      card.className = 'ck ck-course';
      card.style.setProperty('--sub', s.colour);
      card.innerHTML = `
        <div class="cc-top">
          <span class="wm">${subjectIconSvg(course.ssa_code)}</span>
          ${lvlQual ? `<span class="lvl">${lvlQual}</span>` : ''}
          <div class="cc-tt">
            <b>${title}</b>
            <span>${provider || specialism || ''}</span>
          </div>
        </div>
        <div class="cc-bd">
          ${chipsHtml ? `<div class="cc-chips">${chipsHtml}</div>` : ''}
          ${hook       ? `<p class="cc-hook">${hook}</p>`          : ''}
        </div>`;

      // Bookmark button in card body — stopPropagation prevents card-tap navigation
      card.querySelector('.cc-bd').appendChild(bookmarkButton({
        id:       course.course_id,
        type:     'course',
        title:    raw,
        ssa:      course.ssa_code,
        subtitle: [course.qual_type, provider].filter(Boolean).join(' · '),
      }));

      wrap.appendChild(card);
      track.appendChild(wrap);
      cardEls.push(wrap);

      logEvent('course_impression', 'course', course.course_id, raw);

      wrap.addEventListener('click', () => {
        if (_dragDelta > 6) { _dragDelta = 0; return; }
        if (wrap.classList.contains('is-active')) {
          logEvent('course_tap', 'course', course.course_id, raw);
          go('course-detail', { courseId: course.course_id, courseTitle: raw, backRoute: 'course-list' });
        } else {
          smoothScrollTo(wrap.offsetLeft + wrap.offsetWidth / 2 - track.clientWidth / 2);
        }
      });

      const dot = document.createElement('span');
      dot.className = `carousel-dot${idx === 0 ? ' is-active' : ''}`;
      dotsEl.appendChild(dot);
      dotEls.push(dot);
    });

    setActivePin(courseSet[0]);
  }

  // ── Refine ────────────────────────────────────────────────────────────────
  let modeFilter  = null;
  let levelFilter = null;

  el.querySelectorAll('.ccx-chip[data-mode]').forEach(btn => {
    btn.addEventListener('click', () => {
      modeFilter = modeFilter === btn.dataset.mode ? null : btn.dataset.mode;
      el.querySelectorAll('.ccx-chip[data-mode]').forEach(b =>
        b.classList.toggle('on', b.dataset.mode === modeFilter));
      applyRefine();
    });
  });

  el.querySelectorAll('.ccx-chip[data-level]').forEach(btn => {
    btn.addEventListener('click', () => {
      const l = Number(btn.dataset.level);
      levelFilter = levelFilter === l ? null : l;
      el.querySelectorAll('.ccx-chip[data-level]').forEach(b =>
        b.classList.toggle('on', Number(b.dataset.level) === levelFilter));
      applyRefine();
    });
  });

  function applyRefine() {
    const set = applyFilters(sorted, modeFilter, levelFilter);
    if (subEl) subEl.textContent = subText(set.length);
    buildCards(set);
    track.scrollLeft = 0;
  }

  // ── Scroll listener ───────────────────────────────────────────────────────
  let scrollRaf = null;
  track.addEventListener('scroll', () => {
    if (scrollRaf) cancelAnimationFrame(scrollRaf);
    scrollRaf = requestAnimationFrame(() => {
      const cx = track.scrollLeft + track.clientWidth / 2;
      let nearest = 0, minDist = Infinity;
      cardEls.forEach((c, i) => {
        const d = Math.abs(c.offsetLeft + c.offsetWidth / 2 - cx);
        if (d < minDist) { minDist = d; nearest = i; }
      });
      updateActive(nearest);
    });
  }, { passive: true });

  // ── Mouse drag ────────────────────────────────────────────────────────────
  track.addEventListener('pointerdown', e => {
    if (e.pointerType !== 'mouse') return;
    if (_scrollAnimId) { cancelAnimationFrame(_scrollAnimId); _scrollAnimId = null; }
    _dragStart = e.clientX;
    _dragLeft  = track.scrollLeft;
    _dragDelta = 0;
    track.style.cursor = 'grabbing';
  });

  track.addEventListener('pointermove', e => {
    if (_dragStart === null || e.pointerType !== 'mouse') return;
    _dragDelta = Math.abs(_dragStart - e.clientX);
    track.scrollLeft = _dragLeft + (_dragStart - e.clientX);
  });

  function endDrag() {
    if (_dragStart === null) return;
    _dragStart = null;
    track.style.cursor = '';
    const cx = track.scrollLeft + track.clientWidth / 2;
    let nearest = 0, minDist = Infinity;
    cardEls.forEach((c, i) => {
      const d = Math.abs(c.offsetLeft + c.offsetWidth / 2 - cx);
      if (d < minDist) { minDist = d; nearest = i; }
    });
    smoothScrollTo(cardEls[nearest].offsetLeft + cardEls[nearest].offsetWidth / 2 - track.clientWidth / 2);
  }

  track.addEventListener('pointerup',     endDrag);
  track.addEventListener('pointercancel', endDrag);
  window.addEventListener('pointerup',     endDrag);
  window.addEventListener('pointercancel', endDrag);

  // ── Initial card render ───────────────────────────────────────────────────
  buildCards(sorted);

  // ── Leaflet map ───────────────────────────────────────────────────────────
  requestAnimationFrame(() => {
    const mapEl = el.querySelector('#cc-map');
    if (!mapEl || typeof L === 'undefined') return;

    const first   = sorted[0];
    const initLat = first?.lat ?? 53.5;
    const initLng = first?.lng ?? -2.3;

    _activeMap = L.map(mapEl, {
      center:             [initLat, initLng],
      zoom:               13,
      zoomControl:        false,
      attributionControl: true,
      dragging:           false,
      touchZoom:          false,
      scrollWheelZoom:    false,
      doubleClickZoom:    false,
      boxZoom:            false,
      keyboard:           false,
      tap:                false,
    });

    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
      {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>' +
                     ' &copy; <a href="https://carto.com/attributions">CARTO</a>',
        maxZoom: 19,
      }
    ).addTo(_activeMap);

    // Map is now ready — place the first pin (setActivePin deferred silently during buildCards)
    setActivePin(activeCourses[currentIdx]);
    setTimeout(() => _activeMap?.invalidateSize(), 60);
  });
}

// ── Split layout (≥1200px) — list column + open Leaflet map ──────────────────
function renderSplit(el, sorted, userLoc, backRoute) {
  const hasLoc = !!userLoc;
  const modes  = [...new Set(sorted.map(c => modeLabel(c.mode)).filter(Boolean))];
  const levels = [...new Set(sorted.map(c => c.level).filter(v => v != null))].sort((a, b) => a - b);
  const showRefine = modes.length > 1 || levels.length > 1;

  function subText(count) {
    if (!hasLoc) return `${count} found`;
    return count === sorted.length ? 'Near you · nearest first' : `${count} of ${sorted.length} · nearest first`;
  }

  const modeChipsHtml  = modes.map(m => `<button class="ccx-chip" data-mode="${m}">${m}</button>`).join('');
  const dividerHtml    = modes.length && levels.length ? `<span class="ccx-divider"></span>` : '';
  const levelChipsHtml = levels.map(l => `<button class="ccx-chip" data-level="${l}">Level ${l}</button>`).join('');

  el.innerHTML = `
    <div class="dbx">
      <div class="dbx-list">
        <div class="dbx-lh">
          <div class="ccx-title-block">
            <span class="ccx-count" id="dbx-count"><b>${sorted.length}</b> courses</span>
            <span class="ccx-sub" id="dbx-sub">${subText(sorted.length)}</span>
          </div>
          ${showRefine ? `<div class="ccx-refine">${modeChipsHtml}${dividerHtml}${levelChipsHtml}</div>` : ''}
        </div>
        <div class="dbx-rows" id="dbx-rows"></div>
      </div>
      <div class="dbx-map" id="dbx-map"></div>
    </div>`;

  const rowsEl  = el.querySelector('#dbx-rows');
  const countEl = el.querySelector('#dbx-count');
  const subEl   = el.querySelector('#dbx-sub');

  // ── State ─────────────────────────────────────────────────────────────────
  let activeSet  = sorted;
  let selectedId = null;
  const markerById = new Map(); // courseId → L.Marker
  const cardById   = new Map(); // courseId → card element

  // ── Pin helpers ───────────────────────────────────────────────────────────
  function getPinEl(marker) {
    return marker.getElement()?.querySelector('.ffpin');
  }

  function setPinActive(courseId, on) {
    const marker = markerById.get(courseId);
    if (!marker) return;
    getPinEl(marker)?.classList.toggle('is-active', on);
    marker.setZIndexOffset(on ? 1000 : 0);
  }

  // ── Selection ─────────────────────────────────────────────────────────────
  function selectById(courseId, scrollCard) {
    if (selectedId) {
      cardById.get(selectedId)?.classList.remove('is-selected');
      setPinActive(selectedId, false);
    }
    selectedId = courseId;
    cardById.get(courseId)?.classList.add('is-selected');
    setPinActive(courseId, true);

    const course = activeSet.find(c => c.course_id === courseId);
    if (course?.lat != null && _activeMap) {
      _activeMap.panTo([course.lat, course.lng], { animate: true, duration: 0.5 });
    }

    if (scrollCard) {
      const card = cardById.get(courseId);
      if (card) rowsEl.scrollTop = card.offsetTop - 8;
    }
  }

  // ── Card builder ──────────────────────────────────────────────────────────
  function buildCards(set) {
    rowsEl.innerHTML = '';
    cardById.clear();

    set.forEach(course => {
      const s        = subject(course.ssa_code);
      const raw      = course.course_title || '';
      const { title, specialism } = splitTitle(raw);
      const lvlQual  = course.qual_type || (course.level != null ? `L${course.level}` : '');
      const provider = [course.provider_name, course.campus_name].filter(Boolean).join(' · ');
      const hook     = (course.preview_text || '').slice(0, 160);
      const distHtml = (hasLoc && course._distMi != null)
        ? `<span class="dist">${course._distMi.toFixed(1)} mi</span>` : '';

      const chipsHtml = [
        course.qual_type       ? `<span>${course.qual_type}</span>`       : '',
        modeLabel(course.mode) ? `<span>${modeLabel(course.mode)}</span>` : '',
        distHtml,
      ].filter(Boolean).join('');

      const card = document.createElement('div');
      card.className = 'ck ck-course' + (course.course_id === selectedId ? ' is-selected' : '');
      card.style.setProperty('--sub', s.colour);
      card.dataset.courseId = String(course.course_id);
      card.innerHTML = `
        <div class="cc-top">
          <span class="wm">${subjectIconSvg(course.ssa_code)}</span>
          ${lvlQual ? `<span class="lvl">${lvlQual}</span>` : ''}
          <div class="cc-tt">
            <b>${title}</b>
            <span>${provider || specialism || ''}</span>
          </div>
        </div>
        <div class="cc-bd">
          ${chipsHtml ? `<div class="cc-chips">${chipsHtml}</div>` : ''}
          ${hook       ? `<p class="cc-hook">${hook}</p>`          : ''}
        </div>`;

      card.querySelector('.cc-bd').appendChild(bookmarkButton({
        id:       course.course_id,
        type:     'course',
        title:    raw,
        ssa:      course.ssa_code,
        subtitle: [course.qual_type, provider].filter(Boolean).join(' · '),
      }));

      card.addEventListener('mouseenter', () => {
        if (course.course_id !== selectedId) setPinActive(course.course_id, true);
      });
      card.addEventListener('mouseleave', () => {
        if (course.course_id !== selectedId) setPinActive(course.course_id, false);
      });
      card.addEventListener('click', () => {
        if (course.course_id === selectedId) {
          logEvent('course_tap', 'course', course.course_id, raw);
          go('course-detail', { courseId: course.course_id, courseTitle: raw, backRoute: 'course-list' });
        } else {
          selectById(course.course_id, false);
        }
      });

      rowsEl.appendChild(card);
      cardById.set(course.course_id, card);
      logEvent('course_impression', 'course', course.course_id, raw);
    });
  }

  // ── Pin sync — replaces all markers for a given course set ────────────────
  function syncPins(set) {
    if (!_activeMap) return;
    markerById.forEach(m => m.remove());
    markerById.clear();

    const bounds = [];
    set.forEach(course => {
      if (course.lat == null || course.lng == null) return;
      const marker = L.marker([course.lat, course.lng], { icon: makePin(course.ssa_code) });
      marker.addTo(_activeMap);
      markerById.set(course.course_id, marker);
      bounds.push([course.lat, course.lng]);

      marker.on('click', () => selectById(course.course_id, true));
      marker.on('mouseover', () => {
        if (course.course_id !== selectedId) setPinActive(course.course_id, true);
      });
      marker.on('mouseout', () => {
        if (course.course_id !== selectedId) setPinActive(course.course_id, false);
      });
    });

    if (bounds.length > 1) {
      _activeMap.fitBounds(bounds, { padding: [40, 40] });
    } else if (bounds.length === 1) {
      _activeMap.setView(bounds[0], 13);
    }
  }

  // ── Refine ────────────────────────────────────────────────────────────────
  let modeFilter  = null;
  let levelFilter = null;

  el.querySelectorAll('.ccx-chip[data-mode]').forEach(btn => {
    btn.addEventListener('click', () => {
      modeFilter = modeFilter === btn.dataset.mode ? null : btn.dataset.mode;
      el.querySelectorAll('.ccx-chip[data-mode]').forEach(b =>
        b.classList.toggle('on', b.dataset.mode === modeFilter));
      doRefine();
    });
  });

  el.querySelectorAll('.ccx-chip[data-level]').forEach(btn => {
    btn.addEventListener('click', () => {
      const l = Number(btn.dataset.level);
      levelFilter = levelFilter === l ? null : l;
      el.querySelectorAll('.ccx-chip[data-level]').forEach(b =>
        b.classList.toggle('on', Number(b.dataset.level) === levelFilter));
      doRefine();
    });
  });

  function doRefine() {
    const set = applyFilters(sorted, modeFilter, levelFilter);
    activeSet  = set;
    selectedId = null;
    subEl.textContent = subText(set.length);
    buildCards(set);
    syncPins(set);
  }

  // ── Leaflet map init ──────────────────────────────────────────────────────
  requestAnimationFrame(() => {
    const mapEl = el.querySelector('#dbx-map');
    if (!mapEl || typeof L === 'undefined') return;

    const first = sorted[0];
    _activeMap = L.map(mapEl, {
      center:             [first?.lat ?? 53.5, first?.lng ?? -2.3],
      zoom:               11,
      zoomControl:        true,
      attributionControl: true,
      dragging:           true,
      touchZoom:          true,
      scrollWheelZoom:    true,
      doubleClickZoom:    true,
      boxZoom:            false,
      keyboard:           false,
    });

    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
      {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>' +
                     ' &copy; <a href="https://carto.com/attributions">CARTO</a>',
        maxZoom: 19,
      }
    ).addTo(_activeMap);

    if (userLoc) {
      L.circleMarker([userLoc.lat, userLoc.lng], {
        radius: 7, color: '#fff', fillColor: '#2A6BE8', fillOpacity: 1, weight: 2,
      }).addTo(_activeMap);
    }

    syncPins(activeSet);
    setTimeout(() => _activeMap?.invalidateSize(), 60);
  });

  // ── Initial card render ───────────────────────────────────────────────────
  buildCards(activeSet);
}
