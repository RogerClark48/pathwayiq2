/* Course list — horizontal carousel over map backdrop */

import { go }         from '../router.js';
import { logEvent }   from '../analytics.js';
import { SSA_LABELS } from '../ssa.js';

// ── Subject colours (mid-to-deep, white-legible, hue-spread) ────────────────
const SSA_COLOURS = {
  '1':  '#1D7A6E',  // Health & Care — teal-green
  '2':  '#2A6496',  // Science & Maths
  '3':  '#4E7038',  // Agriculture & Land
  '4':  '#3A5A78',  // Engineering & Manufacturing — deep slate-blue
  '5':  '#B5651D',  // Construction & Built Environment — amber-brown
  '6':  '#4B3F8F',  // Computing & IT — indigo-violet
  '7':  '#7A5C3A',  // Retail & Commerce
  '8':  '#4A7A5A',  // Hospitality & Tourism
  '9':  '#B23A6E',  // Creative & Media — magenta-rose
  '10': '#5A5A9A',  // History & Philosophy
  '11': '#4A7A8A',  // Social Sciences
  '12': '#7A3A8A',  // Languages & Culture
  '13': '#3A6A8A',  // Education & Training
  '14': '#5A6E5A',  // Life & Work
  '15': '#6E6A5E',  // Business & Law — warm grey-taupe
  '99': '#4C8B3F',  // Sustainability — leaf-green
};

const DEFAULT_COLOUR = '#1A237E';

function subjectColour(ssa_code) {
  return SSA_COLOURS[String(ssa_code ?? '')] ?? DEFAULT_COLOUR;
}

// ── Mode display normalisation ───────────────────────────────────────────────
const MODE_LABELS = { FT: 'Full-time', PT: 'Part-time', 'FT/PT': 'Full or Part-time' };

function modeLabel(mode) {
  if (!mode) return null;
  return MODE_LABELS[mode] || mode;
}

// ── Title → {title, specialism} ─────────────────────────────────────────────
// Extracts parenthetical specialism: "HNC Foo (Bar Technician)" → {title:"HNC Foo", specialism:"Bar Technician"}
function splitTitle(raw) {
  const m = raw.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  if (m) return { title: m[1].trim(), specialism: m[2].trim() };
  return { title: raw, specialism: null };
}

// ── Leaflet map pin ──────────────────────────────────────────────────────────
function makePin(colour) {
  const svg = `<svg width="20" height="28" viewBox="0 0 20 28" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 0C4.48 0 0 4.48 0 10C0 17.5 10 28 10 28C10 28 20 17.5 20 10C20 4.48 15.52 0 10 0Z" fill="${colour}" stroke="white" stroke-width="1.5"/>
    <circle cx="10" cy="10" r="3.5" fill="white" opacity="0.9"/>
  </svg>`;
  return L.divIcon({
    html:       svg,
    iconSize:   [20, 28],
    iconAnchor: [10, 28],
    className:  '',
  });
}

// ── Watermark SVG (building silhouette, placeholder) ─────────────────────────
const WATERMARK_SVG = `<svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" class="cc-wm-icon">
  <rect x="10" y="36" width="60" height="36" rx="2" fill="currentColor"/>
  <polygon points="5,38 40,10 75,38" fill="currentColor"/>
  <rect x="30" y="50" width="12" height="22" rx="1" fill="white" opacity="0.25"/>
  <rect x="47" y="50" width="10" height="14" rx="1" fill="white" opacity="0.25"/>
  <rect x="18" y="50" width="10" height="14" rx="1" fill="white" opacity="0.25"/>
</svg>`;

// ── Module-level map reference for cleanup on re-entry ───────────────────────
let _activeMap = null;

// ── Main view ────────────────────────────────────────────────────────────────
export function CourseCarouselView(slices = {}) {
  // Destroy previous Leaflet instance if view is re-mounted
  if (_activeMap) {
    try { _activeMap.remove(); } catch (_) {}
    _activeMap = null;
  }

  const data    = slices.courseList || {};
  const courses = data.courses || [];

  const el = document.createElement('div');
  el.className = 'view view-course-carousel';

  // ── Empty state ────────────────────────────────────────────────────────────
  if (courses.length === 0) {
    el.innerHTML = `
      <div class="carousel-empty">
        <p class="carousel-empty-text">No courses found.</p>
        <button class="btn-back-chat">Back to chat</button>
      </div>`;
    el.querySelector('.btn-back-chat').addEventListener('click', () => go('chat-first'));
    return el;
  }

  // ── Header label ───────────────────────────────────────────────────────────
  const ssaCodes = [...new Set(courses.map(c => String(c.ssa_code ?? '')).filter(Boolean))];
  const headerLabel = ssaCodes.length === 1
    ? `${SSA_LABELS[ssaCodes[0]] || 'Courses'} · ${courses.length}`
    : `Courses · ${courses.length}`;

  // ── DOM skeleton ───────────────────────────────────────────────────────────
  el.innerHTML = `
    <div class="carousel-stage">
      <div class="carousel-map-bg" id="cc-map"></div>
      <div class="carousel-overlay">
        <div class="carousel-header-bar">
          <button class="carousel-back-btn" aria-label="Back to chat">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="15 18 9 12 15 6"/>
            </svg>
            <span>${headerLabel}</span>
          </button>
        </div>
        <div class="carousel-cards-zone">
          <div class="carousel-track" role="listbox" aria-label="Courses"></div>
          <div class="carousel-dots" aria-hidden="true"></div>
        </div>
      </div>
    </div>`;

  el.querySelector('.carousel-back-btn').addEventListener('click', () => go('chat-first'));

  const track  = el.querySelector('.carousel-track');
  const dotsEl = el.querySelector('.carousel-dots');

  // ── Build cards ────────────────────────────────────────────────────────────
  courses.forEach((course, idx) => {
    const { title, specialism } = splitTitle(course.course_title);
    const colour  = subjectColour(course.ssa_code);
    const mode    = modeLabel(course.mode);
    const lvlText = course.level != null ? `L${course.level}` : null;
    const hook    = (course.preview_text || '').slice(0, 200);

    const chipsHtml = [
      course.qual_type
        ? `<span class="cc-chip">${course.qual_type}</span>`
        : '',
      mode
        ? `<span class="cc-chip cc-chip--mode">
             <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
               <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
             </svg>${mode}</span>`
        : '',
    ].filter(Boolean).join('');

    const card = document.createElement('div');
    card.className = `carousel-card${idx === 0 ? ' is-active' : ''}`;
    card.setAttribute('role', 'option');
    card.setAttribute('aria-selected', idx === 0 ? 'true' : 'false');
    card.tabIndex = idx === 0 ? 0 : -1;

    card.innerHTML = `
      <div class="cc-header" style="background:${colour}">
        <div class="cc-watermark">${WATERMARK_SVG}</div>
        ${lvlText
          ? `<span class="cc-level-badge">${lvlText}</span>`
          : `<span class="cc-level-badge cc-level-badge--empty" aria-hidden="true"></span>`}
        <div class="cc-title-block">
          <h2 class="cc-title">${title}</h2>
          ${specialism ? `<p class="cc-specialism">${specialism}</p>` : ''}
        </div>
      </div>
      <div class="cc-body">
        <div class="cc-chips">${chipsHtml}</div>
        <p class="cc-hook">${hook}</p>
        <div class="cc-footer">
          <span class="cc-provider">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
            </svg>
            ${course.provider_name || ''}
          </span>
          ${course.duration ? `<span class="cc-duration">${course.duration}</span>` : ''}
        </div>
      </div>`;

    logEvent('course_impression', 'course', course.course_id, course.course_title);

    card.addEventListener('click', () => {
      // Suppress navigation if the pointer just finished a drag
      if (_dragDelta > 6) { _dragDelta = 0; return; }
      if (card.classList.contains('is-active')) {
        logEvent('course_tap', 'course', course.course_id, course.course_title);
        go('course-detail', {
          courseId:    course.course_id,
          courseTitle: course.course_title,
          backRoute:   'course-list',
        });
      } else {
        const targetX = card.offsetLeft + card.offsetWidth / 2 - track.clientWidth / 2;
        track.scrollTo({ left: Math.max(0, targetX), behavior: 'smooth' });
      }
    });

    track.appendChild(card);

    // Dot indicator
    const dot = document.createElement('span');
    dot.className = `carousel-dot${idx === 0 ? ' is-active' : ''}`;
    dotsEl.appendChild(dot);
  });

  // ── Active-card tracking ───────────────────────────────────────────────────
  const cardEls = Array.from(track.querySelectorAll('.carousel-card'));
  const dotEls  = Array.from(dotsEl.querySelectorAll('.carousel-dot'));
  let   currentIdx = 0;
  let   currentPin = null;

  function setActivePin(course) {
    if (!_activeMap || course.lat == null || course.lng == null) return;
    setTimeout(() => {
      if (currentPin) currentPin.remove();
      currentPin = L.marker([course.lat, course.lng], {
        icon: makePin(subjectColour(course.ssa_code)),
      }).addTo(_activeMap);
      _activeMap.panTo([course.lat, course.lng], {
        animate:       true,
        duration:      0.65,
        easeLinearity: 0.5,
      });
    }, 90);
  }

  function updateActive(idx, force = false) {
    if (idx === currentIdx && !force) return;
    currentIdx = idx;
    cardEls.forEach((c, i) => {
      const active = i === idx;
      c.classList.toggle('is-active', active);
      c.setAttribute('aria-selected', active ? 'true' : 'false');
      c.tabIndex = active ? 0 : -1;
    });
    dotEls.forEach((d, i) => d.classList.toggle('is-active', i === idx));
    setActivePin(courses[idx]);
  }

  // ── Scroll listener — updates active card after native or JS-driven scroll ─
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

  // ── Mouse drag — lets desktop users drag the carousel ─────────────────────
  // (Touch uses native CSS scroll-snap; this only activates for mouse pointers.)
  let _dragStart  = null;
  let _dragLeft   = 0;
  let _dragDelta  = 0;

  track.addEventListener('pointerdown', e => {
    if (e.pointerType !== 'mouse') return;
    _dragStart = e.clientX;
    _dragLeft  = track.scrollLeft;
    _dragDelta = 0;
    track.setPointerCapture(e.pointerId);
    track.style.cursor = 'grabbing';
  });

  track.addEventListener('pointermove', e => {
    if (_dragStart === null || e.pointerType !== 'mouse') return;
    const dx = _dragStart - e.clientX;
    _dragDelta = Math.abs(dx);
    track.scrollLeft = _dragLeft + dx;
  });

  function endDrag() {
    if (_dragStart === null) return;
    _dragStart = null;
    track.style.cursor = '';

    // Smooth-snap to nearest card (scroll-snap alone doesn't animate on mouse release)
    const cx = track.scrollLeft + track.clientWidth / 2;
    let nearest = 0, minDist = Infinity;
    cardEls.forEach((c, i) => {
      const d = Math.abs(c.offsetLeft + c.offsetWidth / 2 - cx);
      if (d < minDist) { minDist = d; nearest = i; }
    });
    const target = cardEls[nearest];
    const targetX = target.offsetLeft + target.offsetWidth / 2 - track.clientWidth / 2;
    track.scrollTo({ left: Math.max(0, targetX), behavior: 'smooth' });
  }

  track.addEventListener('pointerup',     endDrag);
  track.addEventListener('pointercancel', endDrag);

  // ── Leaflet map initialisation ─────────────────────────────────────────────
  requestAnimationFrame(() => {
    const mapEl = el.querySelector('#cc-map');
    if (!mapEl || typeof L === 'undefined') return;

    const first = courses[0];
    const initLat = first.lat ?? 53.5;
    const initLng = first.lng ?? -2.3;

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
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        maxZoom: 19,
      }
    ).addTo(_activeMap);

    if (first.lat != null && first.lng != null) {
      currentPin = L.marker([first.lat, first.lng], {
        icon: makePin(subjectColour(first.ssa_code)),
      }).addTo(_activeMap);
    }

    setTimeout(() => _activeMap?.invalidateSize(), 60);
  });

  return el;
}
