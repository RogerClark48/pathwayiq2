/* FutureFinder v3 — entry point */

import { register, go } from './router.js';
import { loadWelcomeData } from './api.js';
import { logEvent } from './analytics.js';
import { WelcomeView }        from './views/welcome.js';
import { EditView }           from './views/edit.js';
import { StartView }          from './views/start.js';
import { StartChatView }      from './views/start-chat.js';
import { CourseListView }     from './views/course-list.js';
import { CourseCarouselView } from './views/course-carousel.js';
import { CourseDetailView }   from './views/course-detail.js';
import { JobDetailView }      from './views/job-detail.js';
import { SavedListView }      from './views/saved-list.js';
import { PathwayMapView }     from './views/pathway-map.js';
import { CareerView }         from './views/career-view.js';

register('welcome',       () => WelcomeView());
register('edit',          () => EditView());
register('start',         () => StartView());
register('chat-first',    (slices) => StartChatView(slices));
register('course-list',   (slices) => CourseCarouselView(slices));
register('course-detail', (slices) => CourseDetailView(slices));
register('job-detail',    (slices) => JobDetailView(slices));
register('saved-list',    (slices) => SavedListView(slices));
register('pathway-map',   (slices) => PathwayMapView(slices));
register('career-view',   (slices) => CareerView(slices));

document.addEventListener('DOMContentLoaded', () => {
  loadWelcomeData(); // kick fetch immediately — welcome view reuses the same promise
  logEvent('session_start');

  // Pre-fill from ?q= parameter (GMIoT embed stub doorway) — skip landing, go direct to chat.
  const qParam = new URLSearchParams(window.location.search).get('q') || '';
  const prefill = qParam.slice(0, 300).trim();
  if (prefill) {
    go('chat-first', { prefill });
  } else {
    go('welcome');
  }
});
