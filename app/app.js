/* FutureFinder v3 — entry point */

import { state } from './state.js';
import { register, go } from './router.js';
import { loadWelcomeData } from './api.js';
import { WelcomeView }    from './views/welcome.js';
import { EditView }       from './views/edit.js';
import { StartView }      from './views/start.js';
import { StartChatView }  from './views/start-chat.js';
import { CourseListView }   from './views/course-list.js';
import { CourseDetailView } from './views/course-detail.js';

register('welcome',       () => WelcomeView());
register('edit',          () => EditView());
register('start',         () => StartView());
register('chat-first',    () => StartChatView());
register('course-list',   () => CourseListView());
register('course-detail', (slices) => CourseDetailView(slices));

function updateBadge() {
  const badge = document.getElementById('saved-count');
  badge.textContent = state.saved.items.length;
}

document.addEventListener('DOMContentLoaded', () => {
  loadWelcomeData(); // kick fetch immediately — welcome view reuses the same promise
  updateBadge();
  go('chat-first');
});
