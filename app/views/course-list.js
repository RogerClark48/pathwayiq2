/* Course list screen — chat pivot result */

import { state } from '../state.js';
import { go }    from '../router.js';
import { logEvent } from '../analytics.js';

export function CourseListView() {
  const data    = state.courseList;
  const courses = data?.courses || [];

  const el = document.createElement('div');
  el.className = 'view view-course-list';

  // Header with close button
  const header = document.createElement('div');
  header.className = 'course-list-header';
  const closeBtn = document.createElement('button');
  closeBtn.className = 'course-list-close';
  closeBtn.setAttribute('aria-label', 'Back to chat');
  closeBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>`;
  closeBtn.addEventListener('click', () => go('chat-first'));
  header.appendChild(closeBtn);
  el.appendChild(header);

  // Intro text from Haiku
  const introEl = document.createElement('p');
  introEl.className = 'course-list-intro';
  introEl.textContent = data?.intro_text || '';
  el.appendChild(introEl);

  if (courses.length === 0) {
    const emptyEl = document.createElement('div');
    emptyEl.className = 'course-list-empty';
    const backBtn = document.createElement('button');
    backBtn.className = 'btn-back-chat';
    backBtn.textContent = 'Back to chat';
    backBtn.addEventListener('click', () => go('chat-first'));
    emptyEl.appendChild(backBtn);
    el.appendChild(emptyEl);
    return el;
  }

  // Course rows
  const listEl = document.createElement('div');
  listEl.className = 'course-list-rows';

  courses.forEach(course => {
    const row = document.createElement('button');
    row.className = 'course-row';
    row.type = 'button';

    const titleEl = document.createElement('div');
    titleEl.className = 'course-row-title';
    titleEl.textContent = course.course_title;

    const previewEl = document.createElement('div');
    previewEl.className = 'course-row-preview';
    previewEl.textContent = course.preview_text;

    row.appendChild(titleEl);
    row.appendChild(previewEl);
    logEvent('course_impression', 'course', course.course_id, course.course_title);
    row.addEventListener('click', () => go('course-detail', {
      courseId:    course.course_id,
      courseTitle: course.course_title,
      backRoute:   'course-list',
    }));
    listEl.appendChild(row);
  });

  el.appendChild(listEl);
  return el;
}
