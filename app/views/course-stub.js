/* Course detail stub — placeholder until course detail screen is built */

import { go } from '../router.js';

export function CourseStubView(slices = {}) {
  const course = slices.course || {};

  const el = document.createElement('div');
  el.className = 'view view-course-stub';

  const backBtn = document.createElement('button');
  backBtn.className = 'course-stub-back';
  backBtn.setAttribute('aria-label', 'Back to course list');
  backBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
    aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg> Back`;
  backBtn.addEventListener('click', () => go('course-list'));
  el.appendChild(backBtn);

  const titleEl = document.createElement('h2');
  titleEl.className = 'course-stub-title';
  titleEl.textContent = course.course_title || 'Course';
  el.appendChild(titleEl);

  const comingEl = document.createElement('p');
  comingEl.className = 'course-stub-coming';
  comingEl.textContent = 'Course detail coming soon.';
  el.appendChild(comingEl);

  return el;
}
