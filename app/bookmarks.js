/* Shared bookmark button — used on all card and header surfaces */
import { isSaved, toggleSave } from './state.js';

const BOOKMARK_SVG = `<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M9 4h6a2 2 0 0 1 2 2v14l-5-3-5 3V6a2 2 0 0 1 2-2"/>
</svg>`;

/**
 * Returns a <button class="ff-bookmark"> that toggles save and reflects isSaved.
 * item = { id, type, title, ssa?, subtitle? }
 * e.stopPropagation() prevents the card-tap navigation from firing.
 */
export function bookmarkButton(item) {
  const b = document.createElement('button');
  b.className = 'ff-bookmark';
  b.innerHTML = BOOKMARK_SVG;

  const paint = () => {
    const on = isSaved(item.id, item.type);
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', String(on));
    b.setAttribute('aria-label', on ? 'Remove from saved' : 'Save');
  };
  paint();

  b.addEventListener('click', e => {
    e.stopPropagation();
    toggleSave(item);
    paint();
  });

  return b;
}
