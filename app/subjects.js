/* app/subjects.js — canonical subject metadata, keyed by SSA code.
   Colour + icon only; label lives in ssa.js (SSA_LABELS). */
import { SSA_LABELS } from './ssa.js';

// Lucide-style 24x24 stroke icons (paths only — wrap with subjectIconSvg below).
export const SUBJECT_ICONS = {
  gear:       '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2Z"/><circle cx="12" cy="12" r="3"/>',
  crane:      '<path d="M3.5 21h5M6 21V6M2.6 6h18.4M6 6V2.6l14.4 3.4M6 2.6 3 6M17.4 6v3.1M3 6v1.9h2"/><path d="M16.3 9.1a1.1 1.1 0 0 0 2.2 0"/>',
  code:       '<path d="M9.4 7.8 5.2 12l4.2 4.2M14.6 7.8 18.8 12l-4.2 4.2"/>',
  palette:    '<path d="M12 3.3c-4.9 0-8.8 3.8-8.8 8.7s3.9 8.7 8.8 8.7c.9 0 1.6-.7 1.6-1.6 0-.4-.2-.7-.4-1-.2-.3-.4-.6-.4-1 0-.9.7-1.6 1.6-1.6h1.8c2.6 0 4.6-2 4.6-4.6 0-4.2-3.9-7.6-8.8-7.6Z"/><circle cx="7.6" cy="12.2" r="1.05"/><circle cx="10.4" cy="8.2" r="1.05"/><circle cx="15" cy="8.7" r="1.05"/>',
  heartpulse: '<path d="M20.6 8.5a4.7 4.7 0 0 0-8.6-2.7A4.7 4.7 0 0 0 3.4 8.5c0 4.4 5.4 8 8.6 10.4 3.2-2.4 8.6-6 8.6-10.4Z"/><path d="M6.9 12.1h2.4l1.3-2.4 1.7 4.2 1.3-2.4h2.5"/>',
  leaf:       '<path d="M5 19.4C4 11.5 9.5 5 19.5 4.5c-.5 10-6 15.4-14.5 14.9Z"/><path d="M9 15.4c2.2-3.4 5-5.6 8.4-6.6"/>',
  people:     '<circle cx="9" cy="8.2" r="3.1"/><path d="M3.6 19c0-3 2.4-4.9 5.4-4.9s5.4 1.9 5.4 4.9"/><path d="M16.2 5.4a3.1 3.1 0 0 1 0 5.7"/><path d="M17.8 14.4c2.1.5 3.6 2 3.6 4.6"/>',
  briefcase:  '<rect x="3" y="7.5" width="18" height="12" rx="2.2"/><path d="M8.5 7.5V6.2a2.2 2.2 0 0 1 2.2-2.2h2.6a2.2 2.2 0 0 1 2.2 2.2v1.3"/><path d="M3 12.6h18"/>',
};

// SSA code → { colour token (for CSS), hex (for SVG/canvas string injection), icon key }
const SUBJECT_META = {
  '1':  { colour: 'var(--sub-health)',       hex: '#EC3D62', icon: 'heartpulse' },
  '4':  { colour: 'var(--sub-engineering)',  hex: '#2A4A9C', icon: 'gear'       },
  '5':  { colour: 'var(--sub-construction)', hex: '#E07B12', icon: 'crane'      },
  '6':  { colour: 'var(--sub-computing)',    hex: '#2D7FF0', icon: 'code'       },
  '9':  { colour: 'var(--sub-creative)',     hex: '#9A45D6', icon: 'palette'    },
  '11': { colour: 'var(--sub-social)',       hex: '#A6398A', icon: 'people'     },
  '15': { colour: 'var(--sub-business)',     hex: '#8F6B17', icon: 'briefcase'  },
  '99': { colour: 'var(--sub-sustain)',      hex: '#1E9E55', icon: 'leaf'       },
};

const NEUTRAL = { colour: 'var(--sub-neutral)', hex: '#5B6473', icon: 'briefcase' };

/** Full metadata for a subject: { label, colour, hex, icon } */
export function subject(ssa) {
  const code = String(ssa ?? '');
  const meta = SUBJECT_META[code] || NEUTRAL;
  return { label: SSA_LABELS[code] || 'Other', ...meta };
}

/** Raw hex for SVG-string / canvas injection (Leaflet pins, etc.) — NOT a CSS var. */
export function subjectHex(ssa) {
  const code = String(ssa ?? '');
  return (SUBJECT_META[code] || NEUTRAL).hex;
}

/** Inline <svg> for a subject (CSS-sized — let context control width/height). */
export function subjectIconSvg(ssa) {
  const { icon } = subject(ssa);
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${SUBJECT_ICONS[icon] || ''}</svg>`;
}
