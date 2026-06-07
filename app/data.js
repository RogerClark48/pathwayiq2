/* FutureFinder v3 — static catalogue data (phase 1) */

export const QUALS = [
  { label: 'T Level',         value: 'T Level' },
  { label: 'A Level',         value: 'A Level' },
  { label: 'HNC / HND',       value: 'HNC / HND' },
  { label: 'Apprenticeship',  value: 'Apprenticeship' },
  { label: 'Degree',          value: 'Degree' },
  { label: 'Postgraduate',    value: 'Postgraduate' },
];

export const SUBJECTS = [
  { label: 'Engineering & Manufacturing',      ssa: '4' },
  { label: 'Construction & Built Environment', ssa: '5' },
  { label: 'Computing & IT',                   ssa: '6' },
  { label: 'Creative & Media',                 ssa: '9' },
  { label: 'Health & Care',                    ssa: '1' },
];

// Course counts per [qual][ssa] — derived from GMIoT catalogue
export const COUNTS = {
  'T Level':        { '1': 2, '4':  0, '5':  0, '6': 2, '9': 0 },
  'A Level':        { '1': 0, '4':  0, '5':  0, '6': 0, '9': 0 },
  'HNC / HND':      { '1': 4, '4': 15, '5': 15, '6': 6, '9': 2 },
  'Apprenticeship': { '1': 1, '4':  2, '5':  3, '6': 3, '9': 0 },
  'Degree':         { '1': 2, '4':  2, '5':  3, '6': 6, '9': 7 },
  'Postgraduate':   { '1': 0, '4':  0, '5':  0, '6': 1, '9': 0 },
};

export function courseCount(qual, subjects) {
  if (!qual || subjects.size === 0) return 0;
  const row = COUNTS[qual] || {};
  let total = 0;
  subjects.forEach(ssa => { total += row[ssa] || 0; });
  return total;
}

export function subjectCount(qual, ssa) {
  if (!qual) return null;
  return (COUNTS[qual] || {})[ssa] || 0;
}
