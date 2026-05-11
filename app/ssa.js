/* FutureFinder v3 — SSA taxonomy config */

// Standard SSA framework codes 1–15, plus custom 99 (Sustainability).
// Used to label subject chips. If a code arrives from the API that isn't
// listed here the frontend console.warns and skips it.
export const SSA_LABELS = {
  '1':  'Health & Care',
  '2':  'Science & Maths',
  '3':  'Agriculture & Land',
  '4':  'Engineering & Manufacturing',
  '5':  'Construction & Built Environment',
  '6':  'Computing & IT',
  '7':  'Retail & Commerce',
  '8':  'Hospitality & Tourism',
  '9':  'Creative & Media',
  '10': 'History & Philosophy',
  '11': 'Social Sciences',
  '12': 'Languages & Culture',
  '13': 'Education & Training',
  '14': 'Life & Work',
  '15': 'Business & Law',
  '99': 'Sustainability',
};

// Preferred display order for qualification chips (roughly ascending RQF level).
// Quals returned from the API that aren't listed here sort alphabetically after.
export const QUAL_ORDER = [
  'T Level',
  'Apprenticeship',
  'Access to HE',
  'HNC',
  'HND',
  'CertHE / DipHE',
  'Foundation Degree',
  "Bachelor's Degree",
  "Master's Degree",
  'Short Course',
];

export function sortQuals(quals) {
  return [...quals].sort((a, b) => {
    const ai = QUAL_ORDER.indexOf(a);
    const bi = QUAL_ORDER.indexOf(b);
    if (ai === -1 && bi === -1) return a.localeCompare(b);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
}
