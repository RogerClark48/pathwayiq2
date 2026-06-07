/* Shared DOM helpers */

export function el(tag, className) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  return e;
}

/**
 * Split a prose blob into paragraph chunks:
 *   1. blank lines      → paragraph boundaries
 *   2. single newlines  → paragraph boundaries
 *   3. one flat block   → ~2-sentence groups
 * Exported so callers that need linkify-per-paragraph can split then annotate.
 */
export function splitProse(s) {
  let paras = s.split(/\n\s*\n/);
  if (paras.length === 1) paras = s.split(/\n/);
  if (paras.length === 1) {
    const sents = s.match(/[^.!?]+[.!?]+(\s|$)/g) || [s];
    paras = [];
    for (let i = 0; i < sents.length; i += 2)
      paras.push(sents.slice(i, i + 2).join(' ').trim());
  }
  return paras;
}

/**
 * Render prose as real paragraphs.
 * Accepts string[] (preferred — one item per paragraph) or a blob string.
 * All text injected via textContent — never innerHTML.
 */
export function renderProse(input, className = 'prose') {
  const box = el('div', className);
  const raw = Array.isArray(input) ? input : splitProse(String(input || '').trim());
  raw.map(p => String(p).trim()).filter(Boolean).forEach(par => {
    const p = el('p');
    p.textContent = par;
    box.appendChild(p);
  });
  return box;
}

/**
 * Render a field that may be prose or a delimited list (•, -, *, digit.).
 * Returns a <ul class="nb-mods"> for list-shaped content, renderProse() otherwise.
 */
export function renderField(text, proseClass = 'prose') {
  const s = String(text || '').trim();
  if (!s) return el('div', proseClass);
  const lines = s.split('\n').map(l => l.trim()).filter(Boolean);
  const isList = lines.length > 1 && lines.some(l => /^[•\-\*\d]/.test(l));
  if (isList) {
    const ul = el('ul', 'nb-mods');
    lines.forEach(line => {
      const clean = line.replace(/^[•\-\*\d]+[.)]*\s*/, '').trim();
      if (!clean) return;
      const li = el('li');
      li.textContent = clean;
      ul.appendChild(li);
    });
    return ul;
  }
  return renderProse(s, proseClass);
}
