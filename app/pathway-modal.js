/* Qualification pathway map — modal, render-on-demand */

import { getWelcomeData } from './api.js';

const PATHWAY_EXPLANATIONS = {
  gcse: {
    heading: 'GCSEs / Level 2',
    body: "GCSEs are the qualifications most students take at the end of secondary school, usually at age 16. They're the starting point for most further study. You don't need specific GCSEs for every path — but having a solid set, especially in English and Maths, opens more doors.",
    leads_to: 'T Levels, A Levels, Apprenticeships',
    gmiot: false,
  },
  t_level: {
    heading: 'T Level',
    body: 'A T Level is a two-year technical qualification equivalent to three A Levels. It combines classroom learning with a substantial industry placement (at least 45 days with an employer), so you graduate with both a qualification and real work experience. Designed specifically for students who want a career in a technical field.',
    leads_to: 'HNC, HND, or degree-level study',
    gmiot: true,
  },
  a_levels: {
    heading: 'A Levels',
    not_gmiot: true,
  },
  apprenticeship: {
    heading: 'Apprenticeship (Level 3)',
    body: "An apprenticeship lets you earn a wage while working towards a qualification. You spend most of your time with an employer and some time in college or training. A Level 3 apprenticeship is equivalent to A Levels. A good route if you know the industry you want to work in and prefer learning on the job.",
    leads_to: 'Higher Apprenticeship or further study',
    gmiot: true,
  },
  hnc_hnd: {
    heading: 'HNC / HND',
    body: "HNCs (Higher National Certificates) and HNDs (Higher National Diplomas) are higher education qualifications delivered at college. An HNC is one year full-time (Level 4, equivalent to the first year of a degree). An HND is two years full-time (Level 5, equivalent to the first two years of a degree). Both can lead directly into employment or onto the second or third year of a Bachelor's degree.",
    leads_to: "Bachelor's degree (often with advanced entry)",
    gmiot: true,
  },
  foundation_degree: {
    heading: 'Foundation Degree',
    body: "A Foundation Degree is a two-year higher education qualification at Level 5 — equivalent to the first two years of a Bachelor's degree. Usually delivered at college in partnership with a university. Can be topped up to a full Bachelor's degree in one further year.",
    leads_to: "Bachelor's degree top-up",
    gmiot: true,
  },
  higher_apprenticeship: {
    heading: 'Higher Apprenticeship',
    body: "A Higher Apprenticeship works the same way as a Level 3 apprenticeship — earning while learning — but at a higher level (Level 4–6). You could come out with an HNC, HND, or even a degree-level qualification, all while being employed. Increasingly common in engineering, digital, and construction sectors.",
    leads_to: "Employment or Bachelor's degree",
    gmiot: false,
  },
  bachelors: {
    heading: "Bachelor's Degree",
    body: "A Bachelor's degree is a full undergraduate degree, typically three years full-time at university (four in Scotland or for some courses with a placement year). It's Level 6 on the qualification framework. The most common route into graduate-level professional roles. Can be entered directly from A Levels, T Levels, or via top-up from an HND or Foundation Degree.",
    leads_to: "Employment or Master's degree",
    gmiot: true,
  },
  masters: {
    heading: "Master's Degree",
    body: "A Master's degree is a postgraduate qualification taken after a Bachelor's degree, usually one year full-time. It allows you to specialise deeply in a subject or field, and is often required or preferred for senior professional or research roles.",
    leads_to: 'Senior professional roles or research',
    gmiot: true,
  },
};

const MODAL_HTML = `
<div class="pathway-panel">
  <div class="pathway-panel-header">
    <span class="pathway-panel-title">Qualification pathways</span>
    <button class="pathway-close" aria-label="Close">✕</button>
  </div>
  <p class="pathway-intro">Tap any qualification to find out more.</p>
  <div class="pathway-map-wrap">
    <svg viewBox="0 0 385 615" width="100%" class="pathway-svg" aria-hidden="true">
      <defs>
        <marker id="am-teal"   markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#0d9488"/></marker>
        <marker id="am-amber"  markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#d97706"/></marker>
        <marker id="am-grey"   markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#9ca3af"/></marker>
        <marker id="am-purple" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#7c3aed"/></marker>
      </defs>
      <line x1="69.5"  y1="73" x2="69.5"  y2="130" stroke="#0d9488" stroke-width="2"   marker-end="url(#am-teal)"/>
      <line x1="192.5" y1="73" x2="192.5" y2="130" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#am-grey)"/>
      <line x1="315.5" y1="73" x2="315.5" y2="130" stroke="#d97706" stroke-width="2"   marker-end="url(#am-amber)"/>
      <line x1="69.5" y1="213" x2="69.5" y2="270" stroke="#0d9488" stroke-width="2" marker-end="url(#am-teal)"/>
      <line x1="192.5" y1="213" x2="74" y2="270" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#am-grey)"/>
      <line x1="192.5" y1="213" x2="192.5" y2="270" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#am-grey)"/>
      <line x1="315.5" y1="213" x2="315.5" y2="270" stroke="#d97706" stroke-width="2" marker-end="url(#am-amber)"/>
      <line x1="69.5"  y1="353" x2="69.5"  y2="410" stroke="#0d9488" stroke-width="2"   marker-end="url(#am-teal)"/>
      <line x1="192.5" y1="353" x2="192.5" y2="410" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#am-grey)"/>
      <line x1="315.5" y1="353" x2="315.5" y2="410" stroke="#d97706" stroke-width="2"   marker-end="url(#am-amber)"/>
      <line x1="192.5" y1="475" x2="192.5" y2="532" stroke="#7c3aed" stroke-width="2" marker-end="url(#am-purple)"/>
      <g class="pathway-node" data-node="gcse" role="button" tabindex="0" aria-label="GCSEs / Level 2">
        <rect x="12" y="10" width="361" height="62" rx="10" fill="#f8fafc" stroke="#374151" stroke-width="1.5"/>
        <text x="192.5" y="35" text-anchor="middle" font-weight="700" font-size="14" fill="#1f2937">GCSEs / Level 2</text>
        <text x="192.5" y="53" text-anchor="middle" font-size="11" fill="#6b7280">Starting point · Level 2</text>
      </g>
      <text x="69.5"  y="124" text-anchor="middle" font-size="9" fill="#0d9488" font-style="italic">Technical</text>
      <text x="192.5" y="124" text-anchor="middle" font-size="9" fill="#9ca3af" font-style="italic">Academic</text>
      <text x="315.5" y="124" text-anchor="middle" font-size="9" fill="#d97706" font-style="italic">Work-based</text>
      <g class="pathway-node" data-node="t_level" role="button" tabindex="0" aria-label="T Level">
        <rect x="12" y="132" width="115" height="80" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.5"/>
        <text x="69.5" y="158" text-anchor="middle" font-weight="700" font-size="13" fill="#0d9488">T Level</text>
        <text x="69.5" y="176" text-anchor="middle" font-size="10" fill="#0d9488" opacity="0.85">Level 3</text>
        <text x="69.5" y="192" text-anchor="middle" font-size="10" fill="#0d9488" opacity="0.85">2 years</text>
      </g>
      <g class="pathway-node pathway-node--muted" data-node="a_levels" role="button" tabindex="0" aria-label="A Levels">
        <rect x="135" y="132" width="115" height="80" rx="8" fill="#f9fafb" stroke="#d1d5db" stroke-width="1.5"/>
        <text x="192.5" y="158" text-anchor="middle" font-weight="700" font-size="13" fill="#9ca3af">A Levels</text>
        <text x="192.5" y="176" text-anchor="middle" font-size="10" fill="#9ca3af">Level 3</text>
        <text x="192.5" y="192" text-anchor="middle" font-size="10" fill="#9ca3af">2 years</text>
      </g>
      <g class="pathway-node" data-node="apprenticeship" role="button" tabindex="0" aria-label="Apprenticeship">
        <rect x="258" y="132" width="115" height="80" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
        <text x="315.5" y="162" text-anchor="middle" font-weight="700" font-size="11.5" fill="#d97706">Apprenticeship</text>
        <text x="315.5" y="180" text-anchor="middle" font-size="10" fill="#d97706" opacity="0.85">Level 3</text>
        <text x="315.5" y="196" text-anchor="middle" font-size="10" fill="#d97706" opacity="0.85">earn &amp; learn</text>
      </g>
      <g class="pathway-node" data-node="hnc_hnd" role="button" tabindex="0" aria-label="HNC / HND">
        <rect x="12" y="272" width="115" height="80" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.5"/>
        <text x="69.5" y="298" text-anchor="middle" font-weight="700" font-size="13" fill="#0d9488">HNC / HND</text>
        <text x="69.5" y="316" text-anchor="middle" font-size="10" fill="#0d9488" opacity="0.85">Level 4–5</text>
        <text x="69.5" y="330" text-anchor="middle" font-size="10" fill="#0d9488" opacity="0.85">1–2 years</text>
      </g>
      <g class="pathway-node" data-node="foundation_degree" role="button" tabindex="0" aria-label="Foundation Degree">
        <rect x="135" y="272" width="115" height="80" rx="8" fill="#f9fafb" stroke="#9ca3af" stroke-width="1.5"/>
        <text x="192.5" y="296" text-anchor="middle" font-weight="700" font-size="12" fill="#4b5563">Foundation</text>
        <text x="192.5" y="311" text-anchor="middle" font-weight="700" font-size="12" fill="#4b5563">Degree</text>
        <text x="192.5" y="328" text-anchor="middle" font-size="10" fill="#6b7280">Level 5</text>
        <text x="192.5" y="342" text-anchor="middle" font-size="10" fill="#6b7280">2 years</text>
      </g>
      <g class="pathway-node" data-node="higher_apprenticeship" role="button" tabindex="0" aria-label="Higher Apprenticeship">
        <rect x="258" y="272" width="115" height="80" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
        <text x="315.5" y="296" text-anchor="middle" font-weight="700" font-size="12" fill="#d97706">Higher</text>
        <text x="315.5" y="311" text-anchor="middle" font-weight="700" font-size="11.5" fill="#d97706">Apprenticeship</text>
        <text x="315.5" y="328" text-anchor="middle" font-size="10" fill="#d97706" opacity="0.85">Level 4–6</text>
        <text x="315.5" y="342" text-anchor="middle" font-size="10" fill="#d97706" opacity="0.85">earn &amp; learn</text>
      </g>
      <g class="pathway-node" data-node="bachelors" role="button" tabindex="0" aria-label="Bachelor's Degree">
        <rect x="12" y="412" width="361" height="62" rx="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
        <text x="192.5" y="436" text-anchor="middle" font-weight="700" font-size="14" fill="#7c3aed">Bachelor's Degree</text>
        <text x="192.5" y="455" text-anchor="middle" font-size="11" fill="#7c3aed" opacity="0.85">Level 6 · 3–4 years</text>
      </g>
      <g class="pathway-node" data-node="masters" role="button" tabindex="0" aria-label="Master's Degree">
        <rect x="12" y="534" width="361" height="62" rx="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
        <text x="192.5" y="558" text-anchor="middle" font-weight="700" font-size="14" fill="#7c3aed">Master's Degree</text>
        <text x="192.5" y="577" text-anchor="middle" font-size="11" fill="#7c3aed" opacity="0.85">Level 7 · 1–2 years</text>
      </g>
    </svg>
  </div>
  <div class="pathway-explanation" hidden></div>
</div>`;

function buildModal() {
  const modal = document.createElement('div');
  modal.className = 'pathway-modal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-label', 'Qualification pathway map');
  modal.innerHTML = MODAL_HTML;

  let activeNode = null;

  const expPanel  = modal.querySelector('.pathway-explanation');
  const closeBtn  = modal.querySelector('.pathway-close');

  function close() {
    modal.remove();
    document.removeEventListener('keydown', onKeydown);
  }

  function onKeydown(e) {
    if (e.key === 'Escape') close();
  }

  function renderExplanation(nodeKey) {
    const data = PATHWAY_EXPLANATIONS[nodeKey];
    if (!data) return;

    modal.querySelectorAll('.pathway-node--active').forEach(el => el.classList.remove('pathway-node--active'));
    modal.querySelector(`[data-node="${nodeKey}"]`).classList.add('pathway-node--active');
    activeNode = nodeKey;

    expPanel.hidden = false;
    expPanel.innerHTML = '';

    const heading = document.createElement('p');
    heading.className = 'pathway-exp-heading';
    heading.textContent = data.heading;
    expPanel.appendChild(heading);

    if (data.not_gmiot) {
      const note = document.createElement('p');
      note.className = 'pathway-exp-body';
      note.textContent = `A Levels are a well-established academic route, typically taken in sixth form or college. ${getWelcomeData()?.institution?.abbrev || 'GMIoT'} does not offer A Level courses — but A Levels remain a valid path into HNC, HND, Foundation Degree, or university study.`;
      expPanel.appendChild(note);
    } else {
      if (data.body) {
        const body = document.createElement('p');
        body.className = 'pathway-exp-body';
        body.textContent = data.body;
        expPanel.appendChild(body);
      }
      if (data.leads_to) {
        const leads = document.createElement('p');
        leads.className = 'pathway-exp-leads';
        leads.innerHTML = `<strong>Leads to:</strong> ${data.leads_to}`;
        expPanel.appendChild(leads);
      }
      if (data.gmiot) {
        const indicator = document.createElement('p');
        indicator.className = 'pathway-exp-gmiot';
        indicator.textContent = `${getWelcomeData()?.institution?.abbrev || 'GMIoT'} offers courses at this level — explore them in the app.`;
        expPanel.appendChild(indicator);
      }
    }

    expPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  closeBtn.addEventListener('click', close);
  modal.addEventListener('click', e => { if (e.target === modal) close(); });
  document.addEventListener('keydown', onKeydown);

  modal.querySelectorAll('.pathway-node').forEach(node => {
    const key = node.dataset.node;
    node.addEventListener('click', () => {
      if (activeNode === key) {
        node.classList.remove('pathway-node--active');
        activeNode = null;
        expPanel.hidden = true;
        expPanel.innerHTML = '';
      } else {
        renderExplanation(key);
      }
    });
    node.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); node.click(); }
    });
  });

  return { modal, closeBtn };
}

export function openPathwayModal() {
  const { modal, closeBtn } = buildModal();
  document.body.appendChild(modal);
  closeBtn.focus();
}
