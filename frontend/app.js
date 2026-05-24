/* ── global state ── */
let sessionId   = null;
let currentPrefs = null;
let historyEntries = [];

/* ══════════════════════════════════════════
   Init
══════════════════════════════════════════ */
async function init() {
  try {
    const resp = await fetch('/candidates');
    const candidates = await resp.json();
    const sel = document.getElementById('candidateSelect');
    sel.innerHTML = '<option value="">— select a candidate —</option>';
    candidates.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.person_id;
      opt.textContent = `${c.name}`;
      sel.appendChild(opt);
    });
  } catch {
    toast('Could not load candidates list.', 'error');
  }
}

/* ══════════════════════════════════════════
   Session
══════════════════════════════════════════ */
async function startSession() {
  const adHocText = document.getElementById('adHocJson').value.trim();
  const selectedId = document.getElementById('candidateSelect').value;

  let body;
  if (adHocText) {
    let parsed;
    try { parsed = JSON.parse(adHocText); }
    catch { toast('Invalid JSON — please check the pasted candidate.', 'error'); return; }
    body = { candidate: parsed };
  } else if (selectedId) {
    body = { candidate_id: parseInt(selectedId) };
  } else {
    toast('Please select a candidate or paste a JSON profile.', 'warn');
    return;
  }

  setSpinner(true, 'Seeding preferences from profile…');
  setBtn('startBtn', true);
  try {
    const resp = await fetch('/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const e = await resp.json();
      throw new Error(e.detail || resp.statusText);
    }
    const data = await resp.json();

    sessionId      = data.session_id;
    currentPrefs   = data.preferences;
    historyEntries = [];

    showCandidateCard(data.candidate);
    renderPreferences(data.preferences);

    // Reset centre
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('controls').classList.remove('hidden');
    document.getElementById('jobCards').innerHTML = '';
    document.getElementById('feedbackBox').classList.add('hidden');
    document.getElementById('historyPanel').classList.add('hidden');
    document.getElementById('historyList').innerHTML = '';
    document.getElementById('debugPanel').classList.add('hidden');
    document.getElementById('roundBadge').classList.add('hidden');
    document.getElementById('sessionPill').classList.remove('hidden');
    document.getElementById('recommendBtn').textContent = 'Get Recommendations';

    toast(`Session started for ${data.candidate.name}`, 'success');
  } catch (err) {
    toast('Error starting session: ' + err.message, 'error');
  } finally {
    setSpinner(false);
    setBtn('startBtn', false);
  }
}

/* ══════════════════════════════════════════
   Recommendations
══════════════════════════════════════════ */
async function getRecommendations() {
  if (!sessionId) return;
  setSpinner(true, 'Retrieving & ranking jobs…');
  setBtn('recommendBtn', true);
  try {
    const resp = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || resp.statusText); }
    const data = await resp.json();
    renderResults(data);
    toast('Top 3 jobs ready', 'success');
  } catch (err) {
    toast('Error: ' + err.message, 'error');
  } finally {
    setSpinner(false);
    setBtn('recommendBtn', false);
  }
}

/* ══════════════════════════════════════════
   Feedback
══════════════════════════════════════════ */
async function sendFeedback() {
  const feedback = document.getElementById('feedbackText').value.trim();
  if (!feedback) { toast('Please enter some feedback first.', 'warn'); return; }
  if (!sessionId) return;

  setSpinner(true, 'Updating preferences & re-ranking…');
  try {
    const resp = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, feedback }),
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || resp.statusText); }
    const data = await resp.json();

    renderHistory(feedback, currentPrefs, data.preferences);
    currentPrefs = data.preferences;
    document.getElementById('feedbackText').value = '';
    renderResults(data);
    toast('Preferences updated — here are your refined results', 'success');
  } catch (err) {
    toast('Error: ' + err.message, 'error');
  } finally {
    setSpinner(false);
  }
}

/* ══════════════════════════════════════════
   Render helpers
══════════════════════════════════════════ */
function renderResults(data) {
  currentPrefs = data.preferences;
  renderPreferences(data.preferences);
  renderDebug(data.debug, data.round);
  renderJobCards(data.jobs);
  document.getElementById('feedbackBox').classList.remove('hidden');
  document.getElementById('roundBadge').classList.remove('hidden');
  document.getElementById('roundNum').textContent = data.round;
  document.getElementById('recommendBtn').textContent = 'Re-fetch fresh round';
  document.getElementById('emptyState').classList.add('hidden');
}

/* ── Job cards ── */
function renderJobCards(jobs) {
  const container = document.getElementById('jobCards');
  container.innerHTML = '';
  jobs.forEach((item, idx) => {
    const { job, bm25_score, rerank_score, reasons, concerns } = item;
    const card = document.createElement('div');
    card.className = 'bg-white rounded-2xl shadow-card hover:shadow-card-hover transition-shadow p-6 fade-in';

    const rank   = ['🥇','🥈','🥉'][idx] || `#${idx+1}`;
    const sClass = rerank_score >= 75 ? 'score-high' : rerank_score >= 50 ? 'score-mid' : 'score-low';

    card.innerHTML = `
      <!-- Header row -->
      <div class="flex items-start gap-3 mb-3">
        <span class="text-xl leading-none mt-0.5">${rank}</span>
        <div class="flex-1 min-w-0">
          <h3 class="font-bold text-slate-900 text-[15px] leading-snug truncate">${esc(job.title)}</h3>
          <p class="text-sm text-slate-500 font-medium mt-0.5">
            ${esc(job.company)}
            ${job.yc_batch ? `<span class="ml-1.5 text-[11px] font-semibold text-orange-500 bg-orange-50 px-1.5 py-0.5 rounded">${esc(job.yc_batch)}</span>` : ''}
          </p>
        </div>
        <!-- Match score badge -->
        <div class="shrink-0 flex flex-col items-center gap-0.5">
          <span class="${sClass} text-sm font-bold px-2.5 py-1 rounded-lg">${rerank_score}</span>
          <span class="text-[10px] text-slate-400 font-medium">/ 100</span>
        </div>
      </div>

      <!-- Meta chips -->
      <div class="flex flex-wrap gap-1.5 mb-3">
        ${chip(job.location, 'slate')}
        ${chip(job.job_type, 'slate')}
        ${chip(job.salary || '', 'slate')}
        ${chip(job.experience ? job.experience + ' exp.' : '', 'slate')}
        ${job.is_remote   ? chip('Remote','green')  : ''}
        ${job.will_sponsor ? chip('Visa sponsor','blue') : ''}
      </div>

      <!-- Retrieval score -->
      <p class="text-[11px] text-slate-300 mb-3 font-mono">
        BM25 retrieval: ${bm25_score.toFixed(3)}
      </p>

      <!-- Divider -->
      <div class="border-t border-slate-100 mb-3"></div>

      <!-- Why this job (open by default) -->
      <details open>
        <summary class="flex items-center gap-1.5 text-xs font-semibold text-emerald-700 mb-1.5 select-none group">
          <svg class="w-3.5 h-3.5 transition group-open:rotate-90" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M7.293 4.293a1 1 0 011.414 0l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414-1.414L11.586 10 7.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
          Why this job
        </summary>
        <ul class="space-y-1 pl-1">
          ${reasons.map(r => `
          <li class="flex gap-2 text-sm text-slate-600">
            <span class="text-emerald-500 shrink-0 mt-0.5">✓</span>
            <span>${esc(r)}</span>
          </li>`).join('')}
        </ul>
      </details>

      <!-- Concerns (collapsed) -->
      ${concerns.length > 0 ? `
      <details class="mt-2">
        <summary class="flex items-center gap-1.5 text-xs font-semibold text-amber-600 mb-1.5 select-none group">
          <svg class="w-3.5 h-3.5 transition group-open:rotate-90" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M7.293 4.293a1 1 0 011.414 0l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414-1.414L11.586 10 7.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
          Possible concerns
        </summary>
        <ul class="space-y-1 pl-1">
          ${concerns.map(c => `
          <li class="flex gap-2 text-sm text-slate-500">
            <span class="text-amber-400 shrink-0 mt-0.5">⚠</span>
            <span>${esc(c)}</span>
          </li>`).join('')}
        </ul>
      </details>` : ''}

      <!-- Link -->
      ${job.url ? `
      <a href="${esc(job.url)}" target="_blank" rel="noopener"
        class="inline-flex items-center gap-1 mt-4 text-xs font-medium text-brand hover:text-brand-dark transition">
        View job posting
        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
        </svg>
      </a>` : ''}
    `;
    container.appendChild(card);
  });
}

/* ── Preferences visual ── */
function renderPreferences(prefs) {
  document.getElementById('prefsPanel').classList.remove('hidden');
  document.getElementById('prefsJson').textContent = JSON.stringify(prefs, null, 2);

  const vis = document.getElementById('prefsVisual');
  vis.innerHTML = '';

  if (prefs.prefer?.length) {
    vis.innerHTML += `
      <div>
        <p class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">Prefer</p>
        <div class="flex flex-wrap gap-1.5">
          ${prefs.prefer.map(t => `<span class="text-xs bg-brand-light text-brand-dark px-2 py-0.5 rounded-full font-medium">${esc(t)}</span>`).join('')}
        </div>
      </div>`;
  }
  if (prefs.avoid?.length) {
    vis.innerHTML += `
      <div>
        <p class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">Avoid</p>
        <div class="flex flex-wrap gap-1.5">
          ${prefs.avoid.map(t => `<span class="text-xs bg-red-50 text-red-600 px-2 py-0.5 rounded-full font-medium">${esc(t)}</span>`).join('')}
        </div>
      </div>`;
  }
  const musts = [];
  if (prefs.must?.remote_only)           musts.push('Remote only');
  if (prefs.must?.require_sponsorship)   musts.push('Visa sponsorship');
  if (prefs.must?.job_types?.length)     musts.push(...prefs.must.job_types);
  if (musts.length) {
    vis.innerHTML += `
      <div>
        <p class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">Must</p>
        <div class="flex flex-wrap gap-1.5">
          ${musts.map(t => `<span class="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-medium">${esc(t)}</span>`).join('')}
        </div>
      </div>`;
  }
  if (prefs.free_text_notes) {
    vis.innerHTML += `<p class="text-xs text-slate-400 italic leading-relaxed border-t border-slate-100 pt-2 mt-1">${esc(prefs.free_text_notes)}</p>`;
  }
}

/* ── Debug visual ── */
function renderDebug(debug, round) {
  document.getElementById('debugPanel').classList.remove('hidden');
  document.getElementById('debugJson').textContent = JSON.stringify({ round, ...debug }, null, 2);

  const vis = document.getElementById('debugVisual');
  vis.innerHTML = `
    <div class="grid grid-cols-2 gap-2">
      <div class="bg-slate-50 rounded-lg p-2.5 text-center">
        <p class="text-[11px] text-slate-400 font-medium">Retrieved</p>
        <p class="text-base font-bold text-slate-700">${debug.retrieved_count}</p>
      </div>
      <div class="bg-slate-50 rounded-lg p-2.5 text-center">
        <p class="text-[11px] text-slate-400 font-medium">After filter</p>
        <p class="text-base font-bold text-slate-700">${debug.after_filter_count}</p>
      </div>
    </div>
    <p class="text-[11px] font-mono text-slate-500 bg-slate-50 rounded-lg px-2.5 py-2 leading-relaxed">
      <span class="font-semibold text-slate-400">query: </span>${esc(debug.query?.slice(0, 100))}…
    </p>
    ${debug.filters_applied?.length ? `
    <div class="flex flex-wrap gap-1">
      ${debug.filters_applied.map(f => `<span class="text-[10px] bg-amber-50 text-amber-600 font-medium px-2 py-0.5 rounded-full">${esc(f)}</span>`).join('')}
    </div>` : '<p class="text-[11px] text-slate-400">No hard filters active</p>'}
  `;
}

/* ── Feedback history ── */
function renderHistory(feedback, prefsBefore, prefsAfter) {
  document.getElementById('historyPanel').classList.remove('hidden');
  const list = document.getElementById('historyList');

  const added    = prefsAfter.prefer.filter(t => !prefsBefore.prefer.includes(t));
  const removed  = prefsBefore.prefer.filter(t => !prefsAfter.prefer.includes(t));
  const newAvoid = prefsAfter.avoid.filter(t => !prefsBefore.avoid.includes(t));

  let diffHtml = '';
  if (added.length)    diffHtml += `<p class="text-emerald-600">＋ prefer: ${added.map(esc).join(', ')}</p>`;
  if (removed.length)  diffHtml += `<p class="text-red-500">－ prefer: ${removed.map(esc).join(', ')}</p>`;
  if (newAvoid.length) diffHtml += `<p class="text-amber-600">＋ avoid: ${newAvoid.map(esc).join(', ')}</p>`;
  if (prefsAfter.must?.remote_only && !prefsBefore.must?.remote_only)
    diffHtml += `<p class="text-blue-600">↑ remote only enabled</p>`;
  if (!diffHtml) diffHtml = '<p class="text-slate-400">No structural preference change</p>';

  const entry = document.createElement('div');
  entry.className = 'bg-slate-50 rounded-xl p-3 fade-in border border-slate-100';
  entry.innerHTML = `
    <p class="text-xs font-medium text-slate-700 leading-snug mb-2">"${esc(feedback)}"</p>
    <div class="text-[11px] space-y-0.5">${diffHtml}</div>
  `;
  list.prepend(entry);
}

/* ── Candidate card ── */
function showCandidateCard(c) {
  document.getElementById('candidateCard').classList.remove('hidden');
  document.getElementById('cName').textContent     = c.name;
  document.getElementById('cHeadline').textContent = c.headline;
  document.getElementById('cLocation').textContent = c.location;
  // Avatar initials
  const initials = c.name.split(' ').map(w => w[0]).join('').slice(0,2).toUpperCase();
  document.getElementById('cAvatar').textContent = initials;
}

/* ══════════════════════════════════════════
   Utilities
══════════════════════════════════════════ */

function setSpinner(on, msg = 'Thinking…') {
  document.getElementById('spinner').classList.toggle('hidden', !on);
  document.getElementById('spinnerMsg').textContent = msg;
  if (on) {
    document.getElementById('jobCards').innerHTML = '';
  }
}

function setBtn(id, disabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = disabled;
}

function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function chip(text, color = 'slate') {
  if (!text) return '';
  const palette = {
    slate: 'bg-slate-100 text-slate-600',
    green: 'bg-emerald-50 text-emerald-700',
    blue:  'bg-blue-50 text-blue-700',
  };
  return `<span class="${palette[color] || palette.slate} text-[11px] font-medium px-2 py-0.5 rounded-full">${esc(text)}</span>`;
}

/* ── Toast notifications ── */
let _toastTimer = null;
function toast(msg, type = 'info') {
  const el    = document.getElementById('toast');
  const inner = document.getElementById('toastInner');
  const icon  = document.getElementById('toastIcon');
  const msgEl = document.getElementById('toastMsg');

  const styles = {
    success: 'bg-emerald-50 text-emerald-900 border border-emerald-200',
    error:   'bg-red-50 text-red-900 border border-red-200',
    warn:    'bg-amber-50 text-amber-900 border border-amber-200',
    info:    'bg-white text-slate-800 border border-slate-200',
  };
  const icons = { success: '✅', error: '❌', warn: '⚠️', info: 'ℹ️' };

  inner.className = `toast-enter flex items-start gap-3 max-w-sm rounded-xl px-4 py-3 shadow-xl text-sm font-medium ${styles[type] || styles.info}`;
  icon.textContent = icons[type] || icons.info;
  msgEl.textContent = msg;

  el.classList.remove('hidden');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.add('hidden'), 4000);
}

/* ══════════════════════════════════════════
   Boot
══════════════════════════════════════════ */
init();
