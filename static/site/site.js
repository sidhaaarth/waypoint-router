// Shared bits for the signed-out site: theme, styles, nav, footer, benchmark fetch.
// Load in <head> BEFORE the Tailwind CDN script, then: <script>tailwind.config = TW</script>.
(function () {
  var t = null;
  try { t = localStorage.getItem('theme'); } catch (e) {}
  if (t === 'light') document.documentElement.classList.remove('dark');
})();

window.TW = {
  darkMode: 'class',
  theme: { extend: {
    fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'], mono: ['JetBrains Mono', 'ui-monospace', 'monospace'] },
    colors: { accent: { DEFAULT: '#76b900', soft: '#8fd400', dim: '#5a8f00' } }
  } }
};

// Same component classes and light-contrast rules as the dashboard (static/index.html).
document.head.insertAdjacentHTML('beforeend', `
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style type="text/tailwindcss">
  @layer components {
    .card { @apply rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900/70 dark:shadow-none; }
    .card-h { @apply flex items-center justify-between gap-3 border-b border-zinc-200 px-4 py-3 dark:border-zinc-800; }
    .card-t { @apply text-sm font-semibold text-zinc-900 dark:text-zinc-100; }
    .btn { @apply inline-flex items-center justify-center gap-2 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-800 transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:bg-zinc-800; }
    .btn-primary { @apply border-accent bg-accent text-zinc-950 hover:bg-accent-soft dark:border-accent dark:bg-accent dark:text-zinc-950 dark:hover:bg-accent-soft; }
    .btn-sm { @apply px-2 py-1 text-xs; }
    .input { @apply w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 placeholder-zinc-400 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/30 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100 dark:placeholder-zinc-600; }
    .label { @apply mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400; }
    .muted { @apply text-zinc-500 dark:text-zinc-400; }
    .badge { @apply inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-4; }
    .b-gray { @apply border-zinc-300 bg-zinc-100 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300; }
    .b-amber { @apply border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300; }
    .tier-nano { @apply border-lime-300 bg-lime-50 text-lime-800 dark:border-accent/30 dark:bg-accent/10 dark:text-accent-soft; }
    .tier-super { @apply border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-300; }
    .tier-ultra { @apply border-violet-300 bg-violet-50 text-violet-800 dark:border-violet-400/30 dark:bg-violet-400/10 dark:text-violet-300; }
    .th { @apply whitespace-nowrap px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400; }
    .td { @apply px-3 py-2 text-sm text-zinc-800 dark:text-zinc-200; }
    .tbl { @apply min-w-full divide-y divide-zinc-200 dark:divide-zinc-800; }
    .tbl tbody { @apply divide-y divide-zinc-100 dark:divide-zinc-800/70; }
    .navlink { @apply shrink-0 rounded-lg px-3 py-2 text-sm font-medium text-zinc-600 transition hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800/70 dark:hover:text-zinc-100; }
    .navlink.active { @apply bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-white; }
    .code { @apply overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 font-mono text-[12.5px] leading-relaxed text-zinc-800 dark:border-zinc-800 dark:bg-black/40 dark:text-zinc-200; }
    .src { @apply mt-2 text-xs muted; }
    .h2 { @apply text-2xl font-semibold tracking-tight md:text-3xl; }
  }
  body { font-feature-settings: "cv11", "ss01"; }
  .nav-scroll { scrollbar-width: none; }
  .nav-scroll::-webkit-scrollbar { display: none; }
</style>
<style>
html:not(.dark) body { background-color: #eceff3; }
html:not(.dark) .card { border-color: #b6bec9; box-shadow: 0 1px 2px rgba(16, 24, 40, .08), 0 1px 1px rgba(16, 24, 40, .04); }
html:not(.dark) .card-h,
html:not(.dark) .border-zinc-200, html:not(.dark) .border-gray-200, html:not(.dark) .border-slate-200 { border-color: #c3cad3; }
html:not(.dark) .btn:not(.btn-primary), html:not(.dark) .input,
html:not(.dark) .border-zinc-300, html:not(.dark) .border-gray-300, html:not(.dark) .border-slate-300 { border-color: #a7afba; }
html:not(.dark) .b-gray { border-color: #b6bec9; }
</style>`);

const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const num = v => (v == null || v === '' || isNaN(Number(v))) ? null : Number(v);
const pctF = (v, d = 0) => num(v) == null ? '—' : (Number(v) * 100).toFixed(d) + '%';   // fraction -> %
const pctP = (v, d = 0) => num(v) == null ? '—' : Number(v).toFixed(d) + '%';          // already percent
function money(v) {
  v = num(v); if (v == null) return '—';
  if (v === 0) return '$0.00';
  if (v < 0.01) return '$' + v.toFixed(Math.min(10, 1 - Math.floor(Math.log10(v))));
  return '$' + v.toFixed(v < 1 ? 4 : 2);
}
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); return true; } catch (e) { return false; } }
};
let _bench = null;
const bench = () => _bench || (_bench = fetch('/api/benchmark').then(r => r.ok ? r.json() : {}).catch(() => ({})));

// Headline numbers from the 600-email classification benchmark (averaged over its datasets, equal size).
function headline(d) {
  const cls = ((d || {}).classification || {}).classification || {}, ks = Object.keys(cls);
  if (!ks.length) return null;
  const avg = f => ks.reduce((a, k) => a + (num(f(cls[k].cascade || {})) || 0), 0) / ks.length;
  const r = avg(c => c.cost_per_1k), u = avg(c => c.ultra_cost_per_1k);
  return { n: ks.reduce((a, k) => a + (num(cls[k].n) || 0), 0), datasets: ks.length,
           saved: u ? 1 - r / u : null, kept: avg(c => c.ultra_acc_test) ? avg(c => c.acc) / avg(c => c.ultra_acc_test) : null,
           onNano: avg(c => c.on_nano) };
}

const LOGO = '<img src="/brand/waypoint-mark-64.png" alt="" class="h-7 w-7 shrink-0 rounded-md" width="28" height="28">';
const SUN = '<svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON = '<svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>';

document.addEventListener('DOMContentLoaded', () => {
  const here = location.pathname.replace(/\/$/, '') || '/';
  const link = (href, label) => `<a href="${href}" class="navlink${here === href ? ' active' : ''}">${label}</a>`;
  document.body.insertAdjacentHTML('afterbegin', `
  <header class="sticky top-0 z-40 border-b border-zinc-200 bg-white/85 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/85">
    <div class="mx-auto flex max-w-6xl flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2.5">
      <a href="/" class="flex items-center gap-2 font-semibold">${LOGO}<span>Waypoint</span></a>
      <div class="ml-auto flex items-center gap-2 md:order-3">
        <span id="authBtns"><a href="/login" class="btn btn-sm${here === '/login' ? ' ring-2 ring-accent/40' : ''}">Log in</a> <a href="/signup" class="btn btn-primary btn-sm">Sign up</a></span>
        <button id="themeBtn" class="btn btn-sm" title="Toggle theme" aria-label="Toggle theme"></button>
      </div>
      <nav class="nav-scroll -mx-1 flex w-full gap-1 overflow-x-auto md:order-2 md:w-auto" aria-label="Site">
        ${link('/', 'Home')}${link('/how', 'How it works')}${link('/results', 'Results')}${link('/models', 'Models &amp; platforms')}${link('/docs', 'Docs')}
      </nav>
    </div>
  </header>`);
  document.body.insertAdjacentHTML('beforeend', `
  <footer class="mt-16 border-t border-zinc-200 dark:border-zinc-800">
    <div class="mx-auto flex max-w-6xl px-4 py-6 text-xs muted">
      <span>Waypoint · © ${new Date().getFullYear()}</span>
    </div>
  </footer>`);
  // Signed-out pages redirect when signed in, so only shared pages (Docs) can land here signed in.
  fetch('/api/session').then(r => r.json()).then(d => {
    if (d.signed_in) $('#authBtns').innerHTML = '<a href="/dashboard" class="btn btn-primary btn-sm">Dashboard</a>';
    if (d.self_hosted) document.querySelectorAll('nav[aria-label="Site"] a:not([href="/docs"])').forEach(x => x.remove());  // no showcase pages
  }).catch(() => {});
  const btn = $('#themeBtn'), paint = () => { btn.innerHTML = document.documentElement.classList.contains('dark') ? SUN : MOON; };
  btn.addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
    store.set('theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
    paint();
  });
  paint();
});
