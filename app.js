/* ═══════════════════════════════════════════════════════════
   CINEWORLD  –  app.js  (Real-Time TMDB API Integration)
   No numpy, pandas, or ML libraries – pure live API calls
═══════════════════════════════════════════════════════════ */

const IMG_BASE   = 'https://image.tmdb.org/t/p/';
const API_BASE   = 'https://api.themoviedb.org/3';
let   API_KEY    = 'a267dba0e5471e56eb27ecd88f2300bc'; // TMDB API Key
let   YT_API_KEY = localStorage.getItem('yt_key') || 'AIzaSyDj9B2gRtSNMZ6ROU5TH2gDjnNA4GSKjCE'; // Updated with provided key
let   selectedGenreIds = []; 
let   currentLang    = '';
let   heroMovies     = [];
let   heroIndex      = 0;
let   heroTimer      = null;
let   gridEndpoint   = '';
let   gridPage       = 1;
let   searchTimer    = null;

// ── Supported languages list ─────────────────────────────────
const LANGUAGES = [
  {code:'en',name:'English'},{code:'hi',name:'Hindi'},{code:'te',name:'Telugu'},
  {code:'ta',name:'Tamil'},{code:'ml',name:'Malayalam'},{code:'kn',name:'Kannada'},
  {code:'ko',name:'Korean'},{code:'ja',name:'Japanese'},{code:'zh',name:'Chinese'},
  {code:'fr',name:'French'},{code:'es',name:'Spanish'},{code:'de',name:'German'},
  {code:'it',name:'Italian'},{code:'pt',name:'Portuguese'},{code:'ru',name:'Russian'},
  {code:'ar',name:'Arabic'},{code:'tr',name:'Turkish'},{code:'th',name:'Thai'},
  {code:'id',name:'Indonesian'},{code:'nl',name:'Dutch'},{code:'pl',name:'Polish'},
  {code:'sv',name:'Swedish'},{code:'no',name:'Norwegian'},{code:'da',name:'Danish'},
  {code:'fi',name:'Finnish'},{code:'cs',name:'Czech'},{code:'hu',name:'Hungarian'},
  {code:'ro',name:'Romanian'},{code:'el',name:'Greek'},{code:'he',name:'Hebrew'},
  {code:'fa',name:'Persian'},{code:'bn',name:'Bengali'},{code:'pa',name:'Punjabi'},
  {code:'mr',name:'Marathi'},{code:'gu',name:'Gujarati'},{code:'ur',name:'Urdu'},
  {code:'uk',name:'Ukrainian'},{code:'vi',name:'Vietnamese'},{code:'ms',name:'Malay'},
  {code:'si',name:'Sinhala'},{code:'ne',name:'Nepali'},{code:'am',name:'Amharic'},
];

// ── API helper ───────────────────────────────────────────────
async function api(endpoint) {
  const sep = endpoint.includes('?') ? '&' : '?';
  const url = `${API_BASE}/${endpoint}${sep}api_key=${API_KEY}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

function getToday() { return new Date().toISOString().split('T')[0]; }

// ── Setup / Launch ────────────────────────────────────────────
async function launchApp() {
  const key = document.getElementById('apiKeyInput').value.trim();
  if (!key) return;
  document.getElementById('setupError').style.display = 'none';
  showLoader();
  try {
    API_KEY = key;
    const test = await api('configuration');
    if (!test.images) throw new Error('Invalid key');
    
    const ytKeyInput = document.getElementById('ytApiKeyInput').value.trim();
    if (ytKeyInput) {
       YT_API_KEY = ytKeyInput;
       localStorage.setItem('yt_key', YT_API_KEY);
    }
    localStorage.setItem('cineworld_key', key);
    localStorage.setItem('yt_key', YT_API_KEY);
    
    document.getElementById('setupScreen').style.display = 'none';
    document.getElementById('mainApp').style.display = 'block';
    await initApp();
  } catch {
    API_KEY = '';
    document.getElementById('setupError').style.display = 'block';
  }
  hideLoader();
}

function logout() {
  localStorage.removeItem('cineworld_key');
  location.reload();
}

// ── Auto-launch on page load (key pre-set) ───────────────────
window.addEventListener('DOMContentLoaded', async () => {
  // Hide setup screen, show app immediately
  document.getElementById('setupScreen').style.display = 'none';
  document.getElementById('mainApp').style.display = 'block';

  // Populate year dropdown
  const y = new Date().getFullYear();
  const sel = document.getElementById('gridYear');
  for (let yr = y + 2; yr >= 1945; yr--) {
    sel.innerHTML += `<option value="${yr}">${yr}</option>`;
  }

  showLoader();
  await fetchAllLanguages(); // Get ALL languages from API
  await initApp();
  hideLoader();
});

async function fetchAllLanguages() {
  try {
    const data = await api('configuration/languages');
    const langSel = document.getElementById('langFilter');
    // Keep common ones at top
    const priority = ['en','hi','te','ta','ml','kn','ko','ja'];
    const sorted = data.sort((a,b) => (priority.includes(b.iso_639_1)?1:0) - (priority.includes(a.iso_639_1)?1:0) || a.english_name.localeCompare(b.english_name));
    
    sorted.forEach(l => {
      langSel.innerHTML += `<option value="${l.iso_639_1}">${l.english_name}</option>`;
    });
  } catch(e) { console.error("Lang fetch failed", e); }
}

// ── Init App ─────────────────────────────────────────────────
async function initApp() {
  setupNavbar();
  setupSearch();
  loadGenres();
  // Load all rows in parallel
  await Promise.all([
    loadHeroBanner(),
    loadRow('trending/all/day',    'trendingCards',    true),
    loadRow('movie/now_playing',   'nowPlayingCards',  true),
    loadRow('discover/movie?with_watch_monetization_types=free&watch_region=IN', 'freeCards'),
    loadRow('movie/upcoming',      'upcomingCards',    true),
    loadRow('movie/top_rated',     'topRatedCards'),
    loadRow('trending/tv/day',     'tvTrendingCards',  true, 'tv'),
    loadRow('tv/popular',          'tvPopularCards',   false, 'tv'),
    loadLangRow('hi','bollywoodCards'),
    loadLangRow('te','tollywoodCards'),
    loadLangRow('ta','kollywoodCards'),
    loadLangRow('ml','mollywoodCards'),
    loadLangRow('kn','sandalwoodCards'),
    loadLangRow('ko','koreanCards'),
    loadLangRow('ja','japaneseCards'),
    loadLangRow('ko','kdramaCards', 'tv'),
    loadRow('discover/movie?with_genres=28',   'actionCards'),
    loadRow('discover/movie?with_genres=35',   'comedyCards'),
    loadRow('discover/movie?with_genres=878',  'scifiCards'),
    loadRow('discover/movie?with_genres=27',   'horrorCards'),
  ]);
  
  // Show base movie count
  // Initialize UI components
  updateSearchTags();
  const counter = document.getElementById('resultsCount');
  if (counter) {
    counter.textContent = '900k+ Movies';
    counter.style.display = 'block';
  }
}


// ── Search History & Tracking ──────────────────────────────────
function logSearch(q) {
  if (!q || q.length < 2) return;
  let history = JSON.parse(localStorage.getItem('cineworld_history') || '[]');
  // Remove duplicate and add to top
  history = [q, ...history.filter(h => h.toLowerCase() !== q.toLowerCase())].slice(0, 8);
  localStorage.setItem('cineworld_history', JSON.stringify(history));
  updateSearchTags();
}

function updateSearchTags() {
  const container = document.getElementById('recentSearchTags');
  if (!container) return;
  const history = JSON.parse(localStorage.getItem('cineworld_history') || '[]');
  if (history.length === 0) {
      document.getElementById('recentSearches').style.display = 'none';
      return;
  }
  document.getElementById('recentSearches').style.display = 'flex';
  container.innerHTML = history.map(h => `<span class="rs-tag" onclick="quickSearch('${h.replace(/'/g,"\\'")}')">${h}</span>`).join('');
}

function quickSearch(q) {
    const input = document.getElementById('globalSearch');
    input.value = q;
    liveSearch(q);
    input.focus();
}

// ── Dataset Collection (CSV) ──────────────────────────────────
async function downloadCSV() {
  const container = document.getElementById('movieGridView');
  const cards = container.querySelectorAll('.movie-card');
  if (!cards.length) return alert("Please open a 'See All' view first to collect data.");

  let csv = 'Title,Year,Rating,TMDB_ID\n';
  // We'll fetch the titles/data from the cards themselves or use a stored list
  // For simplicity, we'll extract from the DOM or re-fetch current grid page
  // A better way is to use the results from the last grid load
  const data = await api(`${gridEndpoint}${gridEndpoint.includes('?')?'&':'?'}page=${gridPage}`);
  (data.results || []).forEach(m => {
    const title = (m.title || m.name || '').replace(/,/g, '');
    const year  = (m.release_date || m.first_air_date || '').slice(0,4);
    csv += `${title},${year},${m.vote_average || 0},${m.id}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.setAttribute('href', url);
  a.setAttribute('download', `cineworld_dataset_${new Date().getTime()}.csv`);
  a.click();
}

// ── Navbar scroll solid ───────────────────────────────────────
function setupNavbar() {
  window.addEventListener('scroll', () => {
    document.getElementById('navbar').classList.toggle('solid', scrollY > 60);
  }, { passive: true });
}

// ── Genres ────────────────────────────────────────────────────
async function loadGenres() {
  try {
    const data = await api('genre/movie/list');
    const inner = document.getElementById('genreInner');
    inner.innerHTML = '<button class="genre-pill active" onclick="filterGenre(\'\',this)">All</button>';
    data.genres.forEach(g => {
      inner.innerHTML += `<button class="genre-pill" onclick="filterGenre(${g.id},this)">${g.name}</button>`;
    });
  } catch {}
}

function filterGenre(id, btn) {
  const allBtn = document.querySelector('.genre-pill[onclick*="\'\'"]');
  
  if (id === '') {
    selectedGenreIds = [];
    document.querySelectorAll('.genre-pill').forEach(p => p.classList.remove('active'));
    if (btn) btn.classList.add('active');
  } else {
    if (selectedGenreIds.includes(id)) {
      selectedGenreIds = selectedGenreIds.filter(gid => gid !== id);
      btn.classList.remove('active');
    } else {
      selectedGenreIds.push(id);
      btn.classList.add('active');
    }
    
    if (selectedGenreIds.length === 0) {
      if (allBtn) allBtn.classList.add('active');
    } else {
      if (allBtn) allBtn.classList.remove('active');
    }
  }
  reloadAllRows();
}

function onLangChange() {
  currentLang = document.getElementById('langFilter').value;
  reloadAllRows();
}

function reloadAllRows() {
  const today = getToday();
  loadRow('trending/movie/day',   'trendingCards',   true);
  loadRow(`discover/movie?sort_by=popularity.desc&release_date.lte=${today}`, 'nowPlayingCards', true);
  loadRow('discover/movie?with_watch_monetization_types=free&watch_region=IN', 'freeCards');
  loadRow(`discover/movie?sort_by=popularity.desc&release_date.gte=${today}`, 'upcomingCards', true);
  loadRow('discover/movie?sort_by=vote_average.desc&vote_count.gte=150', 'topRatedCards');
  loadRow('trending/tv/day',      'tvTrendingCards', true, 'tv');
  loadRow('tv/popular',           'tvPopularCards',  false, 'tv');
  
  const ids = ['bollywoodRow','tollywoodRow','kollywoodRow','mollywoodRow','sandalwoodRow','koreanRow','kdramaRow','japaneseRow','frenchRow','spanishRow','tvTrendingRow','tvPopularRow','actionRow','comedyRow','scifiRow','horrorRow'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = currentLang ? 'none' : 'block';
  });

  if (!currentLang) {
    loadLangRow('hi','bollywoodCards');
    loadLangRow('te','tollywoodCards');
    loadLangRow('ta','kollywoodCards');
    loadLangRow('ml','mollywoodCards');
    loadLangRow('kn','sandalwoodCards');
    loadLangRow('ko','koreanCards');
    loadLangRow('ko','kdramaCards', 'tv');
    loadLangRow('ja','japaneseCards');
    loadLangRow('fr','frenchCards');
    loadLangRow('es','spanishCards');
    loadRow('discover/movie?with_genres=28',   'actionCards');
    loadRow('discover/movie?with_genres=35',   'comedyCards');
    loadRow('discover/movie?with_genres=878',  'scifiCards');
    loadRow('discover/movie?with_genres=27',   'horrorCards');
  }
}

// ── Load a movie row ──────────────────────────────────────────
async function loadRow(endpoint, containerId, showBadge = false, type = 'movie') {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = Array(8).fill('<div class="skeleton-card"></div>').join('');
  try {
    let url = endpoint;
    const params = [];
    if (selectedGenreIds.length) params.push(`with_genres=${selectedGenreIds.join(',')}`);
    if (currentLang) params.push(`with_original_language=${currentLang}`);
    if (params.length) url += (url.includes('?') ? '&' : '?') + params.join('&');
    
    const data = await api(url);
    container.innerHTML = '';
    (data.results || []).slice(0, 20).forEach(m => {
      // Force media_type if not present from the API
      if (!m.media_type) m.media_type = type;
      container.appendChild(createCard(m, showBadge));
    });
  } catch { container.innerHTML = '<p style="color:#666;padding:12px">Failed to load content.</p>'; }
}

async function loadLangRow(lang, containerId, type = 'movie') {
  const container = document.getElementById(containerId);
  container.innerHTML = Array(8).fill('<div class="skeleton-card"></div>').join('');
  try {
    let url = `discover/${type}?with_original_language=${lang}&sort_by=popularity.desc`;
    if (selectedGenreIds.length) url += `&with_genres=${selectedGenreIds.join(',')}`;
    const data = await api(url);
    container.innerHTML = '';
    (data.results || []).forEach(m => {
      if (!m.media_type) m.media_type = type;
      container.appendChild(createCard(m));
    });
  } catch {}
}

// ── Create movie card ─────────────────────────────────────────
function createCard(movie, showNewBadge = false) {
  const div = document.createElement('div');
  div.className = 'movie-card';
  const poster = movie.poster_path ? `${IMG_BASE}w342${movie.poster_path}` : '';
  const rating = movie.vote_average ? movie.vote_average.toFixed(1) : '?';
  const year   = (movie.release_date || movie.first_air_date || '').slice(0, 4);
  const title  = movie.title || movie.name || 'Unknown';
  const isNew  = showNewBadge && year == new Date().getFullYear();

  div.innerHTML = `
    ${poster ? `<img class="card-poster" src="${poster}" alt="${title}" loading="lazy"/>` : `<div class="card-poster" style="background:#1a1a1a;display:flex;align-items:center;justify-content:center;font-size:3rem">🎬</div>`}
    <div class="card-rating-badge">⭐ ${rating}</div>
    ${isNew ? '<div class="card-new-badge">NEW</div>' : ''}
    <div class="card-overlay" onclick="event.stopPropagation();openMovie(${movie.id},'${movie.media_type||'movie'}')">
      <button class="card-play-btn" onclick="event.stopPropagation();playTrailer(${movie.id},'${movie.media_type||'movie'}')">▶</button>
      <div class="card-title-ov">${title}</div>
      <div class="card-meta-ov">${year} • ⭐${rating}</div>
    </div>
    <div class="card-body">
      <div class="card-name" title="${title}">${title}</div>
      <div class="card-sub">${year}</div>
    </div>`;
  div.onclick = () => openMovie(movie.id, movie.media_type || 'movie');
  return div;
}

// ── Hero Banner ───────────────────────────────────────────────
async function loadHeroBanner() {
  try {
    const data = await api('trending/movie/day');
    heroMovies = (data.results || []).slice(0, 8);
    renderHero(0);
    startHeroTimer();
  } catch {}
}

function startHeroTimer() {
  if (heroTimer) clearInterval(heroTimer);
  heroTimer = setInterval(() => { 
    heroIndex = (heroIndex + 1) % heroMovies.length; 
    renderHero(heroIndex); 
  }, 8000);
}

function renderHero(index) {
  const m = heroMovies[index];
  if (!m) return;
  const banner = document.getElementById('heroBanner');
  const inner  = document.getElementById('heroContent');
  const backdrop = m.backdrop_path ? `${IMG_BASE}w1280${m.backdrop_path}` : '';
  
  // Set background with zero-delay transition
  banner.style.backgroundImage = backdrop ? `url(${backdrop})` : 'none';
  banner.style.backgroundSize = 'cover';
  banner.style.backgroundPosition = 'center';

  const rating = m.vote_average ? m.vote_average.toFixed(1) : '?';
  const year   = (m.release_date || '').slice(0, 4);

  inner.innerHTML = `
    <div class="hero-tag">🔥 Trending #${index + 1}</div>
    <h2 class="hero-title">${m.title || m.name}</h2>
    <div class="hero-meta">
      <span class="hero-rating">⭐ ${rating}</span>
      <span class="hero-year">${year}</span>
    </div>
    <p class="hero-overview">${m.overview || ''}</p>
    <div class="hero-btns">
      <button class="btn-play" onclick="playTrailer(${m.id})">▶ Play Trailer</button>
      <button class="btn-info" onclick="openMovie(${m.id},'movie')">ℹ More Info</button>
    </div>
    <div class="hero-image-play" onclick="playTrailer(${m.id})">▶</div>
  `;
}

async function playTrailer(id, type = 'movie') {
  const banner = document.getElementById('heroBanner');
  if (!banner) return;
  
  if (heroTimer) {
     clearInterval(heroTimer);
     heroTimer = null;
  }

  const btns = banner.querySelector('.hero-btns');
  const originalBtns = btns ? btns.innerHTML : '';
  if (btns) btns.innerHTML = `<div style="display:flex;align-items:center;padding:12px 24px;background:rgba(229,9,20,0.8);border-radius:12px;color:#fff;font-weight:700;gap:10px;box-shadow:0 10px 30px rgba(0,0,0,0.5)"><div class="spinner" style="width:20px;height:20px;border-width:2px;border-top-color:#fff"></div> Connecting Official Result...</div>`;

  try {
    const details = await api(`${type}/${id}`);
    const activeYTKey = document.getElementById('ytApiKeyInput')?.value.trim() || YT_API_KEY;
    let vId = null;
    
    try {
      const vData = await api(`${type}/${id}/videos`);
      const v = (vData.results || []).find(x => x.site === 'YouTube' && x.type === 'Trailer') || (vData.results || []).find(x => x.site === 'YouTube');
      if (v) vId = v.key;
    } catch(e) {}

    if (!vId && activeYTKey) {
      const q = encodeURIComponent(`${details.title || details.name} official trailer hd`);
      const res = await fetch(`https://www.googleapis.com/youtube/v3/search?part=snippet&q=${q}&key=${activeYTKey}&maxResults=1&type=video`);
      const d = await res.json();
      if (d.items && d.items[0]) vId = d.items[0].id.videoId;
    }

    if (vId) {
      // 100% DIRECT ACTION: Skipping intermediate hubs for instant results
      window.open(`https://www.youtube.com/watch?v=${vId}`, '_blank', 'noopener,noreferrer');
      if (btns) btns.innerHTML = originalBtns;
      startHeroTimer(); // Resume auto-slide
    } else { throw new Error(); }
  } catch(e) { 
    if (btns) btns.innerHTML = originalBtns; 
    openMovie(id, type); 
    startHeroTimer();
  }
}

// Helper removed as direct YouTube mode is now active
function closeHeroTrailer() {
  const o = document.getElementById('heroVideoOverlay');
  if (o) o.remove();
  startHeroTimer();
}

// ── Open Movie Detail Modal ───────────────────────────────────
async function openMovie(id, type = 'movie', autoPlay = false) {
  const overlay = document.getElementById('modalOverlay');
  if (!overlay) return console.error('Modal overlay not found');
  
  if (type === 'tv') type = 'tv';
  else type = 'movie';

  overlay.classList.add('open');
  document.body.style.overflow = 'hidden'; // Prevent scroll
  document.getElementById('modalInner').innerHTML = `
    <div style="height:350px;display:flex;align-items:center;justify-content:center">
      <div class="spinner"></div>
    </div>`;
  try {
    const [details, credits, videos, providers, similar] = await Promise.all([
      api(`${type}/${id}${type==='movie'?'?append_to_response=release_dates':'?append_to_response=content_ratings'}`),
      api(`${type}/${id}/credits`),
      api(`${type}/${id}/videos`),
      api(`${type}/${id}/watch/providers`),
      api(`${type}/${id}/similar`),
    ]);
    
    // Sync the YouTube API key from both storage and the input field for maximum reliability
    const activeYTKey = document.getElementById('ytApiKeyInput')?.value.trim() || YT_API_KEY;
    
    // Always attempt YouTube Search using your API key as priority
    let ytVideoId = null;
    if (activeYTKey) {
      const query = `${details.title || details.name} ${type} teaser trailer`;
      try {
        const ytRes = await fetch(`https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(query)}&key=${activeYTKey}&maxResults=1&type=video`);
        const ytData = await ytRes.json();
        
        // Detailed logging to verify API usage
        if (ytData.error) {
           console.error("YouTube API Error:", ytData.error);
        } else if (ytData.items && ytData.items[0]) {
           ytVideoId = ytData.items[0].id.videoId;
           console.log("Direct YouTube API Success! Found video:", ytVideoId);
        } else {
           console.warn("YouTube API returned no results for:", query);
        }
      } catch(e) { 
        console.error("Critical YouTube API connectivity error:", e); 
      }
    }

    renderModal(details, credits, videos, providers, similar, type, autoPlay, ytVideoId);
  } catch(e) {
    document.getElementById('modalInner').innerHTML = `
      <div style="padding:60px;text-align:center;color:var(--text3);display:flex;flex-direction:column;align-items:center;gap:20px">
        <div style="font-size:4rem;opacity:0.5">🎬</div>
        <div>
          <h3 style="color:#fff;margin-bottom:8px">Details Unavailable</h3>
          <p>We encountered an error loading this title's information. Please check your connection or API key.</p>
        </div>
        <button class="modal-btn mbt-watch" onclick="closeModalBtn()">Return to Discovery</button>
      </div>`;
  }
}

function renderModal(m, credits, videos, providers, similar, type, autoPlay, searchVideoId = null) {
  const backdropUrl = m.backdrop_path ? `${IMG_BASE}w1280${m.backdrop_path}` : '';
  const posterUrl   = m.poster_path   ? `${IMG_BASE}w342${m.poster_path}`   : '';
  const rating  = m.vote_average ? m.vote_average.toFixed(1) : 'N/A';
  const runtime = m.runtime ? `${Math.floor(m.runtime/60)}h ${m.runtime%60}m` : '';
  const year    = (m.release_date || m.first_air_date || '').slice(0, 4);
  const genres  = (m.genres || []).map(g => `<span class="modal-pill pill-genre">${g.name}</span>`).join('');

  // Trailer logic: PRIORitize YouTube API results for "Direct" playback using user's key
  const vidResults = videos.results || [];
  const tmdbVideo = vidResults.find(v => v.type === 'Trailer' && v.site === 'YouTube')
              || vidResults.find(v => v.type === 'Teaser' && v.site === 'YouTube')
              || vidResults.find(v => v.site === 'YouTube');
  
  // Use searchVideoId (YouTube API result) FIRST if available, fallback to TMDB
  const videoId = searchVideoId || (tmdbVideo ? tmdbVideo.key : null);

  let trailerHTML = '';
  if (videoId) {
    // CINEWORLD STYLE PLAYER: Standard embed logic with a customized preview banner
    // This allows the user to trigger the player securely without seeing Error 153 on-load.
    trailerHTML = `
      <div class="trailer-wrap" id="modalTrailerWrap" style="position:relative;background:#000;aspect-ratio:16/9;overflow:hidden;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.8);border:1px solid rgba(255,255,255,0.05);cursor:pointer" onclick="window.open('https://www.youtube.com/watch?v=${videoId}', '_blank')">
         <img src="${backdropUrl}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.6;filter:blur(1px)"/>
         <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.2)">
            <div class="modal-banner-play-btn" style="width:70px;height:70px;background:var(--red);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;color:#fff;box-shadow:0 0 30px rgba(229,9,20,0.6);animation:pulseGlow 2s infinite;padding-left:5px">▶</div>
         </div>
         <div style="position:absolute;bottom:15px;left:15px;color:#fff;font-weight:900;font-size:0.75rem;letter-spacing:2px;background:rgba(0,0,0,0.5);padding:6px 14px;border-radius:20px;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1)">OFFICIAL TRAILER (HD)</div>
      </div>`;
  } else if (backdropUrl) {
    trailerHTML = `<div class="trailer-wrap"><img class="modal-back-img" src="${backdropUrl}" alt="${m.title}" style="opacity:1;height:100%"/></div>`;
  } else {
    // Replaced generic error symbol with Movie Title and Poster as requested
    trailerHTML = `
      <div class="trailer-wrap" style="position:relative;background:#111;aspect-ratio:16/9;overflow:hidden;border-radius:12px;display:flex;align-items:center;justify-content:center;box-shadow:inset 0 0 100px #000">
        ${posterUrl ? `<img src="${posterUrl}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.2;filter:blur(8px)"/>` : ''}
        <div style="position:relative;z-index:2;text-align:center;padding:20px;display:flex;flex-direction:column;align-items:center;gap:15px">
          ${posterUrl ? `<img src="${posterUrl}" style="width:110px;height:160px;object-fit:cover;border-radius:10px;box-shadow:0 15px 35px rgba(0,0,0,0.8);border:1px solid rgba(255,255,255,0.1)"/>` : '<div style="font-size:4rem">🎬</div>'}
          <div>
            <h3 style="margin:0;font-size:1.6rem;color:#fff;font-family:'Outfit',sans-serif;font-weight:800">${m.title || m.name}</h3>
            <div style="margin-top:5px;display:flex;align-items:center;justify-content:center;gap:10px;color:var(--accent);font-weight:700;font-size:0.9rem">
               <span>${year}</span> • <span>⭐ ${rating}</span>
            </div>
          </div>
          <div style="background:rgba(255,255,255,0.05);padding:6px 16px;border-radius:20px;font-size:0.8rem;color:var(--text3);border:1px solid rgba(255,255,255,0.1)">Official Trailer Currently Unavailable</div>
        </div>
      </div>`;
  }

  // Cast analysis
  const cast = (credits.cast || []).slice(0, 15);
  const director = (credits.crew || []).find(c => c.job === 'Director');
  const males   = cast.filter(a => a.gender === 2).slice(0, 3);
  const females = cast.filter(a => a.gender === 1).slice(0, 3);

  function castCard(person, role) {
    const photo = person.profile_path ? `${IMG_BASE}w185${person.profile_path}` : '';
    const labels = { hero:'Hero', heroine:'Heroine', director:'Director', producer:'Producer', cast:'Cast' };
    const classes= { hero:'role-hero', heroine:'role-heroine', director:'role-director', producer:'role-director', cast:'role-cast' };
    return `<div class="cast-card" onclick="searchPerson('${(person.name||'').replace(/'/g,"\\'")}')">
      ${photo ? `<img class="cast-photo" src="${photo}" alt="${person.name}" loading="lazy"/>` : `<div class="cast-photo" style="display:flex;align-items:center;justify-content:center;font-size:2rem;background:#222">👤</div>`}
      <span class="cast-role-badge ${classes[role]}">${labels[role]}</span>
      <div class="cast-name">${person.name}</div>
      <div class="cast-char">${person.character||person.job||''}</div>
    </div>`;
  }

  let castHTML = '';
  if (director) castHTML += castCard(director, 'director');
  males.forEach(a   => { castHTML += castCard(a, 'hero'); });
  females.forEach(a => { castHTML += castCard(a, 'heroine'); });
  cast.filter(a => !males.includes(a) && !females.includes(a)).slice(0,6).forEach(a => { castHTML += castCard(a, 'cast'); });

  // Streaming providers: Aggregate from ALL global regions for maximum discovery
  const watchData = providers.results || {};
  const streamOptions = [];
  
  Object.values(watchData).forEach(region => {
    if (region.flatrate) region.flatrate.forEach(p => streamOptions.push({...p, type: 'Subscription'}));
    if (region.free) region.free.forEach(p => streamOptions.push({...p, type: 'Free'}));
    if (region.ads) region.ads.forEach(p => streamOptions.push({...p, type: 'With Ads'}));
    if (region.rent) region.rent.forEach(p => streamOptions.push({...p, type: 'Rent'}));
    if (region.buy) region.buy.forEach(p => streamOptions.push({...p, type: 'Buy'}));
  });

  // Remove duplicates
  const uniqueProviders = [];
  const seenIds = new Set();
  streamOptions.forEach(p => {
    if (!seenIds.has(p.provider_id)) {
      uniqueProviders.push(p);
      seenIds.add(p.provider_id);
    }
  });

  let streamHTML = '';
  if (uniqueProviders.length) {
    streamHTML = uniqueProviders.slice(0, 10).map(p => {
      const logo = p.logo_path ? `${IMG_BASE}w92${p.logo_path}` : '';
      let typeLabel = p.type;
      let typeClass = p.type === 'Free' ? 'wp-type free' : 'wp-type';
      return `<div class="wp-item">
        ${logo ? `<img class="wp-logo" src="${logo}" alt="${p.provider_name}"/>` : ''}
        <div><div class="wp-name">${p.provider_name}</div><div class="${typeClass}">${typeLabel}</div></div>
      </div>`;
    }).join('');
  } else {
    const isReleased = m.status === 'Released' || (m.release_date && new Date(m.release_date) < new Date());
    const query = encodeURIComponent(`${m.title || m.name} ${year} where to watch streaming digital`);
    
    streamHTML = `
      <div class="no-stream-container">
        <div class="no-stream-content">
          <div class="no-stream-icon">⚡️</div>
          <div class="no-stream-text">
            <h4>${isReleased ? 'Global Provider Data Unavailable' : 'Coming Soon to Streaming'}</h4>
            <p>${isReleased ? 'We couldn\'t find active streaming links in our current provider database.' : 'This title hasn\'t been released for digital streaming yet.'}</p>
          </div>
        </div>
        <div class="no-stream-actions">
          <button class="ns-btn google" onclick="window.open('https://www.google.com/search?q=${query}','_blank')">🌐 Search Google</button>
          <button class="ns-btn justwatch" onclick="window.open('https://www.justwatch.com/search?q=${encodeURIComponent(m.title || m.name)}','_blank')">🍿 Check JustWatch</button>
        </div>
      </div>`;
  }


  // Similar movies
  const simCards = (similar.results || []).slice(0, 10).map(s => {
    const sPost = s.poster_path ? `${IMG_BASE}w154${s.poster_path}` : '';
    return `<div class="movie-card" style="--card-w:120px;--card-h:180px" onclick="openMovie(${s.id},'${type}')">
      ${sPost ? `<img class="card-poster" src="${sPost}" loading="lazy"/>` : `<div class="card-poster" style="background:#222;display:flex;align-items:center;justify-content:center;font-size:2rem">🎬</div>`}
      <div class="card-body"><div class="card-name" title="${s.title||s.name}">${s.title||s.name}</div></div>
    </div>`;
  }).join('');

  // Budget / Revenue
  const budget  = m.budget  ? `$${(m.budget/1e6).toFixed(1)}M`  : 'N/A';
  const revenue = m.revenue ? `$${(m.revenue/1e6).toFixed(1)}M` : 'N/A';
  const lang    = LANGUAGES.find(l => l.code === m.original_language);
  const langName= lang ? lang.name : m.original_language;

  document.getElementById('modalInner').innerHTML = `
    ${trailerHTML}
    <div class="modal-body">
      <div class="modal-top">
        ${posterUrl ? `<img class="modal-poster-img" src="${posterUrl}" alt="${m.title}"/>` : ''}
        <div class="modal-top-info">
          <h2 class="modal-title">${m.title || m.name}</h2>
          ${m.tagline ? `<div class="modal-tagline">"${m.tagline}"</div>` : ''}
          <div class="modal-pills">
            ${genres}
            <span class="modal-pill pill-lang">🗣️ ${langName}</span>
            <span class="modal-pill pill-rating">⭐ ${rating}/10 (${(m.vote_count||0).toLocaleString()} votes)</span>
            ${m.status ? `<span class="modal-pill pill-status">${m.status}</span>` : ''}
          </div>
          <div class="modal-meta-row">
            ${year ? `<span><strong>Year:</strong> ${year}</span>` : ''}
            ${runtime ? `<span><strong>Runtime:</strong> ${runtime}</span>` : ''}
            ${m.popularity ? `<span><strong>Popularity:</strong> ${m.popularity.toFixed(0)}</span>` : ''}
            <span><strong>Budget:</strong> ${budget}</span>
            <span><strong>Revenue:</strong> ${revenue}</span>
          </div>
          <div class="modal-action-btns">
            ${videoId ? `<button class="modal-btn mbt-trailer" onclick="startModalTrailer('${videoId}')">▶ Play ${tmdbVideo && tmdbVideo.type === 'Teaser' ? 'Teaser' : 'Trailer'}</button>` : ''}
            <button class="modal-btn mbt-watch" onclick="window.open('https://www.themoviedb.org/${type}/${m.id}','_blank')">🔗 More Details</button>
            ${m.homepage ? `<button class="modal-btn mbt-watch" onclick="window.open('${m.homepage}','_blank')">🌐 Official Site</button>` : ''}
          </div>
        </div>
      </div>
      <p class="modal-overview">${m.overview || 'No description available.'}</p>

      ${castHTML ? `<div class="modal-section-title">🎭 Cast & Crew</div><div class="cast-grid">${castHTML}</div>` : ''}

      <div class="modal-section-title">📺 Where to Watch</div>
      <div class="watch-providers">${streamHTML}</div>

      ${simCards ? `<div class="modal-section-title">🎬 Similar Movies</div><div class="similar-row">${simCards}</div>` : ''}
    </div>`;
}

function startModalTrailer(vId) {
  // 100% DIRECT ACTION: Bypasses local computer blocks
  window.open(`https://www.youtube.com/watch?v=${vId}`, '_blank', 'noopener,noreferrer');
}

function closeModal(e) { if (e.target === e.currentTarget) closeModalBtn(); }
function closeModalBtn() {
  document.getElementById('modalOverlay').classList.remove('open');
  document.getElementById('modalInner').innerHTML = '';
  document.body.style.overflow = 'auto'; // Restore scroll
}

// ── Search ────────────────────────────────────────────────────
function setupSearch() {
  const input = document.getElementById('globalSearch');
  const dropdown = document.getElementById('searchResults');

  input.addEventListener('input', function() {
    clearTimeout(searchTimer);
    const q = this.value.trim();
    if (!q) { dropdown.classList.remove('open'); dropdown.innerHTML = ''; return; }
    searchTimer = setTimeout(() => {
        liveSearch(q);
        logSearch(q); // Save search locally
    }, 400);
  });

  document.addEventListener('click', e => {
    if (!e.target.closest('.search-bar')) dropdown.classList.remove('open');
  });
}

async function liveSearch(query) {
  const dropdown = document.getElementById('searchResults');
  try {
    const data = await api(`search/multi?query=${encodeURIComponent(query)}&include_adult=false`);
    const results = (data.results || []).filter(r => r.media_type !== 'person').slice(0, 8);
    dropdown.innerHTML = results.map(m => {
      const poster = m.poster_path ? `${IMG_BASE}w92${m.poster_path}` : '';
      const title  = m.title || m.name;
      const year   = (m.release_date || m.first_air_date || '').slice(0, 4);
      const rating = m.vote_average ? `⭐${m.vote_average.toFixed(1)}` : '';
      return `<div class="search-item" onclick="openMovie(${m.id},'${m.media_type||'movie'}');document.getElementById('searchResults').classList.remove('open')">
        ${poster ? `<img class="si-poster" src="${poster}" alt="${title}"/>` : `<div class="si-poster" style="background:#222;display:flex;align-items:center;justify-content:center">🎬</div>`}
        <div class="si-info">
          <div class="si-title" style="display:flex;align-items:center;justify-content:space-between;gap:10px">
             <span>${title}</span>
             <button class="si-quick-play" onclick="event.stopPropagation();playTrailer(${m.id},'${m.media_type||'movie'}')">▶</button>
          </div>
          <div class="si-meta">${year} ${rating} · ${m.media_type === 'tv' ? '📺 TV' : '🎬 Movie'}</div>
        </div>
      </div>`;
    }).join('');
    dropdown.classList.add('open');
    
    // Update global counter
    const counter = document.getElementById('resultsCount');
    if (counter && data.total_results) {
      counter.textContent = data.total_results.toLocaleString() + ' Found';
      counter.style.background = 'rgba(16,185,129,0.1)';
      counter.style.borderColor = 'rgba(16,185,129,0.3)';
      counter.style.color = '#10b981';
    }
  } catch {}
}

function searchPerson(name) {
  document.getElementById('globalSearch').value = name;
  liveSearch(name);
}

// ── See All Grid ──────────────────────────────────────────────
async function seeAll(endpoint) {
  gridEndpoint = endpoint;
  gridPage = 1;
  document.getElementById('gridSection').style.display = 'block';
  const title = endpoint.split('/').pop().replace(/_/g,' ').replace('?','').toUpperCase();
  document.getElementById('gridTitle').textContent = '🎬 ' + title;
  document.getElementById('gridSection').scrollIntoView({ behavior: 'smooth' });
  await loadGrid();
}

async function loadGrid() {
  const container = document.getElementById('movieGridView');
  container.innerHTML = Array(20).fill('<div class="skeleton-card"></div>').join('');
  try {
    const year  = document.getElementById('gridYear').value;
    const sort  = document.getElementById('gridSort').value;
    let ep = gridEndpoint;
    const extras = [];
    if (year) extras.push(`primary_release_year=${year}`);
    if (sort && !ep.includes('trending')) extras.push(`sort_by=${sort}`);
    if (selectedGenreIds.length) extras.push(`with_genres=${selectedGenreIds.join(',')}`);
    if (currentLang) extras.push(`with_original_language=${currentLang}`);
    extras.push(`page=${gridPage}`);
    if (extras.length) ep += (ep.includes('?') ? '&' : '?') + extras.join('&');

    const data = await api(ep);
    container.innerHTML = '';
    (data.results || []).forEach(m => container.appendChild(createCard(m, false)));

    // Update global counter
    const counter = document.getElementById('resultsCount');
    if (counter && data.total_results) {
      counter.textContent = data.total_results.toLocaleString() + ' in ' + document.getElementById('gridTitle').textContent.replace('🎬 ','');
      counter.style.background = 'rgba(168,85,247,0.1)';
      counter.style.borderColor = 'rgba(168,85,247,0.3)';
      counter.style.color = '#a855f7';
    }

    // Pagination
    renderPagination(data.page, data.total_pages);
  } catch { container.innerHTML = '<p style="color:#666;padding:20px">Failed to load.</p>'; }
}

function renderPagination(page, total) {
  const pg = document.getElementById('pagination');
  const max = Math.min(total, 500);
  let html = '';
  if (page > 1) html += `<button class="page-btn" onclick="goPage(${page-1})">← Prev</button>`;
  const start = Math.max(1, page - 3);
  const end   = Math.min(max, page + 3);
  for (let p = start; p <= end; p++) {
    html += `<button class="page-btn ${p===page?'active':''}" onclick="goPage(${p})">${p}</button>`;
  }
  if (page < max) html += `<button class="page-btn" onclick="goPage(${page+1})">Next →</button>`;
  pg.innerHTML = html;
}

function goPage(p) {
  gridPage = p;
  loadGrid();
  document.getElementById('gridSection').scrollIntoView({ behavior: 'smooth' });
}

function applyGridFilters() { 
  gridPage = 1; 
  loadGrid(); 
}

function closeGrid() {
  document.getElementById('gridSection').style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function loadSection(type) {
  const today = new Date().toISOString().split('T')[0];
  const map = {
    trending:    'trending/movie/day',
    now_playing: 'discover/movie?sort_by=popularity.desc&release_date.lte='+today,
    upcoming:    'discover/movie?sort_by=popularity.desc&release_date.gte='+today,
    top_rated:   'discover/movie?sort_by=vote_average.desc&vote_count.gte=200',
    free:        'discover/movie?with_watch_monetization_types=free&watch_region=IN&sort_by=popularity.desc',
    tv_trending: 'trending/tv/day',
    tv_popular:  'tv/popular'
  };
  if (map[type]) {
    // If it's the Top Rated button, set the sorting dropdown to 'Rating' automatically
    if (type === 'top_rated') document.getElementById('gridSort').value = 'vote_average.desc';
    else document.getElementById('gridSort').value = 'popularity.desc';
    seeAll(map[type]);
  }
}

function scrollToTop() { window.scrollTo({ top: 0, behavior: 'smooth' }); }

// ── Loader ────────────────────────────────────────────────────
function showLoader() { document.getElementById('globalLoader').style.display = 'flex'; }
function hideLoader() { document.getElementById('globalLoader').style.display = 'none'; }

// Allow Enter key on setup screen
document.addEventListener('DOMContentLoaded', () => {
  const inp = document.getElementById('apiKeyInput');
  if (inp) inp.addEventListener('keydown', e => { if (e.key === 'Enter') launchApp(); });
});
