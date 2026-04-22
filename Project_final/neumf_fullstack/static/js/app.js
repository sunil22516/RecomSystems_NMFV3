const state = {
  strategies: ["popularity"],
  currentUser: null,
};

const els = {
  datasetStats: document.getElementById("dataset-stats"),
  status: document.getElementById("status"),
  userId: document.getElementById("user-id"),
  randomUserBtn: document.getElementById("random-user-btn"),
  strategy: document.getElementById("strategy"),
  kRange: document.getElementById("k-range"),
  kValue: document.getElementById("k-value"),
  surpriseMode: document.getElementById("surprise-mode"),
  diversityRange: document.getElementById("diversity-range"),
  diversityValue: document.getElementById("diversity-value"),
  freshnessRange: document.getElementById("freshness-range"),
  freshnessValue: document.getElementById("freshness-value"),
  recoForm: document.getElementById("reco-form"),
  resultsGrid: document.getElementById("results-grid"),
  resultCount: document.getElementById("result-count"),
  historyStrip: document.getElementById("history-strip"),
  historyCount: document.getElementById("history-count"),
  searchBox: document.getElementById("search-box"),
  searchBtn: document.getElementById("search-btn"),
  searchGrid: document.getElementById("search-grid"),
  searchCount: document.getElementById("search-count"),
  movieCardTemplate: document.getElementById("movie-card-template"),
};

function setStatus(text, level = "info") {
  els.status.textContent = text;
  els.status.classList.remove("status-error", "status-success");
  if (level === "error") {
    els.status.classList.add("status-error");
  }
  if (level === "success") {
    els.status.classList.add("status-success");
  }
}

async function fetchJson(path, params = {}) {
  const query = new URLSearchParams(params);
  const url = query.size ? `${path}?${query.toString()}` : path;
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function setStrategies(strategies) {
  state.strategies = strategies.length ? strategies : ["popularity"];
  els.strategy.innerHTML = "";

  for (const strategy of state.strategies) {
    const option = document.createElement("option");
    option.value = strategy;
    option.textContent = strategy.toUpperCase();
    els.strategy.appendChild(option);
  }

  if (state.strategies.includes("hybrid")) {
    els.strategy.value = "hybrid";
  }
}

function renderStats(stats) {
  const modelParts = [];

  if (stats.models?.gmf?.loaded) {
    modelParts.push("GMF");
  }
  if (stats.models?.mlp?.loaded) {
    modelParts.push("MLP");
  }
  if (!modelParts.length) {
    modelParts.push("Popularity Fallback");
  }

  els.datasetStats.innerHTML = `
    <p class="mono">Users: ${stats.n_users} | Items: ${stats.n_items}</p>
    <p class="mono">Available users: ${stats.available_users}</p>
    <p class="mono">Engines: ${modelParts.join(" + ")}</p>
  `;
}

function clearElement(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function makeEmptyState(message) {
  const block = document.createElement("div");
  block.className = "empty-state";
  block.textContent = message;
  return block;
}

function createChip(text) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = text;
  return chip;
}

function createMovieCard(item, rank = null) {
  const node = els.movieCardTemplate.content.cloneNode(true);
  const card = node.querySelector(".movie-card");
  const badge = node.querySelector(".rank-badge");
  const title = node.querySelector(".movie-title");
  const meta = node.querySelector(".movie-meta");
  const score = node.querySelector(".movie-score");
  const chips = node.querySelector(".chips");

  if (rank === null) {
    badge.textContent = "*";
  } else {
    badge.textContent = String(rank);
  }

  const cleanTitle = item.clean_title || item.title || "Untitled";
  title.textContent = cleanTitle;

  const metaParts = [];
  if (item.year) {
    metaParts.push(item.year);
  }
  if (typeof item.item_id === "number" && item.item_id >= 0) {
    metaParts.push(`Movie #${item.item_id}`);
  }
  meta.textContent = metaParts.length ? metaParts.join(" | ") : "Movie metadata unavailable";

  if (typeof item.score === "number") {
    score.textContent = `Score: ${item.score.toFixed(4)}`;
  } else {
    score.textContent = "";
  }

  const genres = Array.isArray(item.genres) ? item.genres.slice(0, 3) : [];
  for (const genre of genres) {
    chips.appendChild(createChip(genre));
  }

  if (item.imdb_url) {
    card.addEventListener("click", () => {
      window.open(item.imdb_url, "_blank", "noopener,noreferrer");
    });
    card.style.cursor = "pointer";
  }

  return node;
}

function renderHistory(items) {
  clearElement(els.historyStrip);
  if (!items.length) {
    els.historyStrip.appendChild(makeEmptyState("No history items available for this user."));
    els.historyCount.textContent = "0 items";
    return;
  }

  items.forEach((item) => {
    els.historyStrip.appendChild(createMovieCard(item, null));
  });
  els.historyCount.textContent = `${items.length} items`;
}

function renderRecommendations(items) {
  clearElement(els.resultsGrid);
  if (!items.length) {
    els.resultsGrid.appendChild(makeEmptyState("No recommendations returned."));
    els.resultCount.textContent = "0 items";
    return;
  }

  items.forEach((item, index) => {
    els.resultsGrid.appendChild(createMovieCard(item, index + 1));
  });
  els.resultCount.textContent = `${items.length} items`;
}

function renderSearchResults(items) {
  clearElement(els.searchGrid);
  if (!items.length) {
    els.searchGrid.appendChild(makeEmptyState("No titles matched your query."));
    els.searchCount.textContent = "0 items";
    return;
  }

  items.forEach((item, index) => {
    els.searchGrid.appendChild(createMovieCard(item, index + 1));
  });
  els.searchCount.textContent = `${items.length} items`;
}

async function loadStats() {
  const stats = await fetchJson("/api/stats");
  renderStats(stats);
  setStrategies(stats.strategies || []);

  if (Array.isArray(stats.popular_preview) && stats.popular_preview.length) {
    renderSearchResults(stats.popular_preview);
    els.searchCount.textContent = `${stats.popular_preview.length} popular titles`;
  }
}

async function chooseRandomUser() {
  const payload = await fetchJson("/api/users/random");
  const userId = Number(payload.user_id);
  state.currentUser = userId;
  els.userId.value = String(userId);
}

async function runRecommendation() {
  const userId = Number(els.userId.value);
  if (!Number.isInteger(userId) || userId < 0) {
    throw new Error("Please enter a valid non-negative user ID.");
  }

  const k = Number(els.kRange.value);
  const strategy = els.strategy.value;
  const randomize = Boolean(els.surpriseMode.checked);
  const diversity = Number(els.diversityRange.value) / 100;
  const noveltyPenalty = Number(els.freshnessRange.value) / 100;

  const payload = await fetchJson("/api/recommend", {
    user_id: userId,
    k,
    strategy,
    randomize,
    diversity,
    novelty_penalty: noveltyPenalty,
  });

  state.currentUser = userId;
  renderHistory(payload.history || []);
  renderRecommendations(payload.recommendations || []);

  setStatus(
    `Generated ${payload.recommendations.length} recommendations for user ${payload.user_id} using ${payload.strategy.toUpperCase()} (${payload.randomize ? "surprise" : "stable"} mode).`,
    "success"
  );
}

async function runSearch() {
  const q = els.searchBox.value.trim();
  if (!q) {
    renderSearchResults([]);
    return;
  }

  const payload = await fetchJson("/api/search", { q, limit: 16 });
  renderSearchResults(payload.items || []);
}

function bindEvents() {
  els.kRange.addEventListener("input", () => {
    els.kValue.textContent = els.kRange.value;
  });

  els.diversityRange.addEventListener("input", () => {
    const value = Number(els.diversityRange.value) / 100;
    els.diversityValue.textContent = value.toFixed(2);
  });

  els.freshnessRange.addEventListener("input", () => {
    const value = Number(els.freshnessRange.value) / 100;
    els.freshnessValue.textContent = value.toFixed(2);
  });

  els.randomUserBtn.addEventListener("click", async () => {
    try {
      setStatus("Picking random user and generating recommendations...");
      await chooseRandomUser();
      await runRecommendation();
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  els.recoForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      setStatus("Running recommendation query...");
      await runRecommendation();
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  els.searchBtn.addEventListener("click", async () => {
    try {
      await runSearch();
    } catch (error) {
      setStatus(error.message, "error");
    }
  });

  els.searchBox.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    try {
      await runSearch();
    } catch (error) {
      setStatus(error.message, "error");
    }
  });
}

async function bootstrap() {
  bindEvents();

  try {
    setStatus("Loading service metadata...");
    await loadStats();

    setStatus("Selecting an initial user...");
    await chooseRandomUser();

    await runRecommendation();
  } catch (error) {
    setStatus(`Startup issue: ${error.message}`, "error");
  }
}

bootstrap();
