from __future__ import annotations

import pickle
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 128) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.output = nn.Linear(emb_dim, 1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.output(self.user_emb(users) * self.item_emb(items)).squeeze(-1)


class MLPModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 128,
        mlp_dims: Sequence[int] = (512, 256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        layers: List[nn.Module] = []
        in_dim = emb_dim * 2
        for out_dim in mlp_dims:
            layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.predict = nn.Linear(mlp_dims[-1], 1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(users), self.item_emb(items)], dim=-1)
        return self.predict(self.mlp(x)).squeeze(-1)


class NeuMFRecommenderService:
    GENRE_COLUMNS = [
        "unknown",
        "action",
        "adventure",
        "animation",
        "childrens",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film_noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci_fi",
        "thriller",
        "war",
        "western",
    ]

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.data_dir = self.project_root / "ml-100k"
        self.device = torch.device("cpu")

        self.n_users = 0
        self.n_items = 0

        self.user_pos: Dict[int, set[int]] = {}
        self.user_history: Dict[int, List[int]] = {}
        self.user2idx: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self.item_lookup: Dict[int, Dict[str, Any]] = {}
        self.recommendable_item_ids: set[int] = set()

        self.popularity_by_item_id: Dict[int, float] = {}
        self.popularity_by_idx: Dict[int, float] = {}

        self.gmf: Optional[GMF] = None
        self.mlp: Optional[MLPModel] = None
        self.model_info: Dict[str, Dict[str, Any]] = {
            "gmf": {"loaded": False},
            "mlp": {"loaded": False},
        }

        self.available_users: List[int] = []
        # Tracks recently served item indices per user to avoid repeated suggestions.
        self.served_history_by_user: Dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=200))

        self._load_all()

    def _load_all(self) -> None:
        self._load_dataset_metadata()
        self._load_item_metadata()
        self._load_popularity()
        self._load_models()

    def _load_dataset_metadata(self) -> None:
        meta_path = self.checkpoint_dir / "dataset_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as handle:
                meta = pickle.load(handle)

            self.n_users = int(meta["n_users"])
            self.n_items = int(meta["n_items"])

            self.user_pos = {int(k): {int(x) for x in v} for k, v in meta["user_pos"].items()}
            self.user_history = {int(k): [int(x) for x in v] for k, v in meta["user_history"].items()}

            self.user2idx = {int(k): int(v) for k, v in meta["user2idx"].items()}
            self.item2idx = {int(k): int(v) for k, v in meta["item2idx"].items()}
        else:
            self._build_metadata_from_raw_ratings()

        self.idx2item = {idx: item_id for item_id, idx in self.item2idx.items()}
        self.idx2user = {idx: user_id for user_id, idx in self.user2idx.items()}

        self.recommendable_item_ids = set(self.item2idx.keys())

        users = []
        for u in range(self.n_users):
            if len(self.user_history.get(u, [])) > 0:
                users.append(u)
        self.available_users = users

    def _build_metadata_from_raw_ratings(self) -> None:
        ratings_path = self.data_dir / "u.data"
        if not ratings_path.exists():
            raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

        ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        pos = ratings[ratings["rating"] >= 3].copy()
        user_ids = sorted(int(x) for x in pos["user_id"].unique().tolist())
        item_ids = sorted(int(x) for x in pos["item_id"].unique().tolist())

        self.user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item2idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)

        self.user_pos = {u: set() for u in range(self.n_users)}
        self.user_history = {u: [] for u in range(self.n_users)}

        pos["user"] = pos["user_id"].map(self.user2idx)
        pos["item"] = pos["item_id"].map(self.item2idx)
        pos = pos.sort_values(["user", "timestamp"])

        for row in pos.itertuples(index=False):
            user_idx = int(row.user)
            item_idx = int(row.item)
            self.user_pos[user_idx].add(item_idx)
            self.user_history[user_idx].append(item_idx)

    def _load_item_metadata(self) -> None:
        item_path = self.data_dir / "u.item"
        if not item_path.exists():
            return

        names = ["item_id", "title", "release_date", "video_date", "imdb_url"] + self.GENRE_COLUMNS
        items = pd.read_csv(
            item_path,
            sep="|",
            encoding="latin-1",
            header=None,
            usecols=list(range(24)),
            names=names,
        )

        items["year"] = items["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).fillna("")
        items["clean_title"] = items["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

        for row in items.itertuples(index=False):
            item_id = int(row.item_id)
            genres: List[str] = []
            for genre in self.GENRE_COLUMNS:
                if int(getattr(row, genre)) == 1:
                    genres.append(genre.replace("_", " ").title())

            self.item_lookup[item_id] = {
                "item_id": item_id,
                "title": str(row.title),
                "clean_title": str(row.clean_title),
                "year": str(row.year) if row.year else "",
                "genres": genres,
                "imdb_url": str(row.imdb_url),
            }

    def _load_popularity(self) -> None:
        ratings_path = self.data_dir / "u.data"
        if not ratings_path.exists():
            return

        ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        positive = ratings[ratings["rating"] >= 3]
        counts = positive["item_id"].value_counts()

        if counts.empty:
            return

        max_count = float(counts.max())
        self.popularity_by_item_id = {int(item_id): float(count) / max_count for item_id, count in counts.items()}

        self.popularity_by_idx = {}
        for idx in range(self.n_items):
            item_id = self.idx2item.get(idx)
            if item_id is None:
                self.popularity_by_idx[idx] = 0.0
            else:
                self.popularity_by_idx[idx] = self.popularity_by_item_id.get(item_id, 0.0)

    def _load_models(self) -> None:
        self.gmf = self._load_gmf_model(self.checkpoint_dir / "gmf.pt")
        self.mlp = self._load_mlp_model(self.checkpoint_dir / "mlp.pt")

    def _load_gmf_model(self, checkpoint_path: Path) -> Optional[GMF]:
        if not checkpoint_path.exists():
            self.model_info["gmf"] = {"loaded": False, "reason": f"Missing {checkpoint_path.name}"}
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state = checkpoint["state_dict"]

            n_users = int(state["user_emb.weight"].shape[0])
            n_items = int(state["item_emb.weight"].shape[0])
            emb_dim = int(state["user_emb.weight"].shape[1])

            model = GMF(n_users=n_users, n_items=n_items, emb_dim=emb_dim)
            model.load_state_dict(state)
            model.eval()

            self.model_info["gmf"] = {
                "loaded": True,
                "path": str(checkpoint_path),
                "n_users": n_users,
                "n_items": n_items,
                "emb_dim": emb_dim,
            }
            return model
        except Exception as exc:
            self.model_info["gmf"] = {"loaded": False, "reason": str(exc)}
            return None

    def _load_mlp_model(self, checkpoint_path: Path) -> Optional[MLPModel]:
        if not checkpoint_path.exists():
            self.model_info["mlp"] = {"loaded": False, "reason": f"Missing {checkpoint_path.name}"}
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state = checkpoint["state_dict"]

            n_users = int(state["user_emb.weight"].shape[0])
            n_items = int(state["item_emb.weight"].shape[0])
            emb_dim = int(state["user_emb.weight"].shape[1])
            mlp_dims = self._infer_mlp_dims(state)
            dropout = float(checkpoint.get("config", {}).get("dropout", 0.1))

            model = MLPModel(
                n_users=n_users,
                n_items=n_items,
                emb_dim=emb_dim,
                mlp_dims=mlp_dims,
                dropout=dropout,
            )
            model.load_state_dict(state)
            model.eval()

            self.model_info["mlp"] = {
                "loaded": True,
                "path": str(checkpoint_path),
                "n_users": n_users,
                "n_items": n_items,
                "emb_dim": emb_dim,
                "mlp_dims": list(mlp_dims),
            }
            return model
        except Exception as exc:
            self.model_info["mlp"] = {"loaded": False, "reason": str(exc)}
            return None

    @staticmethod
    def _infer_mlp_dims(state_dict: Dict[str, torch.Tensor]) -> Sequence[int]:
        linear_layers: List[tuple[int, int]] = []
        for key, tensor in state_dict.items():
            if not key.startswith("mlp.") or not key.endswith(".weight"):
                continue
            if tensor.ndim != 2:
                continue
            try:
                layer_idx = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            linear_layers.append((layer_idx, int(tensor.shape[0])))

        linear_layers.sort(key=lambda x: x[0])
        dims = [dim for _, dim in linear_layers]
        return tuple(dims) if dims else (512, 256, 128, 64)

    def available_strategies(self) -> List[str]:
        strategies = ["popularity"]
        if self.gmf is not None:
            strategies.append("gmf")
        if self.mlp is not None:
            strategies.append("mlp")
        if self.gmf is not None or self.mlp is not None:
            strategies.insert(0, "hybrid")
        return strategies

    def get_stats(self) -> Dict[str, Any]:
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "available_users": len(self.available_users),
            "strategies": self.available_strategies(),
            "models": self.model_info,
            "popular_preview": self.get_popular_items(limit=12),
        }

    def list_users(self, limit: int = 200) -> List[Dict[str, Any]]:
        users = []
        for user_idx in self.available_users:
            users.append(
                {
                    "user_id": int(user_idx),
                    "history_size": int(len(self.user_history.get(user_idx, []))),
                }
            )
        users.sort(key=lambda row: row["history_size"], reverse=True)
        return users[:limit]

    def random_user(self) -> int:
        if not self.available_users:
            raise RuntimeError("No users are available in dataset metadata.")
        return int(random.choice(self.available_users))

    def search_titles(self, query: str, limit: int = 12) -> List[Dict[str, Any]]:
        clean = query.strip().lower()
        if not clean:
            return []

        matches: List[Dict[str, Any]] = []
        for item_id in sorted(self.recommendable_item_ids):
            payload = self.item_lookup.get(item_id)
            if payload is None:
                continue
            if clean in payload["title"].lower() or clean in payload["clean_title"].lower():
                matches.append(
                    {
                        "item_id": int(item_id),
                        "title": payload["title"],
                        "year": payload["year"],
                        "genres": payload["genres"],
                    }
                )
            if len(matches) >= limit:
                break

        return matches

    def get_popular_items(self, limit: int = 12) -> List[Dict[str, Any]]:
        if not self.popularity_by_item_id:
            return []

        ranked = sorted(self.popularity_by_item_id.items(), key=lambda row: row[1], reverse=True)
        output: List[Dict[str, Any]] = []
        for item_id, score in ranked:
            if item_id not in self.recommendable_item_ids:
                continue
            payload = self._payload_for_item_id(item_id)
            payload["score"] = round(float(score), 4)
            output.append(payload)
            if len(output) >= limit:
                break
        return output

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        strategy: str = "hybrid",
        randomize: bool = True,
        diversity: float = 0.45,
        novelty_penalty: float = 0.08,
    ) -> Dict[str, Any]:
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(f"user_id must be in [0, {self.n_users - 1}]")

        strategy = strategy.lower().strip()
        valid = set(self.available_strategies())
        if strategy not in valid:
            strategy = "hybrid" if "hybrid" in valid else "popularity"

        diversity = float(np.clip(diversity, 0.0, 1.0))
        novelty_penalty = float(np.clip(novelty_penalty, 0.0, 1.0))

        seen_items = self.user_pos.get(user_id, set())
        max_item_capacity = self._max_item_capacity(strategy)

        candidate_items = [item_idx for item_idx in range(max_item_capacity) if item_idx not in seen_items]
        if not candidate_items:
            return {
                "user_id": int(user_id),
                "strategy": strategy,
                "k": int(k),
                "recommendations": [],
                "history": self._history_payload(user_id),
                "note": "No unseen candidate items for this user.",
            }

        scores = self._score_candidates(user_id=user_id, item_indices=candidate_items, strategy=strategy)
        k = max(1, min(int(k), 50, len(candidate_items)))
        scores = self._apply_repeat_penalty(
            user_id=user_id,
            item_indices=candidate_items,
            scores=scores,
            novelty_penalty=novelty_penalty,
        )

        top_positions = self._select_positions(scores=scores, k=k, randomize=randomize, diversity=diversity)
        recommendations: List[Dict[str, Any]] = []
        for rank, position in enumerate(top_positions.tolist(), start=1):
            item_idx = int(candidate_items[position])
            item_payload = self._payload_for_item_idx(item_idx)
            item_payload["rank"] = rank
            item_payload["score"] = round(float(scores[position]), 5)
            recommendations.append(item_payload)

        self._remember_served_items(user_id=user_id, recommendations=recommendations)

        return {
            "user_id": int(user_id),
            "strategy": strategy,
            "k": int(k),
            "randomize": bool(randomize),
            "diversity": round(diversity, 3),
            "novelty_penalty": round(novelty_penalty, 3),
            "recommendations": recommendations,
            "history": self._history_payload(user_id),
        }

    def _select_positions(self, scores: np.ndarray, k: int, randomize: bool, diversity: float) -> np.ndarray:
        if not randomize or len(scores) <= k:
            return np.argsort(scores)[::-1][:k]

        # Sample from a high-score pool for diversity while keeping quality high.
        pool_size = min(len(scores), max(k + 3, int(k * (2.5 + 5.0 * diversity))))
        pool_positions = np.argsort(scores)[::-1][:pool_size]
        pool_scores = scores[pool_positions].astype(np.float64)

        temperature = 0.35 + 0.85 * diversity
        scaled = (pool_scores - float(pool_scores.max())) / max(temperature, 1e-6)
        weights = np.exp(scaled)
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()

        rng = np.random.default_rng()
        sampled_positions = rng.choice(pool_positions, size=k, replace=False, p=weights)
        sampled_scores = scores[sampled_positions]
        order = np.argsort(sampled_scores)[::-1]
        return sampled_positions[order]

    def _apply_repeat_penalty(
        self,
        user_id: int,
        item_indices: Sequence[int],
        scores: np.ndarray,
        novelty_penalty: float,
    ) -> np.ndarray:
        if novelty_penalty <= 0:
            return scores

        recent = self.served_history_by_user.get(user_id)
        if not recent:
            return scores

        recency_rank: Dict[int, int] = {}
        for rank, item_idx in enumerate(reversed(recent), start=1):
            if item_idx not in recency_rank:
                recency_rank[item_idx] = rank

        adjusted = scores.copy()
        for pos, item_idx in enumerate(item_indices):
            rank = recency_rank.get(item_idx)
            if rank is None:
                continue
            adjusted[pos] -= novelty_penalty * (1.0 / np.sqrt(float(rank)))
        return adjusted

    def _remember_served_items(self, user_id: int, recommendations: Sequence[Dict[str, Any]]) -> None:
        history = self.served_history_by_user[user_id]
        for row in recommendations:
            item_idx = row.get("item_idx")
            if isinstance(item_idx, int) and item_idx >= 0:
                history.append(item_idx)

    def _max_item_capacity(self, strategy: str) -> int:
        capacities = [self.n_items]
        if strategy in {"gmf", "hybrid"} and self.gmf is not None:
            capacities.append(int(self.gmf.item_emb.num_embeddings))
        if strategy in {"mlp", "hybrid"} and self.mlp is not None:
            capacities.append(int(self.mlp.item_emb.num_embeddings))
        return max(0, min(capacities))

    def _score_candidates(self, user_id: int, item_indices: Sequence[int], strategy: str) -> np.ndarray:
        if strategy == "popularity":
            return np.asarray([self.popularity_by_idx.get(i, 0.0) for i in item_indices], dtype=np.float32)

        model_scores: List[np.ndarray] = []
        if strategy in {"gmf", "hybrid"} and self.gmf is not None and user_id < self.gmf.user_emb.num_embeddings:
            model_scores.append(self._predict_model(self.gmf, user_id, item_indices))

        if strategy in {"mlp", "hybrid"} and self.mlp is not None and user_id < self.mlp.user_emb.num_embeddings:
            model_scores.append(self._predict_model(self.mlp, user_id, item_indices))

        if not model_scores:
            return np.asarray([self.popularity_by_idx.get(i, 0.0) for i in item_indices], dtype=np.float32)

        scores = np.mean(np.vstack(model_scores), axis=0)
        pop = np.asarray([self.popularity_by_idx.get(i, 0.0) for i in item_indices], dtype=np.float32)

        if strategy == "hybrid":
            scores = 0.9 * scores + 0.1 * pop
        else:
            scores = 0.95 * scores + 0.05 * pop

        return scores.astype(np.float32)

    @torch.no_grad()
    def _predict_model(self, model: nn.Module, user_id: int, item_indices: Sequence[int]) -> np.ndarray:
        users = torch.full((len(item_indices),), int(user_id), dtype=torch.long, device=self.device)
        items = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        logits = model(users, items)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    def _history_payload(self, user_id: int, limit: int = 15) -> List[Dict[str, Any]]:
        history = self.user_history.get(user_id, [])
        recent = list(reversed(history[-limit:]))
        return [self._payload_for_item_idx(item_idx) for item_idx in recent]

    def _payload_for_item_idx(self, item_idx: int) -> Dict[str, Any]:
        item_id = self.idx2item.get(item_idx)
        if item_id is None:
            return {
                "item_idx": int(item_idx),
                "item_id": -1,
                "title": f"Item {item_idx}",
                "clean_title": f"Item {item_idx}",
                "year": "",
                "genres": [],
                "imdb_url": "",
            }

        payload = self._payload_for_item_id(int(item_id))
        payload["item_idx"] = int(item_idx)
        return payload

    def _payload_for_item_id(self, item_id: int) -> Dict[str, Any]:
        item = self.item_lookup.get(item_id)
        if item is None:
            return {
                "item_id": int(item_id),
                "title": f"Movie {item_id}",
                "clean_title": f"Movie {item_id}",
                "year": "",
                "genres": [],
                "imdb_url": "",
            }

        return {
            "item_id": int(item["item_id"]),
            "title": item["title"],
            "clean_title": item["clean_title"],
            "year": item["year"],
            "genres": list(item["genres"]),
            "imdb_url": item["imdb_url"],
        }
