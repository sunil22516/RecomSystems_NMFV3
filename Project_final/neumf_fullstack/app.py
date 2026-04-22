from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from recommender import NeuMFRecommenderService


APP_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_DIR.parent

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)
service = NeuMFRecommenderService(WORKSPACE_ROOT)


def _json_error(message: str, status: int = 400) -> tuple[Any, int]:
    return jsonify({"ok": False, "error": message}), status


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/health")
def health() -> Any:
    return jsonify(
        {
            "ok": True,
            "service": "neumf-fullstack",
            "workspace_root": str(WORKSPACE_ROOT),
        }
    )


@app.route("/api/stats")
def stats() -> Any:
    payload = service.get_stats()
    payload["ok"] = True
    return jsonify(payload)


@app.route("/api/users")
def users() -> Any:
    limit = int(request.args.get("limit", 200))
    limit = max(1, min(limit, 1000))
    return jsonify({"ok": True, "users": service.list_users(limit=limit)})


@app.route("/api/users/random")
def random_user() -> Any:
    try:
        return jsonify({"ok": True, "user_id": service.random_user()})
    except Exception as exc:
        return _json_error(str(exc), 500)


@app.route("/api/search")
def search() -> Any:
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 12))
    limit = max(1, min(limit, 50))
    return jsonify({"ok": True, "items": service.search_titles(query, limit=limit)})


@app.route("/api/recommend")
def recommend() -> Any:
    if "user_id" not in request.args:
        return _json_error("Missing required query parameter: user_id")

    try:
        user_id = int(request.args.get("user_id", ""))
    except ValueError:
        return _json_error("user_id must be an integer")

    try:
        k = int(request.args.get("k", 10))
    except ValueError:
        return _json_error("k must be an integer")

    strategy = request.args.get("strategy", "hybrid")
    randomize = _parse_bool(request.args.get("randomize"), default=True)

    try:
        diversity = float(request.args.get("diversity", 0.45))
    except ValueError:
        return _json_error("diversity must be a number between 0 and 1")

    try:
        novelty_penalty = float(request.args.get("novelty_penalty", 0.08))
    except ValueError:
        return _json_error("novelty_penalty must be a number between 0 and 1")

    try:
        payload = service.recommend(
            user_id=user_id,
            k=k,
            strategy=strategy,
            randomize=randomize,
            diversity=diversity,
            novelty_penalty=novelty_penalty,
        )
        payload["ok"] = True
        return jsonify(payload)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        return _json_error(str(exc), 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
