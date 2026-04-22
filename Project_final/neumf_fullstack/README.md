# NeuMF Full-Stack Web App

This is a standalone frontend + backend app for your MovieLens NeuMF project.
It does not modify or depend on changing your running notebook.

## What it uses

- Existing checkpoints from `../checkpoints/`
  - `gmf.pt`
  - `mlp.pt`
  - `dataset_meta.pkl`
- Existing MovieLens metadata from `../ml-100k/`
  - `u.data`
  - `u.item`

If model files are missing, the API falls back to popularity-based recommendation.

## Project structure

- `app.py` -> Flask server + API routes
- `recommender.py` -> model loading and ranking service
- `templates/index.html` -> main page
- `static/css/styles.css` -> styling
- `static/js/app.js` -> frontend logic

## Run

From this folder:

```powershell
pip install -r requirements.txt
python app.py
```

Then open:

- http://127.0.0.1:8000/

## API endpoints

- `GET /api/health`
- `GET /api/stats`
- `GET /api/users?limit=200`
- `GET /api/users/random`
- `GET /api/search?q=toy&limit=12`
- `GET /api/recommend?user_id=0&k=10&strategy=hybrid`

Recommendation parameters:

- `randomize=true|false` -> enables surprise sampling from top candidates
- `diversity=0..1` -> higher values increase variation in each call
- `novelty_penalty=0..1` -> penalizes recently served items for the same user

Example:

- `GET /api/recommend?user_id=0&k=10&strategy=hybrid&randomize=true&diversity=0.45&novelty_penalty=0.08`

Strategies depend on checkpoint availability:

- `hybrid` (GMF + MLP + popularity blend)
- `gmf`
- `mlp`
- `popularity`
