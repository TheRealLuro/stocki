# Stocki frontend

React + TypeScript (Vite) dashboard for the Stocki dataset and prediction APIs.

## What it shows

- Health pills for the backend API and the ONNX model API
- Candlestick + volume chart over a selectable day range
  (`/api/v1/bars/{ticker}/{day}` per day in the range)
- Session details: volume, news headline, fundamentals
- Ticker × day coverage matrix (gaps like AAPL days 1–12 are visible)
- Dataset stats (`/api/v1/dataset/stats`), overall and per ticker
- Prediction panel: fetches a CNN-ready window from
  `/api/v1/dataset/windows` and POSTs it to the model's `/predict`

## Run

From the repo root:

```bash
docker compose up -d          # backend API on http://localhost:8000
cd model && uvicorn main:app --port 8001   # model API (8000 is taken)
```

Then here:

```bash
npm install
npm run dev                   # http://localhost:5173
```

The backend allows CORS from `localhost:5173` already. The model API has no
CORS middleware, so the Vite dev server proxies `/model-api/*` to
`http://localhost:8001` (see `vite.config.ts`).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | backend API base URL |
| `VITE_MODEL_API_URL` | `/model-api` | model API base URL (dev proxy) |

The dev proxy target is fixed to `http://localhost:8001` in `vite.config.ts`.

## Scripts

```bash
npm run dev       # dev server
npm run build     # typecheck (tsc) + production build to dist/
npm run preview   # serve the production build
```
