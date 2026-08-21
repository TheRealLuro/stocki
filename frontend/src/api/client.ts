import type {
  Bar,
  CoverageRow,
  DatasetStats,
  Fundamentals,
  Health,
  ModelHealth,
  News,
  PredictResponse,
  TickerSummary,
  WindowsPayload,
} from "./types";

const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
// In dev this hits the Vite proxy (see vite.config.ts); override with
// VITE_MODEL_API_URL for a directly reachable model service.
const MODEL_API_URL: string = import.meta.env.VITE_MODEL_API_URL ?? "/model-api";

/** An API failure. `isNoSession` marks the normal 404 "session not collected". */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
    public readonly requestId?: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }

  get isNoSession(): boolean {
    return this.status === 404;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    let requestId = res.headers.get("X-Request-ID") ?? undefined;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail != null) detail = JSON.stringify(body.detail);
      if (typeof body.request_id === "string") requestId = body.request_id;
    } catch {
      // non-JSON error body -- keep statusText
    }
    throw new ApiError(res.status, detail, requestId);
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number>): string {
  const search = new URLSearchParams(
    Object.entries(params).map(([k, v]) => [k, String(v)]),
  );
  return `?${search.toString()}`;
}

export const api = {
  // --- backend (read-only dataset API) -----------------------------------
  // /health returns 503 when degraded; callers that render status should catch.
  health: () => request<Health>(`${API_URL}/health`),
  tickers: () => request<TickerSummary[]>(`${API_URL}/api/v1/tickers`),
  coverage: () => request<CoverageRow[]>(`${API_URL}/api/v1/coverage`),
  sessionBars: (ticker: string, day: number) =>
    request<Bar[]>(`${API_URL}/api/v1/bars/${ticker}/${day}`),
  fundamentals: (ticker: string, day: number) =>
    request<Fundamentals>(`${API_URL}/api/v1/fundamentals/${ticker}/${day}`),
  news: (ticker: string, day: number) =>
    request<News>(`${API_URL}/api/v1/news/${ticker}/${day}`),
  datasetStats: (ticker?: string) =>
    request<DatasetStats>(
      `${API_URL}/api/v1/dataset/stats${ticker ? qs({ ticker }) : ""}`,
    ),
  windows: (ticker: string, limit = 1) =>
    request<WindowsPayload>(
      `${API_URL}/api/v1/dataset/windows${qs({ ticker, limit })}`,
    ),

  // --- model (prediction API) --------------------------------------------
  modelHealth: () => request<ModelHealth>(`${MODEL_API_URL}/health`),
  predict: (sequence: number[][]) =>
    request<PredictResponse>(`${MODEL_API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    }),
};
