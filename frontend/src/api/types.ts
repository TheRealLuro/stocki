// Mirrors the pydantic schemas in backend/stocki/api/schemas.py and the
// response shapes of model/main.py.

export interface Health {
  status: string;
  database: string;
  bar_count: number | null;
}

export interface TickerSummary {
  ticker: string;
  name_en: string | null;
  name_cn: string | null;
  currency: string | null;
  session_count: number;
  first_day: number;
  last_day: number;
  bar_count: number;
}

export interface CoverageRow {
  ticker: string;
  day: number;
  bar_count: number;
  first_bar: string;
  last_bar: string;
}

export interface Bar {
  ticker: string;
  day: number;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Fundamentals {
  ticker: string;
  day: number;
  fundamentals: Record<string, unknown>;
}

export interface News {
  ticker: string;
  day: number;
  news_count: number | null;
  news_latest_headline: string | null;
}

export interface DatasetStats {
  n_windows: number;
  shape: number[];
  feature_names: string[];
  up_rate: number;
  tickers: string[];
  days: number[];
  window: number;
  horizon: number;
  subset: string;
}

export interface WindowsPayload {
  shape: number[];
  feature_names: string[];
  data: number[][][];
  target: number[];
  ticker: string[];
  day: number[];
  timestamps: string[];
}

export interface ModelHealth {
  model_loaded: boolean;
  model_path: string;
  detail?: Record<string, unknown> | null;
  error?: string | null;
}

export interface PredictResponse {
  prediction: number[];
}
