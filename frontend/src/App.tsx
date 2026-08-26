import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import type { CoverageRow, TickerSummary } from "./api/types";
import { CoverageGrid } from "./components/CoverageGrid";
import { DatasetPanel } from "./components/DatasetPanel";
import { HealthBanner } from "./components/HealthBanner";
import { PredictPanel } from "./components/PredictPanel";
import { SessionExplorer } from "./components/SessionExplorer";

const ALL_DAYS = Array.from({ length: 20 }, (_, i) => i + 1);

export default function App() {
  const [tickers, setTickers] = useState<TickerSummary[]>([]);
  const [coverage, setCoverage] = useState<CoverageRow[]>([]);
  const [ticker, setTicker] = useState<string | null>(null);
  const [startDay, setStartDay] = useState<number | null>(null);
  const [endDay, setEndDay] = useState<number | null>(null);
  const [initError, setInitError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.tickers(), api.coverage()])
      .then(([t, c]) => {
        setTickers(t);
        setCoverage(c);
        if (t.length > 0) setTicker((prev) => prev ?? t[0].ticker);
      })
      .catch((e) => setInitError(e instanceof Error ? e.message : String(e)));
  }, []);

  const coveredDays = useMemo(
    () => new Set(coverage.filter((r) => r.ticker === ticker).map((r) => r.day)),
    [coverage, ticker],
  );

  // Keep the selected range on sessions that were actually collected.
  useEffect(() => {
    if (coveredDays.size === 0) return;
    const latest = Math.max(...coveredDays);
    const end =
      endDay === null || !coveredDays.has(endDay) ? latest : endDay;
    const start =
      startDay === null || !coveredDays.has(startDay) || startDay > end
        ? end
        : startDay;
    if (end !== endDay) setEndDay(end);
    if (start !== startDay) setStartDay(start);
  }, [coveredDays, startDay, endDay]);

  const summary = tickers.find((t) => t.ticker === ticker) ?? null;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Stocki</h1>
          <p className="subtitle">5-minute OHLCV &rarr; next-move prediction</p>
        </div>
        <HealthBanner />
      </header>

      {initError && (
        <div className="card error-card">
          Could not reach the backend API: {initError}. Is{" "}
          <code>docker compose up -d</code> running?
        </div>
      )}

      <div className="controls card">
        <label>
          Ticker
          <select
            value={ticker ?? ""}
            onChange={(e) => setTicker(e.target.value)}
          >
            {tickers.map((t) => (
              <option key={t.ticker} value={t.ticker}>
                {t.ticker}
                {t.name_en ? ` — ${t.name_en}` : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          From day
          <select
            value={startDay ?? ""}
            onChange={(e) => {
              const d = Number(e.target.value);
              setStartDay(d);
              if (endDay !== null && d > endDay) setEndDay(d);
            }}
          >
            {ALL_DAYS.map((d) => (
              <option key={d} value={d} disabled={!coveredDays.has(d)}>
                Day {d}
                {coveredDays.has(d) ? "" : " (not collected)"}
              </option>
            ))}
          </select>
        </label>
        <label>
          To day
          <select
            value={endDay ?? ""}
            onChange={(e) => {
              const d = Number(e.target.value);
              setEndDay(d);
              if (startDay !== null && d < startDay) setStartDay(d);
            }}
          >
            {ALL_DAYS.map((d) => (
              <option key={d} value={d} disabled={!coveredDays.has(d)}>
                Day {d}
                {coveredDays.has(d) ? "" : " (not collected)"}
              </option>
            ))}
          </select>
        </label>
        {summary && (
          <div className="ticker-meta">
            {summary.session_count} sessions · {summary.bar_count.toLocaleString()}{" "}
            bars · {summary.currency ?? ""}
          </div>
        )}
      </div>

      <div className="layout">
        <div className="main-col">
          {ticker !== null && startDay !== null && endDay !== null && (
            <SessionExplorer
              key={`${ticker}-${startDay}-${endDay}`}
              ticker={ticker}
              startDay={startDay}
              endDay={endDay}
            />
          )}
          <CoverageGrid
            coverage={coverage}
            tickers={tickers}
            selectedTicker={ticker}
            selectedDay={endDay}
            onSelect={(t, d) => {
              setTicker(t);
              setStartDay(d);
              setEndDay(d);
            }}
          />
        </div>
        <aside className="side-col">
          <DatasetPanel ticker={ticker} />
          {ticker !== null && endDay !== null && (
            <PredictPanel ticker={ticker} day={endDay} />
          )}
        </aside>
      </div>
    </div>
  );
}
