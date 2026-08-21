import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import type { Bar, Fundamentals, News } from "../api/types";
import { PriceChart } from "./PriceChart";

type State =
  | { kind: "loading" }
  | { kind: "no-session"; message: string }
  | { kind: "error"; message: string }
  | { kind: "ok"; bars: Bar[]; fundamentals: Fundamentals | null; news: News | null };

// A few well-known fundamental fields worth surfacing when present.
const INTERESTING_KEYS = [
  "si_market_cap",
  "si_pe_trailing",
  "si_beta",
  "si_sector",
  "si_industry",
];

export function SessionExplorer({ ticker, day }: { ticker: string; day: number }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ kind: "loading" });
    Promise.all([
      api.sessionBars(ticker, day),
      api.fundamentals(ticker, day).catch(() => null),
      api.news(ticker, day).catch(() => null),
    ])
      .then(([bars, fundamentals, news]) => {
        if (!cancelled) setState({ kind: "ok", bars, fundamentals, news });
      })
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof ApiError && e.isNoSession) {
          setState({ kind: "no-session", message: e.detail });
        } else {
          setState({
            kind: "error",
            message: e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, day]);

  if (state.kind === "loading") {
    return <div className="card">Loading {ticker} day {day}…</div>;
  }
  if (state.kind === "no-session") {
    return (
      <div className="card muted-card">
        {state.message} — pick a filled cell in the coverage matrix below.
      </div>
    );
  }
  if (state.kind === "error") {
    return <div className="card error-card">{state.message}</div>;
  }

  const { bars, fundamentals, news } = state;
  const totalVolume = bars.reduce((acc, b) => acc + b.volume, 0);
  const fundamentalsEntries = INTERESTING_KEYS.filter(
    (k) => fundamentals?.fundamentals[k] != null,
  ).map((k) => [k, String(fundamentals!.fundamentals[k])] as const);

  return (
    <div className="card">
      <h2>
        {ticker} · Day {day}
      </h2>
      <PriceChart bars={bars} />
      <div className="detail-grid">
        <div>
          <h3>Session</h3>
          <p>{bars.length} bars · total volume {totalVolume.toLocaleString()}</p>
          <p>
            {bars[0].timestamp.slice(11, 16)} –{" "}
            {bars[bars.length - 1].timestamp.slice(11, 16)} UTC
          </p>
        </div>
        <div>
          <h3>News</h3>
          {news ? (
            <p>
              {news.news_count ?? 0} item(s)
              {news.news_latest_headline
                ? ` — latest: “${news.news_latest_headline}”`
                : ""}
            </p>
          ) : (
            <p className="muted">No news data.</p>
          )}
        </div>
        <div>
          <h3>Fundamentals</h3>
          {fundamentalsEntries.length > 0 ? (
            <ul>
              {fundamentalsEntries.map(([k, v]) => (
                <li key={k}>
                  <span className="muted">{k}</span> {v}
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No fundamentals data.</p>
          )}
        </div>
      </div>
    </div>
  );
}
