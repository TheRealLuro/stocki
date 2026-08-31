import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import type { Bar, Fundamentals, News } from "../api/types";
import { PriceChart } from "./PriceChart";

type State =
  | { kind: "loading" }
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

export function SessionExplorer({
  ticker,
  startDay,
  endDay,
}: {
  ticker: string;
  startDay: number;
  endDay: number;
}) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ kind: "loading" });
    const days: number[] = [];
    for (let d = startDay; d <= endDay; d++) days.push(d);
    Promise.all([
      // Skip days that were never collected; surface any other failure.
      Promise.all(
        days.map((d) =>
          api.sessionBars(ticker, d).catch((e) => {
            if (e instanceof ApiError && e.isNoSession) return null;
            throw e;
          }),
        ),
      ),
      api.fundamentals(ticker, endDay).catch(() => null),
      api.news(ticker, endDay).catch(() => null),
    ])
      .then(([sessions, fundamentals, news]) => {
        if (cancelled) return;
        const bars = sessions.filter((s): s is Bar[] => s !== null).flat();
        setState({ kind: "ok", bars, fundamentals, news });
      })
      .catch((e) => {
        if (cancelled) return;
        setState({
          kind: "error",
          message: e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, startDay, endDay]);

  if (state.kind === "loading") {
    return (
      <div className="card">
        Loading {ticker} days {startDay}–{endDay}…
      </div>
    );
  }
  if (state.kind === "error") {
    return <div className="card error-card">{state.message}</div>;
  }

  const { bars, fundamentals, news } = state;
  const rangeLabel =
    startDay === endDay ? `Day ${startDay}` : `Days ${startDay}–${endDay}`;

  if (bars.length === 0) {
    return (
      <div className="card muted-card">
        No collected sessions for {ticker} on {rangeLabel.toLowerCase()} — pick
        filled cells in the coverage matrix below.
      </div>
    );
  }
  const totalVolume = bars.reduce((acc, b) => acc + b.volume, 0);
  const multiDay = bars[0].day !== bars[bars.length - 1].day;
  const fmtBarTime = (b: Bar) =>
    multiDay ? `D${b.day} ${b.timestamp.slice(11, 16)}` : b.timestamp.slice(11, 16);
  const fundamentalsEntries = INTERESTING_KEYS.filter(
    (k) => fundamentals?.fundamentals[k] != null,
  ).map((k) => [k, String(fundamentals!.fundamentals[k])] as const);

  return (
    <div className="card">
      <h2>
        {ticker} · {rangeLabel}
      </h2>
      <PriceChart bars={bars} />
      <div className="detail-grid">
        <div>
          <h3>Session</h3>
          <p>{bars.length} bars · total volume {totalVolume.toLocaleString()}</p>
          <p>
            {fmtBarTime(bars[0])} – {fmtBarTime(bars[bars.length - 1])} UTC
          </p>
        </div>
        <div>
          <h3>News{multiDay ? ` (day ${endDay})` : ""}</h3>
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
          <h3>Fundamentals{multiDay ? ` (day ${endDay})` : ""}</h3>
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
