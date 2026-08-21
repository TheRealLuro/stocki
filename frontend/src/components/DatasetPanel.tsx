import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DatasetStats } from "../api/types";

function StatsBlock({ title, stats }: { title: string; stats: DatasetStats }) {
  return (
    <div className="stats-block">
      <h3>{title}</h3>
      <dl>
        <div>
          <dt>Windows</dt>
          <dd>{stats.n_windows.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Shape</dt>
          <dd>
            ({stats.shape.join(" × ")}) · {stats.feature_names.length} features
          </dd>
        </div>
        <div>
          <dt>UP rate</dt>
          <dd>{(stats.up_rate * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt>Window / horizon</dt>
          <dd>
            {stats.window} / {stats.horizon}
          </dd>
        </div>
      </dl>
    </div>
  );
}

export function DatasetPanel({ ticker }: { ticker: string | null }) {
  const [all, setAll] = useState<DatasetStats | null>(null);
  const [perTicker, setPerTicker] = useState<DatasetStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .datasetStats()
      .then(setAll)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (!ticker) {
      setPerTicker(null);
      return;
    }
    api
      .datasetStats(ticker)
      .then(setPerTicker)
      .catch(() => setPerTicker(null));
  }, [ticker]);

  return (
    <div className="card">
      <h2>Dataset</h2>
      {error && <p className="error-text">{error}</p>}
      {all ? <StatsBlock title="All tickers" stats={all} /> : !error && <p>Loading…</p>}
      {perTicker && ticker && <StatsBlock title={ticker} stats={perTicker} />}
      {all && (
        <p className="muted small">features: {all.feature_names.join(", ")}</p>
      )}
    </div>
  );
}
