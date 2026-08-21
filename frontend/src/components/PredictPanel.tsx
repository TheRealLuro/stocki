import { useState } from "react";
import { ApiError, api } from "../api/client";

interface Result {
  prediction: number[];
  timestamp: string;
  day: number;
}

const WINDOW_SIZE = 32;

function buildWindowFromSession(
  bars: { open: number; high: number; low: number; close: number; volume: number; timestamp: string }[],
) {
  const slice = bars.slice(-WINDOW_SIZE);
  if (slice.length === 0) throw new Error("No bars available for the selected day.");

  const rows = slice.map((bar, index) => {
    const prevClose = index === 0 ? slice[0].close : slice[index - 1].close;
    const logReturn = prevClose !== 0 ? Math.log(bar.close / prevClose) : 0;
    const hlRange = bar.close !== 0 ? (bar.high - bar.low) / bar.close : 0;
    const coRange = bar.open !== 0 ? (bar.close - bar.open) / bar.open : 0;

    return [bar.open, bar.high, bar.low, bar.close, bar.volume, logReturn, hlRange, coRange];
  });

  return rows;
}

export function PredictPanel({ ticker, day }: { ticker: string; day: number }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const bars = await api.sessionBars(ticker, day);
      const sequence = buildWindowFromSession(bars);
      const res = await api.predict(sequence);
      const lastBar = bars[bars.length - 1];
      setResult({
        prediction: res.prediction,
        timestamp: lastBar.timestamp,
        day,
      });
    } catch (e) {
      setResult(null);
      setError(
        e instanceof ApiError
          ? `${e.status}: ${e.detail}`
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setBusy(false);
    }
  }

  const probability = result?.prediction[0] ?? null;
  const predictedUp = probability != null ? probability >= 0.5 : null;

  return (
    <div className="card">
      <h2>Prediction</h2>
      <p className="muted small">
        Predicts the next move using the selected day’s last {WINDOW_SIZE} bars.
      </p>
      <button onClick={run} disabled={busy}>
        {busy ? "Running…" : `Predict ${ticker}`}
      </button>
      {error && <p className="error-text">{error}</p>}
      {result && (
        <div className="prediction-result">
          <div className={predictedUp ? "direction up" : "direction down"}>
            <span className="confidence">
              {predictedUp == null ? "—" : predictedUp ? "UP" : "DOWN"}
            </span>
          </div>
          <p className="small">
            selected sample: {result.timestamp.slice(0, 16).replace("T", " ")} · day {result.day}
          </p>
        </div>
      )}
    </div>
  );
}
