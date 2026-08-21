import type { CoverageRow, TickerSummary } from "../api/types";

const DAYS = Array.from({ length: 20 }, (_, i) => i + 1);

interface Props {
  coverage: CoverageRow[];
  tickers: TickerSummary[];
  selectedTicker: string | null;
  selectedDay: number | null;
  onSelect: (ticker: string, day: number) => void;
}

export function CoverageGrid({
  coverage,
  tickers,
  selectedTicker,
  selectedDay,
  onSelect,
}: Props) {
  const filled = new Map(coverage.map((r) => [`${r.ticker}:${r.day}`, r.bar_count]));

  return (
    <div className="card">
      <h2>Coverage</h2>
      <div className="coverage-scroll">
        <table className="coverage">
          <thead>
            <tr>
              <th></th>
              {DAYS.map((d) => (
                <th key={d}>{d}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((t) => (
              <tr key={t.ticker}>
                <th>{t.ticker}</th>
                {DAYS.map((d) => {
                  const bars = filled.get(`${t.ticker}:${d}`);
                  const selected = t.ticker === selectedTicker && d === selectedDay;
                  return (
                    <td key={d}>
                      {bars != null ? (
                        <button
                          className={`cell covered${selected ? " selected" : ""}`}
                          title={`${bars} bars`}
                          onClick={() => onSelect(t.ticker, d)}
                        />
                      ) : (
                        <span className="cell empty" title="not collected" />
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
