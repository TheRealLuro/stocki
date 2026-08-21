import type { Bar } from "../api/types";

const W = 780;
const PAD_L = 56;
const PAD_R = 12;
const PAD_T = 12;
const PRICE_H = 220;
const VOL_H = 56;
const GAP = 14;
const H = PAD_T + PRICE_H + GAP + VOL_H + 24;

function fmtTime(ts: string): string {
  // "2026-07-20T13:30:00Z" -> "13:30"
  return ts.slice(11, 16);
}

export function PriceChart({ bars }: { bars: Bar[] }) {
  const n = bars.length;
  const innerW = W - PAD_L - PAD_R;
  const step = innerW / n;
  const candleW = Math.max(2, Math.min(10, step * 0.6));

  let min = Infinity;
  let max = -Infinity;
  let maxVol = 0;
  for (const b of bars) {
    if (b.low < min) min = b.low;
    if (b.high > max) max = b.high;
    if (b.volume > maxVol) maxVol = b.volume;
  }
  const span = max - min || 1;

  const y = (v: number) => PAD_T + (1 - (v - min) / span) * PRICE_H;
  const volTop = PAD_T + PRICE_H + GAP;
  const cx = (i: number) => PAD_L + i * step + step / 2;

  const gridLevels = [0, 0.25, 0.5, 0.75, 1].map((f) => min + f * span);
  const timeIdx = [0, Math.floor(n / 2), n - 1];

  const change = bars[n - 1].close - bars[0].open;
  const changePct = (change / bars[0].open) * 100;
  const up = change >= 0;

  return (
    <div>
      <div className="chart-summary">
        <span className={up ? "up" : "down"}>
          {up ? "▲" : "▼"} {change >= 0 ? "+" : ""}
          {change.toFixed(2)} ({changePct.toFixed(2)}%)
        </span>
        <span>O {bars[0].open.toFixed(2)}</span>
        <span>H {max.toFixed(2)}</span>
        <span>L {min.toFixed(2)}</span>
        <span>C {bars[n - 1].close.toFixed(2)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="price-chart" role="img">
        {gridLevels.map((v) => (
          <g key={v}>
            <line
              x1={PAD_L}
              x2={W - PAD_R}
              y1={y(v)}
              y2={y(v)}
              className="grid"
            />
            <text x={PAD_L - 6} y={y(v) + 4} className="axis" textAnchor="end">
              {v.toFixed(2)}
            </text>
          </g>
        ))}

        {bars.map((b, i) => {
          const rising = b.close >= b.open;
          const color = rising ? "#26a69a" : "#ef5350";
          const bodyTop = y(Math.max(b.open, b.close));
          const bodyH = Math.max(1, Math.abs(y(b.open) - y(b.close)));
          return (
            <g key={b.timestamp}>
              <line
                x1={cx(i)}
                x2={cx(i)}
                y1={y(b.high)}
                y2={y(b.low)}
                stroke={color}
                strokeWidth={1}
              />
              <rect
                x={cx(i) - candleW / 2}
                y={bodyTop}
                width={candleW}
                height={bodyH}
                fill={color}
              />
              <rect
                x={cx(i) - candleW / 2}
                y={volTop + VOL_H - (b.volume / (maxVol || 1)) * VOL_H}
                width={candleW}
                height={(b.volume / (maxVol || 1)) * VOL_H}
                fill={color}
                opacity={0.55}
              />
            </g>
          );
        })}

        {timeIdx.map((i) => (
          <text
            key={i}
            x={cx(i)}
            y={H - 6}
            className="axis"
            textAnchor="middle"
          >
            {fmtTime(bars[i].timestamp)}
          </text>
        ))}
      </svg>
    </div>
  );
}
