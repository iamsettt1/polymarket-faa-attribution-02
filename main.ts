// polymarket-faa-attribution-02: quantile forecaster for Polymarket Yes prices.
//
// What the data says (all of it visible in any case's own 72 hours):
//   - These prices are sticky: most minutes do not move at all, then a tick,
//     then now and then a jump. The best median forecast is the last price
//     itself; a median built from past drift only chases noise.
//   - The honest spread is each market's own: the distribution of its h-step
//     price changes over recent history. A fixed sigma for every market (the
//     house baseline's 0.0025) is far too wide for the quiet markets and too
//     narrow for the jumpy ones.
//   - Volatility clusters and decays: changes from the last 12 hours describe
//     the next 4 far better than changes from three days ago, and even they
//     overstate it, so the empirical offsets are scaled by 0.65 and nudged by
//     the ratio of the last 2 hours' step volatility to the window's. The two
//     outer quantiles get a further 1.15 tail widening: the loss in these
//     markets lives in rare jumps, and the tails are where a jump is paid for.
//   - The median offset is pinned to exactly 0: the empirical median change,
//     shrunk or not, is drift noise around the persistence centre.
//   - A market's first minutes are price discovery (it opens at 0.5 and falls
//     to its level), so the first 480 minutes are never used as sample starts.
//   - A market with little usable history borrows the case's other markets'
//     changes, and one with almost none falls back to a normal random-walk
//     term from its own step volatility, then to the house sigma. Borrowed
//     changes are first rescaled into the borrowing market's own volatility
//     units (source changes x own step-RMS / source step-RMS, the ratio
//     clamped to [1/2, 2]): the pooled distribution's shape — its fat atom at
//     zero and its jump tails — is shared, but its width is not, and feeding
//     a quiet market the raw changes of a jumpy one (or the reverse) was the
//     single largest error source in v1's backtests. Normalisation applies
//     only when the borrowing market's own window holds at least 240 steps;
//     below that its volatility estimate is itself noise, so it borrows raw.
//     In the final this matters doubly: the family has only four markets, all
//     young, all driven by the same news, so pooled jump tails are the best
//     evidence any one of them has about its own next four hours.
//
// The forecast is a deterministic function of the input file: no clock, no
// randomness, no network, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
const BASE_SIGMA = 0.0025;
const MIN_HISTORY = 50; // below this many prices, a market gets the plain normal fallback

const WARMUP_MIN = 480; // opening minutes never used as change-sample starts
const RECENT_MIN = 720; // change samples start within the last 12 hours
const SAMPLE_CAP = 2500; // max change samples per horizon (stride-sampled)
const MIN_SAMPLES = 30; // below this, a horizon uses the normal fallback term
const SPREAD_SCALE = 0.65; // shrink empirical offsets toward the median
const OUTER_SCALE = 1.15; // extra widening on the q05/q95 offsets only
const POOL_BELOW = 1500; // markets with fewer own samples borrow pooled changes
const POOL_FRAC = 0.75; // pooled share of the mixture when borrowing
const POOL_CAP = 2500; // max pooled samples taken per source market
const POOL_NORM_CLAMP = 2; // pooled changes rescaled by own/source step-RMS, ratio clamped to [1/2, 2]
const POOL_NORM_GATE = 240; // normalise only when the target's window holds at least this many steps
const REGIME_MIN = 120; // minutes of step history in the short vol window
const REGIME_LO = 0.6; // clamp for the vol-regime multiplier
const REGIME_HI = 1.3;
const HORIZON_MINUTES = 240; // forecast horizon — the grid's last point is the full 4 hours ahead
const GRID_H = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 63, 79, 100, 126, 158, 199, HORIZON_MINUTES];

interface CaseMarket { id: string; prices: number[]; }
interface Case { horizon_minutes: number; quantiles: number[]; markets: CaseMarket[]; }
interface PoolSrc { p: Float64Array; firstStart: number; }

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const round6 = (v: number): number => Math.round(v * 1e6) / 1e6 + 0;

/** Quantile of a sorted Float64Array, linear interpolation (type 7). */
function quantileOf(sorted: Float64Array, tau: number): number {
  const m = sorted.length;
  if (m === 0) return 0;
  if (m === 1) return sorted[0];
  const pos = tau * (m - 1);
  const lo = Math.floor(pos);
  const frac = pos - lo;
  return lo + 1 < m ? sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]) : sorted[lo];
}

/** First usable sample start: warmup and recency both applied. */
function firstStartOf(n: number): number {
  return Math.max(0, Math.min(WARMUP_MIN, n - 2), n - RECENT_MIN);
}

/** H-step changes of one series, starts in [firstStart, n-1-H], stride-capped. */
function collectChanges(p: Float64Array, firstStart: number, H: number, cap: number): Float64Array {
  const n = p.length;
  const count = n - 1 - H - firstStart + 1;
  if (count <= 0) return new Float64Array(0);
  const stride = Math.max(1, Math.floor(count / cap));
  const s = new Float64Array(Math.ceil(count / stride));
  let w = 0;
  for (let i = firstStart; i <= n - 1 - H; i += stride) s[w++] = p[i + H] - p[i];
  return s.subarray(0, w);
}

/** RMS of a series' 1-step changes over [from, len-1). */
function stepRms(p: Float64Array, from: number): number {
  let ss = 0, c = 0;
  for (let i = from; i < p.length - 1; i++) { const d = p[i + 1] - p[i]; ss += d * d; c++; }
  return c ? Math.sqrt(ss / c) : 0;
}

/** The five quantile offsets of the price change at one grid horizon. */
function horizonOffsets(p: Float64Array, firstStart: number, H: number, sigma: number, pool: PoolSrc[] | null, self: Float64Array): number[] {
  const own = collectChanges(p, firstStart, H, SAMPLE_CAP);
  let used = own;
  if (pool && own.length < POOL_BELOW) {
    const parts: Float64Array[] = [own];
    let poolLen = 0;
    // Rescale borrowed changes into this market's volatility units, but only
    // when this market's own window is long enough to trust its step-RMS.
    const norm = p.length - 1 - firstStart >= POOL_NORM_GATE && sigma > 0;
    for (const src of pool) {
      if (src.p === self) continue;
      let c = collectChanges(src.p, src.firstStart, H, POOL_CAP);
      if (norm) {
        const sSig = stepRms(src.p, src.firstStart);
        if (sSig > 0) {
          const f = Math.min(POOL_NORM_CLAMP, Math.max(1 / POOL_NORM_CLAMP, sigma / sSig));
          if (f !== 1) {
            const cc = new Float64Array(c.length);
            for (let i = 0; i < c.length; i++) cc[i] = c[i] * f;
            c = cc;
          }
        }
      }
      parts.push(c);
      poolLen += c.length;
    }
    const take = Math.min(poolLen, Math.round((own.length * POOL_FRAC) / (1 - POOL_FRAC)));
    if (take > 0) {
      const all = parts.reduce((a, b) => a + b.length, 0);
      const merged = new Float64Array(all);
      let off = 0;
      for (const part of parts) { merged.set(part, off); off += part.length; }
      // own samples first, then a stride through the rest of the pool
      const rest = merged.subarray(own.length);
      const stride = Math.max(1, Math.floor(rest.length / take));
      const mixed = new Float64Array(own.length + Math.ceil(rest.length / stride));
      mixed.set(own, 0);
      let w = own.length;
      for (let i = 0; i < rest.length; i += stride) mixed[w++] = rest[i];
      used = mixed.subarray(0, w);
    }
  }
  if (used.length < MIN_SAMPLES) return Z.map((z) => z * sigma * Math.sqrt(H));
  const sorted = Float64Array.from(used).sort();
  // The median offset is pinned to 0: persistence is the median forecast.
  return QUANTILES.map((tau, j) => (j === 2 ? 0 : quantileOf(sorted, tau) * SPREAD_SCALE));
}

function forecastMarket(m: CaseMarket, horizon: number, pool: PoolSrc[] | null, self: Float64Array): number[][] {
  const p = Float64Array.from(m.prices);
  const n = p.length;
  const x = p[n - 1];
  if (n < MIN_HISTORY) {
    const rows: number[][] = [];
    for (let h = 1; h <= horizon; h++) rows.push(Z.map((z) => round6(clamp01(x + z * BASE_SIGMA * Math.sqrt(h)))));
    return rows;
  }
  const firstStart = firstStartOf(n);
  // Own step volatility (RMS of 1-step changes over the usable window): the
  // fallback term uses it, and the vol-regime multiplier.
  const rms = (a: number, b: number): number => {
    let ss = 0, c = 0;
    for (let i = a; i < b; i++) { const d = p[i + 1] - p[i]; ss += d * d; c++; }
    return c ? Math.sqrt(ss / c) : 0;
  };
  const sigWin = rms(firstStart, n - 1);
  let sigma = sigWin;
  if (!(sigma > 0)) sigma = BASE_SIGMA;
  let regime = 1;
  if (sigWin > 0) {
    const sigShort = rms(Math.max(firstStart, n - 1 - REGIME_MIN), n - 1);
    regime = Math.min(REGIME_HI, Math.max(REGIME_LO, sigShort / sigWin));
  }
  const gridOff = GRID_H.map((H) => horizonOffsets(p, firstStart, H, sigma, pool, self));
  const gridSq = GRID_H.map((H) => Math.sqrt(H));
  const rows: number[][] = [];
  let g = 0;
  for (let h = 1; h <= horizon; h++) {
    const sq = Math.sqrt(h);
    while (g < GRID_H.length - 2 && gridSq[g + 1] < sq) g++;
    const f = (sq - gridSq[g]) / (gridSq[g + 1] - gridSq[g]);
    const row = gridOff[g].map((v, j) => {
      const interp = v + (gridOff[g + 1][j] - v) * f;
      if (j === 2) return interp; // 0 by construction: persistence median
      const om = j === 0 || j === 4 ? OUTER_SCALE : 1;
      return interp * regime * om;
    });
    // Keep the rows ordered regardless of the multipliers above.
    const med = row[2];
    row[0] = Math.min(row[0], row[1], med);
    row[1] = Math.min(row[1], med);
    row[3] = Math.max(row[3], med);
    row[4] = Math.max(row[4], row[3]);
    rows.push(row.map((off) => round6(clamp01(x + off))));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}`);
  }
  const pool: PoolSrc[] = c.markets.map((m) => {
    const p = Float64Array.from(m.prices);
    return { p, firstStart: firstStartOf(p.length) };
  });
  const forecasts = c.markets.map((m, i) => ({ id: m.id, q: forecastMarket(m, c.horizon_minutes, pool, pool[i].p) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
