// polymarket-faa-attribution-02 — empirical-quantile forecaster (Node 22, standard library only).
//
// What it does
// ------------
// For every market in a case it looks at how that market's Yes price has actually moved in the
// recent past, minute to horizon, and turns the empirical distribution of those moves into the
// five forecast quantiles for each of the next `horizon_minutes` minutes:
//
//   * Moves are studied in "return space": a move over h minutes is divided by sqrt(h), because
//     the spread of a wandering price grows with the square root of time. Every (start minute,
//     horizon) pair in the history window contributes one standardised move, so a market's whole
//     window becomes a single sample of per-root-minute moves.
//   * The forecast for horizon h takes the empirical quantiles of that sample, recentres them on
//     the sample median, and scales them back up by sqrt(h). The median itself is pinned to the
//     last observed price: the crowd's current price is the centre of the forecast.
//   * A market whose own history is thin borrows strength from its siblings: every market's
//     standardised moves are divided by that market's own volatility, pooled into one
//     unit-volatility sample, and the pooled quantiles are re-scaled by the borrowing market's
//     volatility.
//   * Spreads are shaped before use: offsets from the median are multiplied by SPREAD_SCALE,
//     the two outer quantiles (5% and 95%) are pushed a further OUTER_SCALE wider, and the whole
//     spread is multiplied by a volatility-regime factor — the ratio of the last hour's
//     volatility to the window's, clamped to [0.6, 1.3] — so a market that has just woken up
//     gets a wider fan than one that has gone quiet.
//   * Every value is clamped to [0, 1], each row is kept non-decreasing, and numbers are rounded
//     to 6 decimals. The program is deterministic: two runs on the same case write the same bytes.
//
// Input and output follow the contest contract: read every `<id>.in` in $MAKEFUN_INPUT_DIR
// (else /input), write `<id>.out` — {"forecasts":[{"id","q"}]} — to $MAKEFUN_OUTPUT_DIR
// (else /output).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each output row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Offsets from the median are multiplied by this before they become a forecast. */
const SPREAD_SCALE = 0.65;
/** The outer pair of quantiles (0.05 and 0.95) is widened by this extra factor. */
const OUTER_SCALE = 1.15;
/** Minutes of history the empirical move sample is built from (the recent 12 hours). */
const HISTORY_MINUTES = 720;
/** A market with fewer minutes of history than this forecasts from the pooled sample instead. */
const THIN_MINUTES = 720;
/** Minutes at the end of the window whose volatility defines the current regime. */
const RECENT_MINUTES = 60;
/** The regime multiplier (recent volatility / window volatility) is clamped to this band. */
const REGIME_FLOOR = 0.6;
const REGIME_CEIL = 1.3;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** Round to 6 decimals after clamping into [0, 1]; the +0 turns a rounded -0 into 0. */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** The tau-quantile of an ascending-sorted sample, linearly interpolated between neighbours. */
function sampleQuantile(sorted: number[], tau: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0] as number;
  const pos = tau * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  const frac = pos - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

/** Population standard deviation; 0 for a sample of fewer than two values. */
function stdDev(xs: number[]): number {
  if (xs.length < 2) return 0;
  let mean = 0;
  for (const x of xs) mean += x;
  mean /= xs.length;
  let acc = 0;
  for (const x of xs) {
    const d = x - mean;
    acc += d * d;
  }
  return Math.sqrt(acc / xs.length);
}

/** Square roots of the horizons, computed once and reused by every market and case. */
const ROOTS: number[] = [0];
function rootOf(h: number): number {
  let r = ROOTS[h];
  if (r === undefined) {
    r = Math.sqrt(h);
    ROOTS[h] = r;
  }
  return r;
}

/** Consecutive one-minute changes of a price series. */
function minuteChanges(prices: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < prices.length; i++) out.push((prices[i] as number) - (prices[i - 1] as number));
  return out;
}

/**
 * Every standardised move in a window: for each horizon h from 1 to `horizon` and each start
 * minute t, (price[t + h] - price[t]) / sqrt(h). One sample describing how this market moves
 * per root-minute, whatever the horizon.
 */
function standardisedMoves(prices: number[], horizon: number): number[] {
  const out: number[] = [];
  const n = prices.length;
  const maxH = Math.min(horizon, n - 1);
  for (let h = 1; h <= maxH; h++) {
    const root = rootOf(h);
    for (let t = 0; t + h < n; t++) out.push(((prices[t + h] as number) - (prices[t] as number)) / root);
  }
  return out;
}

interface MarketModel {
  id: string;
  last: number;
  /** Minutes of prices this market's window holds; decides own-sample vs pooled. */
  minutes: number;
  /** Sorted standardised moves of the market's own window, in price units per root-minute. */
  ownSorted: number[];
  /** Volatility of the market's one-minute changes over its window. */
  vol: number;
  /** Clamped ratio of the last hour's volatility to the window's volatility. */
  regime: number;
}

/** Build the per-market statistics a forecast is drawn from. */
function modelMarket(m: CaseMarket, horizon: number): MarketModel {
  const prices = m.prices.slice(-HISTORY_MINUTES);
  const moves = standardisedMoves(prices, horizon);
  moves.sort((a, b) => a - b);
  const changes = minuteChanges(prices);
  const vol = stdDev(changes);
  const recentVol = stdDev(changes.slice(-RECENT_MINUTES));
  // The regime factor applies only when both volatilities are measurable: a last hour with no
  // move at all carries no information about the minutes ahead (the price may simply not have
  // traded), so it leaves the fan at its window width rather than narrowing it.
  let regime = 1;
  if (vol > 0 && recentVol > 0) regime = Math.min(REGIME_CEIL, Math.max(REGIME_FLOOR, recentVol / vol));
  return {
    id: m.id,
    last: prices[prices.length - 1] as number,
    minutes: prices.length,
    ownSorted: moves,
    vol,
    regime,
  };
}

/**
 * The five quantile offsets (from the median, per root-minute, price units) a model forecasts
 * with: its own sample when its history is long enough, otherwise the pooled unit-volatility
 * sample re-scaled by its own volatility.
 */
function quantileOffsets(model: MarketModel, pooledSorted: number[] | null): number[] {
  let sorted: number[] | null = null;
  let unit = 1;
  if (model.minutes >= THIN_MINUTES || pooledSorted === null) {
    sorted = model.ownSorted;
  } else if (model.vol > 0) {
    sorted = pooledSorted;
    unit = model.vol;
  } else {
    sorted = model.ownSorted;
  }
  if (sorted === null || sorted.length === 0) return QUANTILES.map(() => 0);
  const centre = sampleQuantile(sorted, 0.5) * unit;
  return QUANTILES.map((tau, j) => {
    const raw = sampleQuantile(sorted, tau) * unit - centre;
    const isOuter = j === 0 || j === QUANTILES.length - 1;
    return raw * SPREAD_SCALE * (isOuter ? OUTER_SCALE : 1) * model.regime;
  });
}

/** One market's forecast: horizon rows of five quantiles fanning out from its last price. */
function forecastRows(model: MarketModel, offsets: number[], horizon: number): number[][] {
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const root = rootOf(h);
    const row = offsets.map((off) => price6(model.last + off * root));
    // The median offset is 0 by construction, so the row is already ordered; enforce it anyway
    // so a clamped row can never come out decreasing.
    for (let j = 1; j < row.length; j++) if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const horizon = c.horizon_minutes;
  const models = c.markets.map((m) => modelMarket(m, horizon));
  // The pooled sample: every market's standardised moves divided by its own volatility, so
  // markets of different scales can lend each other their move shapes.
  const pooled: number[] = [];
  for (const model of models) {
    if (model.vol <= 0) continue;
    for (const s of model.ownSorted) pooled.push(s / model.vol);
  }
  pooled.sort((a, b) => a - b);
  const pooledSorted = pooled.length > 0 ? pooled : null;
  const forecasts = models.map((model) => ({
    id: model.id,
    q: forecastRows(model, quantileOffsets(model, pooledSorted), horizon),
  }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
