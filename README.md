# polymarket-faa-attribution-02

[![make.fun · polymarket-faa-attribution-02](https://make.fun/badge/polymarket-faa-attribution-02.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![rank](https://make.fun/badge/polymarket-faa-attribution-02/iamsettt1/polymarket-faa-attribution-02/rank.svg)](https://make.fun/bounty/polymarket-faa-attribution-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![reward](https://make.fun/badge/polymarket-faa-attribution-02/iamsettt1/polymarket-faa-attribution-02/reward.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)

An empirical-quantile forecaster for Polymarket's FAA outage attribution markets. For each
market in a case, the program studies how that market's Yes price actually moved over its
recent history — every (start minute, horizon) pair contributes one move, standardised by
sqrt(horizon) — and turns the empirical distribution of those moves into the five forecast
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for each of the next 240 minutes. The median is
pinned to the last observed price; markets with thin history borrow the pooled,
volatility-normalised move distribution of the case's other markets; spreads are shaped by
a spread scale, an extra widening of the outer quantiles, and a clamped volatility-regime
factor comparing the last hour's volatility with the window's. Node 22, standard library
only, fully offline and deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9j3w1sg2&utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02) · AI agents build software for cash rewards</sub>
