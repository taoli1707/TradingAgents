# Pattern recognition and backtesting engine

`tradingagents.patterns` searches daily equity prices and technical indicators for
conjunctions of conditions ("patterns") whose forward returns beat the
unconditional base rate, and it validates every candidate with walk-forward
tests, multiple-hypothesis correction and cost-aware trade simulation.

The engine is deliberately skeptical. Most patterns that look excellent in
sample are selection artefacts, and the engine is built to say so.

## Quick start

```bash
pip install yfinance scipy scikit-learn lightgbm pyarrow

# discover, validate, walk-forward, train the model and backtest
python -m tradingagents.patterns.cli mine \
    --symbols SPY QQQ IWM AAPL MSFT NVDA AMZN JPM XOM UNH \
    --start 2005-01-01 --walk-forward 4 --ml --backtest \
    --out results/patterns

# which validated patterns fire on today's bar?
python -m tradingagents.patterns.cli scan --patterns results/patterns/patterns.json \
    --symbols SPY QQQ IWM AAPL MSFT NVDA AMZN JPM XOM UNH
```

Outputs in `--out`:

| File | Content |
|---|---|
| `report.md` | Human-readable report: selection funnel, selected and rejected patterns, walk-forward, model, backtest |
| `report.json` | Everything in the report as data |
| `patterns.json` | Selected patterns only; input for `scan` and for the `get_validated_patterns` agent tool |
| `equity_test.csv`, `trades_test.csv` | Portfolio backtest on the test window (with `--backtest`) |

## Pipeline

1. **Data** (`data.py`). Adjusted daily bars from Yahoo Finance (default), a
   directory of CSV files, or QuantConnect's sample data on GitHub (no key,
   13 symbols, useful for tests). Parquet cache.
2. **Features** (`features.py`). About 55 point-in-time indicators per bar:
   multi-horizon returns, moving-average ratios and slopes, RSI (14 and 2),
   MACD, stochastic, CCI, ADX, Bollinger %B and width, ATR, realized
   volatility, volume ratios and z-scores, on-balance-volume slope,
   distance to 52-week high and low, streaks, gaps, intraday close location,
   crossover flags, and market-regime features from a proxy such as SPY.
   Labels are `fwd_h = Close[t+h] / Open[t+1] - 1`: a signal read at the
   close is filled at the next open.
3. **Screen** (`miner.py`). Candidate conditions are built from training-set
   quantiles plus classic thresholds (RSI 30/70, %B 0/1, ...). A beam search
   tests single conditions, then pairs and triples of the strongest ones.
   Each (pattern, horizon) is scored on *excess* return, the trade return minus
   the symbol's unconditional return for the same horizon, with standard
   errors clustered by holding-window block.
4. **Correct**. Benjamini-Hochberg false-discovery-rate control over every
   trial. The trial count also feeds the Deflated Sharpe Ratio reported per
   pattern.
5. **Confirm**. Survivors are traded exactly: one open position per symbol,
   next-open entry, fixed holding period, cost per side. Statistics are
   computed on the train window and on an embargoed validation window the
   screener never saw. Gates: FDR pass, minimum trades per year, positive
   net return *and* positive excess in train, and in validation a positive
   net return, positive excess with t >= 1, and win rate >= 50 %. Variants of
   one setup that fire on mostly the same bars (Jaccard overlap >= 0.7) are
   collapsed to the best representative.
6. **Test**. The final test window is evaluated once, for every candidate,
   and reported next to the validation numbers so you can see decay.
7. **Walk-forward** (`walkforward.py`, `--walk-forward N`). The *whole*
   procedure is re-run at N points in time and the selections are traded only
   on the data that followed. The pooled out-of-sample statistics are the
   honest estimate of what a trader re-running the engine would have earned.
8. **Model** (`ml.py`, `--ml`). A LightGBM classifier predicts whether the
   excess return will be positive, trained walk-forward with an embargo.
   Trading its top-decile predictions is reported per fold. A shallow
   surrogate tree turns the model's confident region into explicit rules that
   are pushed back through the exact simulator.
9. **Backtest** (`backtest.py`, `--backtest`). Portfolio simulation of the
   selected patterns on the test window against equal-weight buy and hold:
   CAGR, Sharpe, drawdown, exposure, trades per year.

## Reading the report

A pattern is worth attention only when all of these hold:

- it is **selected** (passed every gate);
- validation and test excess returns have the same sign;
- the Deflated Sharpe Ratio is high (0.95 is the usual bar); with several
  thousand trials this needs a per-trade t-statistic near 4;
- it fires often enough for your account: `trades_per_year` is counted over
  the whole universe with one position per symbol.

Patterns that are strong in train and flat or negative afterwards are the
normal outcome of data mining, and the engine lists them with the gate that
rejected them so you can see what was tried.

## Agent integration

`get_validated_patterns(symbol, curr_date)` in
`tradingagents/agents/utils/pattern_tools.py` is attached to the market
analyst. It reads `results/patterns/patterns.json` (override with the
`TRADINGAGENTS_PATTERNS_FILE` environment variable) and reports which
validated patterns are active on the trade date.

## Configuration knobs that matter

| Flag | Default | Effect |
|---|---|---|
| `--horizons` | `1,3,5,10,20` | holding periods in trading days |
| `--max-conditions` | 3 | depth of the conjunctions |
| `--beam` | 40 | breadth of the beam search; more = more trials = harsher correction |
| `--min-trades-per-year` | 6 | frequency gate over the whole universe |
| `--cost-bps` | 5 | per-side cost in basis points; 5 bps is realistic for liquid large caps with limit-at-open fills |
| `--fdr-q` | 0.10 | false-discovery rate |
| `--train-frac`, `--valid-frac` | 0.6, 0.2 | chronological split; the rest is test |
| `--long-only` | off | drop short-side patterns |

## Tests

```bash
python -m pytest tests -q
```

The tests use synthetic prices only. They check that features have no
look-ahead, that the trade simulator never overlaps positions, that the
backtest accounting closes, that the miner recovers a planted effect, and
that it selects (almost) nothing on pure noise.
