# Delta Hedging & Adversarial Stock Path Generation Simulator

A Python/PyTorch simulator that studies **option hedging under transaction costs** by comparing classical Black–Scholes delta hedging with a **neural-network hedger** trained on a CVaR objective. The neural hedger is tested on synthetic price paths from three generators: **GBM**, **stress scenarios**, and an **adversarial generator** in the spirit of *Adversarial Deep Hedging* (Hirano, Minami & Imajo, 2023, [arXiv:2307.13217](https://arxiv.org/abs/2307.13217)).

Built during the AlgoZenith HFT Bootcamp (May–Jun 2026).

---

## The problem

An option seller (the *hedger*) is short a call and trades the underlying to offset the payoff risk. Black–Scholes gives the ideal position (delta), but in practice:

- rebalancing is **discrete**, not continuous, so residual risk remains;
- every trade pays a **transaction cost**, so trading more often is not free.

Deep hedging replaces the delta formula with a neural network that outputs the hedge position directly and is trained to minimize tail risk (CVaR) of the final profit & loss. Its quality depends heavily on the price paths it is trained on, which is where the adversarial generator comes in.

## Method

**P&L of the short call (per path)**

```
PL = -max(S_T - K, 0)  +  sum_t delta_t * (S_{t+1} - S_t)  -  sum_t c * S_t * |delta_t - delta_{t-1}|
       option payout        trading gains                        transaction costs
```

**Hedger (neural network).** A small MLP that takes the current price, the Black–Scholes delta and the previous position, and outputs the new position `delta_t` (hidden size 32, tanh). Feeding back `delta_{t-1}` is what lets the network learn to trade less when costs make small adjustments unprofitable.

**Loss.** CVaR at the 5% level: the average of the worst 5% of P&L outcomes in a batch.

**Path generators** (vectorized NumPy)

| Generator | Description |
|---|---|
| GBM | Geometric Brownian motion with constant volatility (baseline). |
| Stress | Regime switch to higher volatility / negative drift mid-path, optionally with Merton-style jumps. |
| Adversarial | A GRU network outputs a per-step volatility `sigma_t` in [0.1, 1.5]. It is trained to *maximize* the hedger's CVaR loss while the hedger trains to minimize it, using alternating updates (5 hedger steps : 1 generator step). |

The volatility clip on the generator addresses a known issue from the paper: without a bound, the adversary can push volatility arbitrarily high.

**Real data.** Minute-level NIFTY futures and options (expiry day 5 Feb 2026 and a non-expiry day, 4 Feb 2026). Used to compute implied volatility, Greeks, and a real-market delta-hedging run with mark-to-market (1-minute vs. 15-minute rebalancing).

## Repository contents

| File | Purpose |
|---|---|
| `HFT_CAMP_CODE.ipynb` | Main notebook: real-data analysis, path generators, hedging engine, neural hedger, adversarial training. |
| `20260205_option_minute_prices_expiry.csv` | Minute option/futures prices, expiry day. |
| `20260204_option_minute_prices_non_expiry.csv` | Minute option/futures prices, non-expiry day. |

## How to run

The notebook was developed in Google Colab.

1. Open `HFT_CAMP_CODE.ipynb` in Colab.
2. Upload the expiry-day CSV when the first cell asks for it.
3. Run the cells in order (dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`).

## What is implemented

- Black–Scholes pricing, Greeks, and implied-volatility solvers (bisection and vectorized Newton–Raphson)
- Real-data delta hedging with mark-to-market portfolio tracking
- Vectorized GBM, stress (regime switch + jumps), and adversarial path generators
- Discrete-time hedging engine with configurable rebalancing steps (50 / 100 / 250) and proportional transaction costs
- P&L computation, CVaR / VaR / downside-deviation metrics
- PyTorch neural hedger with CVaR training loop
- GRU adversarial generator with alternating min–max training

## Results so far

All P&L figures below are for a **short call** and exclude the premium received, so the mean is roughly minus the option price. "Hedging error" is the standard deviation of P&L.

**1. Sanity check: frictionless GBM, normalized neural hedger vs. Black–Scholes**
(S0 = K = 100, sigma = 0.2, T = 1 year, 50 rebalancing steps; trained on 20,000 paths, tested on 10,000 unseen paths)

| | Mean P&L | Hedging error (std) | VaR 95% | CVaR 95% |
|---|---|---|---|---|
| Black–Scholes delta | -10.91 | 0.97 | 12.51 | 13.19 |
| Normalized deep hedger | -10.90 | 1.46 | 13.39 | 13.87 |

With normalized inputs (moneyness, time to expiry, BS delta, previous position) and a scaled CVaR loss, the neural hedger comes within about 5% of Black–Scholes on CVaR. On frictionless GBM, Black–Scholes is essentially optimal, so matching it is the expected outcome.

**2. NIFTY-scale benchmark: 10,000 GBM paths, sigma = 0.6, 374 one-minute steps (prices in paise)**
This uses the *earlier, unnormalized* hedger.

| Strategy | Mean P&L | Hedging error (std) | CVaR 5% | Downside dev |
|---|---|---|---|---|
| Black–Scholes (no costs) | -21,069 | 727 | 22,759 | 521 |
| Neural network (no costs) | -21,062 | 12,255 | 53,601 | 9,929 |
| Black–Scholes (c = 0.005) | -106,048 | 30,097 | 162,139 | 21,540 |
| Neural network (c = 0.005) | -28,922 | 12,255 | 61,461 | 9,929 |

This hedger did **not** learn a useful policy: without costs its hedging error is about 17x that of Black–Scholes. Its std is identical with and without costs, meaning its position barely changes after the first step (saturated tanh layers from unnormalized inputs). It "wins" at c = 0.005 only because that cost rate (0.5% per unit traded) is so high that the constant-position strategy avoids it. This is a training-setup problem that the normalized model above addresses; re-running the NIFTY-scale benchmark with the normalized hedger is the next step.

## Status and known limitations

- **Done:** real-data analysis, three path generators, hedging engine with costs, CVaR metrics, normalized neural hedger validated against Black–Scholes on frictionless GBM, GRU adversarial generator with alternating training.
- **Adversarial training is a short demonstration** (10 alternating epochs; the hedger is not given the BS-delta input in this loop). Losses are logged, but a full evaluation of the adversarially trained hedger against the baselines is pending.
- **Rebalancing-interval experiments (50 / 100 / 250 steps)** currently train and log losses. A full evaluation table across step counts is pending.
- **Neural hedger with transaction costs** has not yet been validated with the normalized model, and realistic cost rates (about 1e-4) still need to be tested.
- **Stress paths** are implemented and visualized but not yet used to evaluate the hedgers.
- The notebook still contains some exploratory cells from earlier iterations.

## Roadmap

- [x] Normalize inputs and scale the loss; verify the hedger approaches Black–Scholes on frictionless GBM
- [ ] Re-run the NIFTY-scale benchmark with the normalized hedger and a realistic cost rate
- [ ] Evaluate {GBM, stress, adversarial} x {BS, NN, adversarial NN} x {50, 100, 250 steps}
- [ ] Clean up the notebook into a linear, reproducible pipeline

## Reference

M. Hirano, K. Minami, K. Imajo. *Adversarial Deep Hedging: Learning to Hedge without Price Process Modeling.* arXiv:2307.13217, 2023.
H. Buehler, L. Gonon, J. Teichmann, B. Wood. *Deep Hedging.* Quantitative Finance, 2019.
