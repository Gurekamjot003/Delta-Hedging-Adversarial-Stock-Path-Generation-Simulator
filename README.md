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

## Status and known limitations

This is a work in progress. Being explicit about where it stands:

- **Neural hedger training is not yet stable.** Inputs are currently unnormalized (prices are in the millions), which saturates the tanh layers, and the CVaR is estimated from small batches. Normalizing inputs (S/S0, time to expiry) and larger batches is the next step. Until then, the neural hedger does not match Black–Scholes.
- **The full benchmark table is pending.** The comparison of Black–Scholes vs. the neural hedger over 10,000 paths, across all three generators and rebalancing frequencies, has not been completed.
- **Adversarial hedger.** The generator and the alternating training loop are implemented; longer training and a proper evaluation against the baselines are pending.
- **Transaction cost scale.** Experiments so far used a very high cost rate; realistic values (about 1e-4) are planned.
- The notebook still contains some legacy exploratory cells from earlier iterations.

## Roadmap

- [ ] Normalize inputs and scale P&L; verify the neural hedger matches Black–Scholes on frictionless GBM
- [ ] Run the benchmark: {GBM, stress, adversarial} × {BS, NN, adversarial NN} × {50, 100, 250 steps} × {no costs, costs}
- [ ] Report mean P&L, hedging error (std of P&L + premium), CVaR and downside deviation, with P&L histograms
- [ ] Clean up the notebook into a linear, reproducible pipeline

## Reference

M. Hirano, K. Minami, K. Imajo. *Adversarial Deep Hedging: Learning to Hedge without Price Process Modeling.* arXiv:2307.13217, 2023.
H. Buehler, L. Gonon, J. Teichmann, B. Wood. *Deep Hedging.* Quantitative Finance, 2019.
