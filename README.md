# Forecasting Next-Day Bank Stock Prices: ARIMA vs. a Neural Network

Does a neural network beat a classical time-series model at forecasting next-day closing prices? This project answers that for three large US banks over six years of daily data, with both models scored on **identical test dates** so the comparison actually means something.

**Answer: no.** ARIMA wins, and the more expensive ARIMA variant doesn't beat the cheap one either.

---

## Result

Both models evaluated on the intersection of their test windows:

| Metric | ARIMA | MLP |
|---|---|---|
| RMSE (USD) | **2.53** | 2.59 |
| MAE (USD) | **1.95** | 2.03 |
| MAPE | **1.36%** | 1.41% |

Static vs. dynamic ARIMA, on the ARIMA test window:

| Variant | RMSE | MAE | MAPE | Relative runtime |
|---|---|---|---|---|
| Static order fixed | **1.8196** | **1.3329** | **0.955%** | 1x |
| Dynamic, order re-searched each step | 1.8239 | 1.3373 | 0.958% | ~25x |

Re-estimating ARIMA order at every forecast step costs roughly 25 times the compute and returns nothing. Worth establishing before it goes into a production pipeline.

MLP architecture selection, on validation MSE:

| Architecture | Hidden layers | Train MSE | Val MSE |
|---|---|---|---|
| **Deep** | 64, 64, 64 | 0.000978 | **0.001417** |
| Medium | 64, 32 | 0.001035 | 0.001543 |
| Large | 128, 64, 32 | 0.000916 | 0.001593 |
| Small | 32 | 0.001358 | 0.001952 |

Note the Large model has the lowest *training* error and the second-worst *validation* error - the clearest overfitting signal in the set, and the reason selection runs on the validation split rather than on training loss.

---

## Data

| | |
|---|---|
| Universe | JPMorgan Chase (JPM), Bank of America (BAC), Wells Fargo (WFC) |
| Period | Jan 2018 - Jan 2024 (~1,500 trading days) |
| Source | Yahoo Finance via yfinance |
| Fields | Open, High, Low, Close, Adj Close, Volume |
| Target | Next-day closing price |
| Modelled series | JPM |

The window deliberately spans the COVID crash, the 2022 rate-hiking cycle, and the March 2023 regional banking stress - three regime changes that stress-test whether a model generalises or just fits a calm market.

---

## Method

**1. Data acquisition and cleaning**
Missing values are counted against a full calendar-day grid, not the trading-day index. This separates genuine data gaps from weekends and market holidays - only the former justify interpolation. Gaps are filled by linear interpolation between trading days; cleaned frames are cached to CSV.

**2. Exploratory and technical analysis**
14- and 30-day moving averages, expanding mean, and a volatility band at plus/minus one rolling standard deviation, used to characterise trend persistence and locate the volatility regimes before any modelling.

**3. MACD**
Implemented from first principles rather than pulled from a library, so every parameter is explicit: MACD is EMA(12) minus EMA(26) of the close, Signal is the 9-period EMA of MACD, and the Histogram is their difference. All three components feed the neural network as momentum features.

**4. MLP forecasting**
10 lagged closing prices plus the three MACD components. Chronological 70/15/15 split - no shuffling, which would leak future information into training. MinMaxScaler fitted on the training split only, features and target scaled separately. Four architectures trained for 200 epochs on a shared seed, compared on learning curves and validation MSE, with the winner evaluated once on the held-out test set.

**5. ARIMA benchmark**
ACF/PACF diagnostics, then an AIC-based order search. 80/20 chronological split. Walk-forward validation: the model is refit at each test step on the *actual* observed history, not on its own predictions, so forecast errors do not compound the way they would in recursive multi-step forecasting.

**6. Head-to-head**
The two models were developed on different splits, so their test windows do not coincide. All comparison metrics are recomputed on the **intersection** of the two date ranges. Residuals are compared over time and by distribution, to check whether the models fail in the same periods (common cause) or different ones (where an ensemble would help).

---

## Keeping the comparison honest

Three decisions do most of the work, and each is defended inline in the notebook:

- **No look-ahead bias.** Every feature row uses information available up to and including day t; the target is the close on day t+1. Scalers see training data only.
- - **Walk-forward, not recursive.** ARIMA refits on observed history at each step, so errors do not compound.
  - - **The test set is used exactly once.** Architecture selection happens on validation. Any earlier peek at test performance would turn it into a second validation set.
   
    - ---

    ## Interpretation

    ARIMA wins because JPM's price series is well described by a short-memory linear process with an integrated component - precisely the structure ARIMA is built for. The MLP's momentum features add value in trending sub-periods where momentum persists, but contribute noise rather than signal in range-bound or shock-driven windows, which is where its residuals are largest.

    The practical read: for daily operational forecasting where interpretability and validation matter, ARIMA is the right starting point. The MLP earns its place as a complement - flagging momentum divergences, or as the base for a richer feature set (macro indicators, sentiment) where non-linearity is more likely to pay.

    Neither model can anticipate earnings surprises or macro shocks. Both are decision support, not decisions.

    ---

    ## Running it

    Clone the repo, install dependencies from requirements.txt, then open the notebook in Jupyter. It runs top to bottom. The dynamic ARIMA cell is the slow one - it runs a full order search at every test step and takes several minutes.

    Outputs are committed with the notebook, so the analysis is readable on GitHub without executing anything.

    ---

    ## Stack

    pandas, numpy, matplotlib, yfinance, scikit-learn (MLPRegressor, MinMaxScaler), statsmodels (ARIMA, ACF/PACF), pmdarima (auto_arima)

    ---

    ## About

    Built by **Tien Nam Luong** - MSc Finance & Investment, University of Leeds. Background in corporate credit and project finance; this project is part of moving that work from spreadsheets into code.

    luongnampk34@gmail.com
    
