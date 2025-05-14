📈 Quantitative Factor Research: Three Intraday Alpha Signals
This repository contains three alpha factors constructed from minute-level and auction-phase data in the Chinese A-share market. Each factor targets a specific holding period and is designed to capture short-term alpha from intraday market microstructure.

🔍 Factor 1: Intraday Spread Volatility Ratio
Formula:

Mean(high - low) / Std(high - low) on minute-level bars per stock per day

Intuition:

This factor measures the stability of intraday price ranges. A high ratio indicates consistent but wide price movement, possibly reflecting hidden liquidity or passive accumulation.

Usage & Backtest Logic:

Data Used: Minute-level data during trading hours

Signal Time: Calculated by market close each day

Trade Logic: Buy at close, sell at next open

Target: Predicts overnight return (close-to-open)

Sharpe Ratio: 1.805

⚖️ Factor 2: Pre-market VWAP Imbalance Indicator
Formula:

(VWAP of executed buy orders − Match Price)
÷
(Match Price − VWAP of executed sell orders)

Explanation:

This factor captures imbalance in trading pressure during the call auction phase (typically 9:15–9:25). A higher value reflects more aggressive buyer participation.

Usage & Backtest Logic:

Data Used: Order book and matched trade data during auction

Signal Time: Available before market open

Trade Logic: Buy at open, sell at next open

Target: Predicts open-to-open return

Note: Performance remains significant even when buying at the first minute post-open (to account for slippage)

Sharpe Ratio: 1.245

🧪 Factor 3: VWAP Expansion Factor (Exponential Range Signal)
Formula:

(high - vwap)^{(vwap - low)}
]

Construction Logic:

Originally tested as a multiplication of two positively correlated signals, this design was later improved using an exponentiation to amplify extreme co-movements. The transformation significantly enhanced predictive power.

Usage & Backtest Logic:

Data Used: Intraday minute-level OHLC and VWAP

Signal Time: Calculated at market close

Trade Logic: Buy at close, sell at next open

Target: Predicts overnight return (close-to-open)

Sharpe Ratio: 2.798

🧾 Performance Summary
Factor Name	Signal Type	Trade Logic	Sharpe Ratio
Intraday Spread Volatility Ratio	Minute (Intraday)	Close → Next Open	1.805
Pre-market VWAP Imbalance Indicator	Auction (Pre-open)	Open → Next Open	1.245
VWAP Expansion Factor	Minute (Intraday)	Close → Next Open	2.798

🗂️ Repository Structure
factor1.ipynb: Spread volatility ratio implementation

factor2.ipynb: VWAP imbalance indicator from call auction

factor3.ipynb: Exponential breakout-style signal using VWAP deviation

README.md: Description and performance summary of all factors

📌 Notes
All signals are constructed without lookahead bias

Designed for short-horizon, daily rebalancing strategies

Further improvements possible through signal combination, risk-adjusted scaling, and cross-validation

