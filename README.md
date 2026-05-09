# Portfolio Optimization with Pyomo

## Overview
Mixed-Integer Linear Programming (MILP) model for financial portfolio  optimization, implemented in Python using **Pyomo** and solved with **CBC**.

## Problem
Select the optimal allocation among 8 assets (AAPL, MSFT, GOOG, SAP, RGTI, TSLA, GLD, BTC) maximizing risk-adjusted return under:
- Capital budget constraint (B = $10,000)
- Diversification constraint (min K assets)
- Min/max allocation bounds per asset
- Optional: Binary exclusivity (BTC XOR TSLA)
- Risk Measures: MAD (Mean Absolute Deviation) |

## Results
| Alpha | Return | Risk (MAD) |
|-------|--------|------------|
| 0.9   | 3.1%   | 0.0813     |

## Tech Stack
- Python, Pyomo, CBC solver
- yfinance (data), NumPy, pandas

