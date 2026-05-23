"""Quick sanity check that the data, utils, and risk metrics all work.

Run with:  python smoke_test.py
"""

import utils as U


panel = U.load_panel()
rets = U.log_returns(panel)
btc = panel["Bitcoin"].dropna()

print(f"Loaded {panel.shape[1]} coins from {panel.index.min().date()} to {panel.index.max().date()}")
print(f"  total days: {len(panel)}")
print()
print("Per-coin coverage:")
for coin, count in panel.notna().sum().sort_values().items():
    print(f"  {coin:14s} {count:>5} days")
print()
print(f"BTC metrics:")
print(f"  Sharpe:       {U.sharpe(rets['Bitcoin'].dropna()):.3f}")
print(f"  Sortino:      {U.sortino(rets['Bitcoin'].dropna()):.3f}")
print(f"  Max drawdown: {U.max_drawdown(btc)*100:.1f}%")
print(f"  Calmar:       {U.calmar(btc):.3f}")
print()
no_tether = panel.drop(columns=["Tether"], errors="ignore")
apc = U.avg_pairwise_corr(U.log_returns(no_tether).dropna(), window=60).dropna()
print(f"Cross-coin contagion:")
print(f"  Avg pairwise correlation (60d window):")
print(f"    mean={apc.mean():.2f}  min={apc.min():.2f}  max={apc.max():.2f}")
print()
print("All checks passed.")
