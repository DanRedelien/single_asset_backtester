"""
Strategies package.

Available strategies:
  SmaCrossoverStrategy  — Dual SMA trend following with ATR-scaled SL/TP.
  MeanReversionStrategy — RSI + Bollinger Bands with optional regime filters.

Reusable components in filters.py:
  VolatilityRegimeFilter, TrendFilter, ADFFilter, KalmanBeta.
"""

from src.strategies.base import BaseStrategy
from src.strategies.sma_crossover import SmaCrossoverStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

__all__ = [
    "BaseStrategy",
    "SmaCrossoverStrategy",
    "MeanReversionStrategy",
]
