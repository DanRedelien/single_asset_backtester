"""
Reusable signal filters for single-asset strategies.

All filters are stateless and operate on pre-computed pandas Series so they
fit neatly into the vectorised pre-computation pattern inside BaseStrategy
__init__.  Each filter returns a boolean Series (True = trade allowed).

Available filters
─────────────────
VolatilityRegimeFilter  — blocks entries in low/high volatility regimes.
TrendFilter             — blocks mean-reversion entries during strong trends.
ADFFilter               — verifies stationarity of any price / spread series.
KalmanBeta              — Numba-compiled dynamic hedge-ratio estimator.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import adfuller


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def _kalman_beta_loop(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    Q: float,
    R: float,
) -> np.ndarray:
    """
    Numba-compiled 2-D Kalman Filter for rolling dynamic Beta estimation.

    State vector: [alpha, beta].  Runs in O(n) with ~50-100x speedup vs
    a pure Python loop of equivalent complexity.

    Args:
        x: Independent variable array (hedge / regressor).
        y: Dependent variable array (lead / regressand).
        n: Number of observations.
        Q: Process noise variance  (higher → faster adaptation).
        R: Measurement noise variance (higher → smoother estimate).

    Returns:
        beta_arr: Rolling beta estimates of length n.
    """
    beta_arr = np.zeros(n)
    a, b = 0.0, 1.0
    P00, P01, P10, P11 = 1.0, 0.0, 0.0, 1.0

    for t in range(n):
        xt, yt = x[t], y[t]

        # Predict
        P00 += Q
        P11 += Q

        # Innovation covariance
        S = P00 + xt * (P10 + P01) + xt * xt * P11 + R

        # Kalman gains
        K0 = (P00 + P01 * xt) / S
        K1 = (P10 + P11 * xt) / S

        error = yt - (a + b * xt)
        a += K0 * error
        b += K1 * error

        # Update covariance  (P = P - K*H*P)
        t00 = P00 - (K0 * P00 + K0 * xt * P10)
        t01 = P01 - (K0 * P01 + K0 * xt * P11)
        t10 = P10 - (K1 * P00 + K1 * xt * P10)
        t11 = P11 - (K1 * P01 + K1 * xt * P11)
        P00, P01, P10, P11 = t00, t01, t10, t11

        beta_arr[t] = b

    return beta_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Public filter classes
# ═══════════════════════════════════════════════════════════════════════════════


class VolatilityRegimeFilter:
    """
    Blocks trading when the asset is in a Compression or Expansion regime.

    Methodology:
        1. Compute a short-term rolling volatility (std of returns or prices).
        2. Rank it within a longer historical window to get a percentile score.
        3. Block entries below `min_pct` (compression) or above `max_pct`
           (expansion / panic), because mean-reversion edges vanish in both.

    The returned `allowed` Series is shifted back by one bar so that the
    strategy only sees past information — zero look-ahead bias.
    """

    def __init__(
        self,
        price: pd.Series,
        regime_window: int,
        history_window: int,
        min_pct: float = 0.20,
        max_pct: float = 0.80,
    ) -> None:
        """
        Args:
            price: Close prices (or any value series) indexed by bar timestamp.
            regime_window: Bars used for short-term volatility estimation.
            history_window: Bars used for percentile ranking of that volatility.
            min_pct: Lower percentile bound; entries blocked below this level.
            max_pct: Upper percentile bound; entries blocked above this level.
        """
        self.min_pct = min_pct
        self.max_pct = max_pct
        rolling_vol = price.rolling(
            window=regime_window, min_periods=regime_window // 2
        ).std()
        vol_pct = rolling_vol.rolling(
            window=history_window, min_periods=history_window // 2
        ).rank(pct=True)
        # Shift prevents look-ahead: strategy sees yesterday's regime label.
        self._pct: pd.Series = vol_pct.shift(1)

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the current bar's volatility regime permits trading.

        Args:
            timestamp: Current bar's index value.

        Returns:
            True if regime percentile is within [min_pct, max_pct].
        """
        pct = self._pct.get(timestamp, np.nan)
        if np.isnan(pct):
            return True  # No data yet → do not restrict
        return self.min_pct <= pct <= self.max_pct

    def as_series(self) -> pd.Series:
        """Returns raw percentile Series for inspection / overlay plotting."""
        return self._pct


class TrendFilter:
    """
    Blocks mean-reversion entries when a strong directional trend is detected.

    Methodology:
        Fits a rolling OLS regression: price ~ trend_index.
        Computes the T-statistic of the slope.  A high |T-stat| indicates
        that the price series is trending, making mean-reversion bets risky.

    Returns `is_allowed = True` only when |T-stat| < `max_t_stat`.
    """

    def __init__(
        self,
        price: pd.Series,
        window: int,
        max_t_stat: float = 2.0,
    ) -> None:
        """
        Args:
            price: Close price Series indexed by bar timestamp.
            window: Rolling regression window in bars.
            max_t_stat: Maximum absolute T-statistic to allow entry.
        """
        self.max_t_stat = max_t_stat
        t_arr = np.arange(len(price))
        t_series = pd.Series(t_arr, index=price.index)

        cov_st = price.rolling(window=window, min_periods=window // 2).cov(t_series)
        var_t  = t_series.rolling(window=window, min_periods=window // 2).var()
        slope  = cov_st / var_t
        intercept = (
            price.rolling(window=window, min_periods=window // 2).mean()
            - slope * t_series.rolling(window=window, min_periods=window // 2).mean()
        )
        residual = price - (intercept + slope * t_series)
        res_var  = residual.rolling(window=window, min_periods=window // 2).var()
        se_slope = np.sqrt(np.maximum(res_var / ((window - 1) * var_t), 0.0))
        t_stat   = slope / se_slope

        self._t_stat: pd.Series = t_stat.shift(1)

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the trend T-statistic is below the threshold.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if |T-stat| < max_t_stat (weak trend → mean-reversion viable).
        """
        t = self._t_stat.get(timestamp, np.nan)
        if np.isnan(t):
            return True
        return abs(t) < self.max_t_stat

    def as_series(self) -> pd.Series:
        """Returns raw T-statistic Series for inspection."""
        return self._t_stat


class ADFFilter:
    """
    Macro stationarity filter using the Augmented Dickey-Fuller (ADF) test.

    Methodology:
        Runs a rolling ADF test over a resampled lower-resolution Series.
        Blocks trading when the p-value exceeds `max_pvalue`, which indicates
        that the series has a unit root (non-stationary / trending) and that
        mean-reversion assumptions break down.

    Useful for Z-score / spread strategies on any single time-series.
    """

    def __init__(
        self,
        series: pd.Series,
        adf_window: int = 72,
        timeframe: str = "1h",
        max_pvalue: float = 0.05,
    ) -> None:
        """
        Args:
            series: Price / spread Series at the base bar resolution.
            adf_window: Number of resampled bars per ADF rolling window.
            timeframe: Resample frequency (e.g. '1h', '4h', '1D').
            max_pvalue: Block entries when ADF p-value exceeds this level.
        """
        self.max_pvalue = max_pvalue
        resampled = series.resample(timeframe, label="right", closed="right").last().dropna()
        pvalues = pd.Series(index=resampled.index, dtype=float, name="adf_pvalue")

        failures = 0
        for i in range(adf_window, len(resampled)):
            window_slice = resampled.iloc[i - adf_window : i]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = adfuller(window_slice.values, maxlag=1, autolag=None)
                    pvalues.iloc[i] = result[1]
            except Exception:
                failures += 1
                pvalues.iloc[i] = np.nan

        if failures > 0:
            print(f"[ADFFilter] ADF fit failed on {failures} windows (NaN assigned).")

        self._pvalue: pd.Series = (
            pvalues.reindex(series.index, method="ffill").shift(1).reindex(series.index, method="ffill")
        )

    def is_allowed(self, timestamp) -> bool:
        """
        Returns True when the ADF test indicates a stationary regime.

        Args:
            timestamp: Current bar index value.

        Returns:
            True if ADF p-value < max_pvalue (reject unit root → stationary).
        """
        pval = self._pvalue.get(timestamp, np.nan)
        if np.isnan(pval):
            return False  # No result yet → block by default (conservative)
        return pval < self.max_pvalue

    def as_series(self) -> pd.Series:
        """Returns raw p-value Series for inspection."""
        return self._pvalue


class KalmanBeta:
    """
    Dynamic hedge-ratio estimator using a Numba-accelerated Kalman Filter.

    Suitable for any two co-moving price series where the relationship may
    drift over time (e.g. intraday beta to SPY, currency pair correlation).
    Wraps the compiled `_kalman_beta_loop` with scale-normalisation to keep
    Q and R numerically stable regardless of price magnitude.
    """

    def __init__(
        self,
        x: pd.Series,
        y: pd.Series,
        Q: float = 1e-5,
        R: float = 1e-1,
    ) -> None:
        """
        Args:
            x: Independent variable price Series (regressor / hedge leg).
            y: Dependent variable price Series (regressand / lead leg).
            Q: Process noise variance.  Larger → beta adapts faster.
            R: Measurement noise variance.  Larger → beta smoother.
        """
        common_idx = x.index.intersection(y.index)
        x_a = x.loc[common_idx].values.astype(float)
        y_a = y.loc[common_idx].values.astype(float)

        # Scale to unit magnitude to ensure Q/R remain numerically meaningful.
        scale = float(x_a[0]) if x_a[0] != 0 else 1.0
        beta_arr = _kalman_beta_loop(x_a / scale, y_a / scale, len(x_a), Q, R)

        self._beta: pd.Series = pd.Series(beta_arr, index=common_idx).shift(1)

    def get(self, timestamp, default: float = 1.0) -> float:
        """
        Returns the beta estimate for a given bar timestamp.

        Args:
            timestamp: Current bar index value.
            default: Fallback value when no estimate is available.

        Returns:
            Float beta (hedge ratio).
        """
        val = self._beta.get(timestamp, np.nan)
        return default if np.isnan(val) else float(val)

    def as_series(self) -> pd.Series:
        """Returns the full beta Series for inspection or plotting."""
        return self._beta
