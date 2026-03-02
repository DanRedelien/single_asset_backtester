from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import scipy.stats as stats

from src.backtest_engine.settings import get_settings

class PerformanceMetrics:
    """
    Calculates performance metrics for the backtest.
    Supports regime-based breakdowns (Calm / Stress / All).
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize PerformanceMetrics.
        
        Args:
            risk_free_rate: Annualized risk-free rate for Sharpe calculation.
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        portfolio_history: pd.DataFrame,
        trades: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculates key performance indicators including Sortino, Calmar, and Trade stats.
        
        Args:
            portfolio_history: DataFrame containing 'total_value' column indexed by date.
            trades: List of trade objects with 'pnl' and 'entry_time' attributes.
            
        Returns:
            Dictionary of metrics.
        """
        if portfolio_history.empty:
            return {}

        df = portfolio_history.copy()
        df['returns'] = df['total_value'].pct_change().fillna(0)
        
        # Basic Stats
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1
        
        # CAGR
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25 if days > 0 else 1.0
        cagr = max(0.0, 1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        # Annualization factor for intraday returns
        bars_per_year = len(df) / years if years > 0 else 252

        # Volatility (Annualized)
        std_dev = df['returns'].std() * np.sqrt(bars_per_year)

        # Sharpe Ratio
        sharpe = (cagr - self.risk_free_rate) / std_dev if std_dev > 0 else 0.0

        # Sortino Ratio (Downside Deviation)
        downside_returns = df.loc[df['returns'] < 0, 'returns']
        downside_std = downside_returns.std() * np.sqrt(bars_per_year)
        sortino = (cagr - self.risk_free_rate) / downside_std if downside_std > 0 else 0.0

        # Drawdown
        running_max = df['total_value'].cummax()
        drawdown = (df['total_value'] - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        metrics = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': std_dev,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar
        }

        # Trade Analytics
        if trades:
            metrics.update(self._calculate_trade_stats(trades))

        return metrics



    def _calculate_trade_stats(self, trades: List[Any]) -> Dict[str, float]:
        """Calculates statistics based on list of trades."""
        if not trades:
            return {
                'Total Trades': 0,
                'Win Rate': 0.0,
                'Profit Factor': 0.0,
                'Avg Trade': 0.0,
                'T-Statistic': 0.0,
                'P-Value': 1.0
            }
            
        # Handle both dicts and objects for flexibility
        pnls = []
        for t in trades:
            if isinstance(t, dict):
                pnls.append(t.get('pnl', 0.0))
            else:
                pnls.append(getattr(t, 'pnl', 0.0))
                
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        
        total_trades = len(trades)
        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0
        
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # T-statistic & p-value for PnL
        t_stat = 0.0
        p_val = 1.0
        if total_trades > 1 and np.std(pnls) > 0:
            t_stat, p_val = stats.ttest_1samp(pnls, 0.0)

        return {
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Avg Trade': sum(pnls) / total_trades if total_trades > 0 else 0.0,
            'Avg Win': sum(winners) / len(winners) if winners else 0.0,
            'Avg Loss': sum(losers) / len(losers) if losers else 0.0,
            'T-Statistic': t_stat,
            'P-Value': p_val
        }

    def print_full_report(self, metrics: Dict[str, float], trades: List[Any]):
        """Prints a comprehensive, formatted backtest report to the terminal."""
        if not metrics:
            print("No metrics to display.")
            return

        # Calculate hold times
        hold_times = []
        if trades:
            for t in trades:
                if hasattr(t, 'entry_time') and hasattr(t, 'exit_time'):
                    hold_times.append(t.exit_time - t.entry_time)

        if hold_times:
            avg_hold = sum(hold_times, pd.Timedelta(0)) / len(hold_times)
            max_hold = max(hold_times)
            min_hold = min(hold_times)
        else:
            avg_hold = max_hold = min_hold = pd.Timedelta(0)

        # Helper to format timedelta cleanly
        def fmt_td(td):
            if pd.isna(td): return "N/A"
            total_sec = int(td.total_seconds())
            days, remainder = divmod(total_sec, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)
            if days > 0:
                return f"{days}d {hours}h {minutes}m"
            return f"{hours}h {minutes}m"

        COL_W = 16
        LABEL_W = 20
        sep = "─" * (LABEL_W + COL_W + 4)

        def fmt(value, is_pct=False, is_money=False, is_int=False):
            if pd.isna(value) or value is None: return "NaN"
            if is_int: return f"{int(value):,}"
            if is_pct: return f"{value:.2%}"
            if is_money: return f"${value:,.0f}"
            return f"{value:.4f}"

        print("\n" + sep)
        print(f"{'BACKTEST RESULTS':^{LABEL_W + COL_W + 4}}")
        print(sep)

        # 1. Core Performance
        core_rows = [
            ("Total Return", metrics.get('Total Return'), dict(is_pct=True)),
            ("CAGR",         metrics.get('CAGR'),         dict(is_pct=True)),
            ("Volatility",   metrics.get('Volatility'),   dict(is_pct=True)),
            ("Sharpe Ratio", metrics.get('Sharpe Ratio'), {}),
            ("Sortino Ratio",metrics.get('Sortino Ratio'),{}),
            ("Max Drawdown", metrics.get('Max Drawdown'), dict(is_pct=True)),
            ("Calmar Ratio", metrics.get('Calmar Ratio'), {}),
        ]
        for label, val, args in core_rows:
            print(f"{label:<{LABEL_W}}{fmt(val, **args):>{COL_W}}")
        
        print(sep)

        # 2. Trade Statistics
        trade_rows = [
            ("Total Trades",   metrics.get('Total Trades', 0), dict(is_int=True)),
            ("Win Rate",       metrics.get('Win Rate', 0),    dict(is_pct=True)),
            ("Profit Factor",  metrics.get('Profit Factor', 0),{}),
            ("Avg Trade ($)",  metrics.get('Avg Trade', 0),   dict(is_money=True)),
        ]
        
        total_pnl = sum([getattr(t, 'pnl', 0.0) if not isinstance(t, dict) else t.get('pnl', 0.0) for t in (trades or [])])
        trade_rows.append(("Total PnL ($)", total_pnl, dict(is_money=True)))
        trade_rows.append(("Avg Win ($)", metrics.get('Avg Win', 0), dict(is_money=True)))
        trade_rows.append(("Avg Loss ($)", metrics.get('Avg Loss', 0), dict(is_money=True)))
        trade_rows.append(("T-Statistic", metrics.get('T-Statistic', 0), {}))
        trade_rows.append(("P-Value", metrics.get('P-Value', 1), {}))

        for label, val, args in trade_rows:
            print(f"{label:<{LABEL_W}}{fmt(val, **args):>{COL_W}}")

        print(sep)

        # 3. Hold Times
        print(f"{'Max Hold Time':<{LABEL_W}}{fmt_td(max_hold):>{COL_W}}")
        print(f"{'Min Hold Time':<{LABEL_W}}{fmt_td(min_hold):>{COL_W}}")
        print(f"{'Avg Hold Time':<{LABEL_W}}{fmt_td(avg_hold):>{COL_W}}")
        print(sep + "\n")
