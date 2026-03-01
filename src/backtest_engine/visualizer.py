"""
Institutional-grade backtest dashboard (single-asset).

Layout (3 rows):
  Row 0 (tall)  : Equity Curve — strategy vs Buy-and-Hold benchmark
  Row 1 (medium): Drawdown %
  Row 2 (medium): P&L Distribution (left) | Exit Breakdown table (right)
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


class Visualizer:
    """
    Renders a clean, generic 3-panel backtest dashboard.

    No pairs-/spread-specific panels are included.  The Z-score overlay
    and lead-benchmark lines have been removed; any strategy-specific
    visualisation should be done by the strategy itself after calling
    plot_dashboard().
    """

    # ── Color palette ──────────────────────────────────────────────────────────
    C: dict = {
        "combined":  "#2980B9",   # Blue  – strategy equity curve
        "long":      "#27AE60",   # Green – long-only cumulative P&L
        "short":     "#E74C3C",   # Red   – short-only cumulative P&L
        "bench":     "#BDC3C7",   # Gray  – buy-and-hold benchmark
        "dd_fill":   "#FADBD8",   # Light-red fill for drawdown area
        "dd_line":   "#E74C3C",   # Red line for drawdown curve
        "winner":    "#27AE60",   # Green P&L histogram bars
        "loser":     "#E74C3C",   # Red P&L histogram bars
        "header":    "#D6E4F0",   # Table header background
        "row_alt":   "#F8F9FA",   # Alternating row background
        "text":      "#2C3E50",
        "spine":     "#BDC3C7",
        "grid":      "#ECF0F1",
    }

    # Exit reason → background colour for the exit table
    _EXIT_COLORS: dict = {
        "STOP_LOSS":   "#FADBD8",   # Red   – loss
        "TAKE_PROFIT": "#D5F5E3",   # Green – profit
        "TIME_STOP":   "#FDEBD0",   # Orange – timed out
        "EOD_CLOSE":   "#EBF5FB",   # Light blue – forced close
        "SIGNAL":      "#F9F9F9",   # Neutral
    }

    def __init__(self) -> None:
        self._apply_style()

    # ── Style helpers ──────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        plt.rcParams.update(
            {
                "font.family":      "sans-serif",
                "axes.facecolor":   "#FFFFFF",
                "figure.facecolor": "#FFFFFF",
                "text.color":       self.C["text"],
                "axes.labelcolor":  self.C["text"],
                "xtick.color":      self.C["text"],
                "ytick.color":      self.C["text"],
                "grid.color":       self.C["grid"],
                "grid.alpha":       0.6,
                "grid.linestyle":   "-",
                "axes.spines.top":  False,
                "axes.spines.right":False,
                "axes.edgecolor":   self.C["spine"],
            }
        )

    def _watermark(self, fig) -> None:
        fig.text(
            0.99, 0.005,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ha="right", fontsize=7, color="#95A5A6",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def plot_dashboard(
        self,
        portfolio_history: pd.DataFrame,
        trades: Optional[List] = None,
        benchmark: Optional[pd.Series] = None,
    ) -> None:
        """
        Renders the full 3-panel backtest dashboard.

        Args:
            portfolio_history: DataFrame with 'total_value' indexed by timestamp.
            trades: List of Trade dataclass objects from ExecutionHandler.
            benchmark: Optional price Series for buy-and-hold overlay (gray dashed).
        """
        trades = trades or []

        fig = plt.figure(figsize=(16, 11))
        fig.suptitle(
            "Backtest Dashboard",
            fontsize=13, fontweight="bold",
            x=0.01, ha="left", color=self.C["text"],
        )

        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[2.2, 1.0, 1.5],
            hspace=0.40, wspace=0.08,
        )

        ax_eq  = fig.add_subplot(gs[0, :])
        ax_dd  = fig.add_subplot(gs[1, :], sharex=ax_eq)
        ax_pnl = fig.add_subplot(gs[2, 0])
        ax_tbl = fig.add_subplot(gs[2, 1])

        self._draw_equity(ax_eq, portfolio_history, trades, benchmark)
        self._draw_drawdown(ax_dd, portfolio_history)
        self._draw_pnl_hist(ax_pnl, trades)
        self._draw_exit_table(ax_tbl, trades)

        self._watermark(fig)
        plt.show()

    # ── Panel: Equity Curve ────────────────────────────────────────────────────

    def _draw_equity(
        self,
        ax,
        history: pd.DataFrame,
        trades: List,
        benchmark: Optional[pd.Series],
    ) -> None:
        initial_cap = history["total_value"].iloc[0]
        idx = history.index

        # Buy-and-hold benchmark (gray dashed)
        if benchmark is not None:
            b = benchmark if isinstance(benchmark, pd.Series) else benchmark.iloc[:, 0]
            common = idx.intersection(b.index)
            if len(common) > 1:
                b_norm = (b.loc[common] / b.loc[common].iloc[0]) * initial_cap
                ax.plot(
                    common, b_norm,
                    color=self.C["bench"], linewidth=1.0, alpha=0.6,
                    linestyle="--", label="B&H (Buy-and-Hold)", zorder=1,
                )
                ax.fill_between(
                    common, b_norm, initial_cap,
                    color=self.C["bench"], alpha=0.10, zorder=0,
                )

        # Cumulative P&L by direction (long / short decomposition)
        if trades:
            t_df = pd.DataFrame(
                [
                    {"direction": t.direction, "exit_time": t.exit_time, "pnl": t.pnl}
                    for t in trades
                ]
            )
            for direction, color, label in [
                ("LONG",  self.C["long"],  "Long"),
                ("SHORT", self.C["short"], "Short"),
            ]:
                sub = t_df[t_df["direction"] == direction].copy()
                if sub.empty:
                    continue
                pnl_s = sub.set_index("exit_time")["pnl"].sort_index()
                pnl_s = pnl_s.groupby(level=0).sum()
                full_idx = idx.union(pnl_s.index)
                cum = pnl_s.reindex(full_idx, fill_value=0).reindex(idx, fill_value=0).cumsum()
                ax.plot(idx, cum + initial_cap, color=color, linewidth=1.2,
                        alpha=0.85, label=label, zorder=3)

        # Strategy equity (on top)
        ax.plot(
            idx, history["total_value"],
            color=self.C["combined"], linewidth=1.8,
            label="Strategy", zorder=4,
        )
        ax.axhline(initial_cap, color=self.C["spine"], linewidth=0.8,
                   linestyle="--", alpha=0.6)

        ax.set_title("Equity Curve", fontweight="bold", loc="left", fontsize=11, pad=8)
        ax.set_ylabel("Value ($)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(frameon=False, fontsize=7, loc="lower left",
                  ncol=4, columnspacing=0.8, handlelength=1.2)
        ax.grid(True, alpha=0.4)

    # ── Panel: Drawdown ────────────────────────────────────────────────────────

    def _draw_drawdown(self, ax, history: pd.DataFrame) -> None:
        running_max = history["total_value"].cummax()
        dd = (history["total_value"] - running_max) / running_max * 100

        ax.fill_between(dd.index, dd, 0, color=self.C["dd_fill"], alpha=0.8, zorder=1)
        ax.plot(dd.index, dd, color=self.C["dd_line"], linewidth=0.8, zorder=2)

        ax.set_title("Drawdown %", fontweight="bold", loc="left", fontsize=10)
        ax.set_ylabel("%", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(True, alpha=0.4)

    # ── Panel: P&L Distribution ────────────────────────────────────────────────

    def _draw_pnl_hist(self, ax, trades: List) -> None:
        ax.set_title("P&L Distribution", fontweight="bold", loc="left", fontsize=10)

        if not trades:
            ax.text(0.5, 0.5, "No Trades", ha="center", va="center",
                    transform=ax.transAxes, color=self.C["spine"])
            ax.axis("off")
            return

        pnls = np.array([t.pnl for t in trades])
        winners = pnls[pnls > 0]
        losers  = pnls[pnls <= 0]
        bins = np.linspace(pnls.min(), pnls.max(), 50)

        ax.hist(losers,  bins=bins, color=self.C["loser"],  alpha=0.85, label="Losers",  rwidth=0.9)
        ax.hist(winners, bins=bins, color=self.C["winner"], alpha=0.85, label="Winners", rwidth=0.9)
        ax.axvline(0, color=self.C["text"], linewidth=0.8, linestyle="--")
        ax.set_xlabel("P&L ($)", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Panel: Exit Breakdown Table ────────────────────────────────────────────

    def _draw_exit_table(self, ax, trades: List) -> None:
        """
        Colour-coded exit reason breakdown.

        Each exit type gets a distinct background so performance patterns
        (e.g. 90% of P&L from STOP_LOSS hits) are immediately visible.

        Args:
            ax: Matplotlib axes panel slot.
            trades: List of Trade dataclass objects.
        """
        ax.axis("off")
        ax.set_title("Exit Breakdown", fontweight="bold", loc="left", fontsize=10)

        if not trades:
            ax.text(0.5, 0.5, "No Trades", ha="center", va="center",
                    transform=ax.transAxes, color=self.C["spine"])
            return

        reasons = [t.exit_reason for t in trades]
        df = pd.DataFrame({"reason": reasons})
        total = len(df)
        counts = df["reason"].value_counts()

        rows = [[r, f"{c:,}", f"{c / total:.1%}"] for r, c in counts.items()]
        row_reasons = list(counts.index)
        rows.append(["TOTAL", f"{total:,}", "100.0%"])
        col_labels = ["Exit Reason", "Count", "%"]

        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        n_cols = len(col_labels)
        n_rows = len(rows)

        for c in range(n_cols):
            tbl.auto_set_column_width(c)

        # Header row
        for c in range(n_cols):
            cell = tbl[0, c]
            cell.set_facecolor(self.C["header"])
            cell.set_text_props(fontweight="bold", color=self.C["text"])
            cell.set_edgecolor(self.C["spine"])

        # Data rows
        for r in range(1, n_rows + 1):
            is_total = r == n_rows
            if is_total:
                bg = "#ECF0F1"
            else:
                reason = row_reasons[r - 1]
                bg = self._EXIT_COLORS.get(
                    reason,
                    "#FFFFFF" if r % 2 == 1 else self.C["row_alt"],
                )
            for c in range(n_cols):
                cell = tbl[r, c]
                cell.set_facecolor(bg)
                cell.set_edgecolor(self.C["spine"])
                if is_total:
                    cell.set_text_props(fontweight="bold")
