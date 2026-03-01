from src.backtest_engine.engine import BacktestEngine
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.backtest_engine.settings import get_settings

if __name__ == "__main__":
    print("Running Risk Management Verification...")
    
    # Force settings
    settings = get_settings()
    settings.vol_lookback = 90
    settings.vol_percentile = 0.01 # Loose trigger to force trades
    settings.fixed_qty = 5 # Large size to force PnL swings
    settings.max_daily_loss = 50.0 # Very tight limit
    
    print(f"Max Daily Loss: {settings.max_daily_loss}")
    print(f"Slippage: {settings.slippage_rate}")
    print(f"Initial Capital: {settings.initial_capital}")
    
    engine = BacktestEngine()
    engine.run(VolatilityBreakoutStrategy)
    
    # Check if halted
    print(f"Trading Halted Permanently: {engine.trading_halted_permanently}")
    engine.show_results()
