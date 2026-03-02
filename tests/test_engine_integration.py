import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.backtest_engine.engine import BacktestEngine
from src.backtest_engine.settings import get_settings
from src.strategies.sma_crossover import SmaCrossoverStrategy

@pytest.fixture
def mock_data():
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    df = pd.DataFrame({
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 100.0,
        'volume': 1000
    }, index=dates)
    return df

def test_engine_runs_strategy_successfully(mock_data):
    # Tests that the BacktestEngine can initialize and run
    # over a mocked slice of standard dataframe data
    # without crashing or referencing non-existent strategies.
    
    settings = get_settings().model_copy()
    settings.default_symbol = "YM"
    settings.initial_capital = 100000.0
    
    engine = BacktestEngine(data=mock_data, settings=settings)
    engine.run(SmaCrossoverStrategy)
    
    assert engine.portfolio is not None
    assert engine.portfolio.total_value > 0
    assert len(engine.portfolio.history) == len(mock_data)
