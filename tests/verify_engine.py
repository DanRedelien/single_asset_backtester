from src.backtest_engine.engine import BacktestEngine
from src.backtest_engine.execution import Order
import random

class TaggedStrategy:
    """
    Verification strategy that randomly opens Long AND Short positions
    and exits them via SL / TP / TIME tags.
    """
    def __init__(self, engine):
        self.engine = engine
        self.position = None   # None | 'LONG' | 'SHORT'
        self.hold_time = 0

    def on_bar(self, bar):
        symbol   = self.engine.settings.default_symbol
        price    = bar['close']
        cash     = self.engine.portfolio.current_cash
        pos_qty  = self.engine.portfolio.positions.get(symbol, 0)

        # ── No position: randomly open Long or Short ──────────────────────────
        if self.position is None:
            if random.random() > 0.6:
                direction = random.choice(['LONG', 'SHORT'])
                qty = (cash * 0.1) / price

                if direction == 'LONG':
                    order = Order(symbol=symbol, quantity=qty,
                                  side='BUY', order_type='MARKET',
                                  reason='SIGNAL', timestamp=bar.name)
                else:
                    # Short: sell qty we don't own (engine must support negative pos)
                    order = Order(symbol=symbol, quantity=qty,
                                  side='SELL', order_type='MARKET',
                                  reason='SIGNAL', timestamp=bar.name)

                self.position  = direction
                self.hold_time = 0
                return [order]

        # ── In position: check exit conditions ────────────────────────────────
        else:
            self.hold_time += 1
            p = random.random()
            reason = None
            if self.hold_time > 10:
                reason = 'TIME'
            elif p > 0.8:
                reason = 'TP'
            elif p < 0.2:
                reason = 'SL'

            if reason:
                if self.position == 'LONG' and pos_qty > 0:
                    order = Order(symbol=symbol, quantity=abs(pos_qty),
                                  side='SELL', order_type='MARKET',
                                  reason=reason, timestamp=bar.name)
                elif self.position == 'SHORT' and pos_qty < 0:
                    order = Order(symbol=symbol, quantity=abs(pos_qty),
                                  side='BUY', order_type='MARKET',
                                  reason=reason, timestamp=bar.name)
                else:
                    # Position already flat somehow
                    self.position = None
                    return []

                self.position = None
                return [order]

        return []


if __name__ == '__main__':
    print('Running Dashboard Verification...')
    engine = BacktestEngine()
    engine.run(TaggedStrategy)
    engine.show_results()
