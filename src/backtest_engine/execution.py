from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import random

@dataclass
class Order:
    """
    Represents a trade order.
    """
    symbol: str
    quantity: float
    side: str # 'BUY' or 'SELL'
    order_type: str = 'MARKET'
    reason: str = 'SIGNAL' # e.g., 'SIGNAL', 'SL', 'TP', 'TIME'
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Fill:
    """
    Represents a filled order (execution).
    """
    order: Order
    fill_price: float
    commission: float
    slippage: float
    cost: float
    timestamp: datetime
    
@dataclass
class Trade:
    """
    Represents a completed trade (Entry + Exit).
    """
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    direction: str # 'LONG' or 'SHORT'
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    exit_reason: str = 'SIGNAL'
    entry_signal_time: Optional[datetime] = None

class ExecutionHandler:
    """
    Handles order execution, simulation, and trade tracking.
    """
    
    def __init__(self, settings: Any):
        """
        Initialize the ExecutionHandler.
        
        Args:
            settings: Configuration object (BacktestSettings).
        """
        self.settings = settings
        self.fills: List[Fill] = []
        self.trades: List[Trade] = []
        
        # Position tracking for Trade matching (FIFO/LIFO simplified)
        # Dictionary of symbol -> list of Fills (Open positions)
        self._positions: Dict[str, List[Fill]] = {} 

    def execute_order(self, order: Order, data_bar: pd.Series, execute_at_close: bool = False) -> Optional[Fill]:
        """
        Simulates order execution with slippage and commission.
        
        Args:
            order: The Order object to execute.
            data_bar: The current OHLCV bar (Series). Uses 'open', 'high', 'low', 'close'.
            execute_at_close: If True, execute at the bar's Close. Otherwise at Open.
            
        Returns:
            Fill object if successful, None otherwise.
        """
        # 1. Determine Execution Price
        # Assumption: Market orders execute at Open of this bar (Next Open).
        # passed 'data_bar' is the *current* market state (Time T Open).
        
        price = data_bar['close'] if execute_at_close else data_bar['open']
        
        if order.order_type == 'MARKET':
            price = data_bar['close'] if execute_at_close else data_bar['open']
        
        # Simple Slippage Model (Random Amount per unit from 0 to max)
        spec = self.settings.get_instrument_spec(order.symbol)
        max_ticks = getattr(self.settings, 'max_slippage_ticks', 1)
        
        actual_slippage_ticks = random.randint(0, max_ticks)
        slippage = actual_slippage_ticks * spec["tick_size"]
        
        executed_price = price + slippage if order.side == 'BUY' else price - slippage
        
        # 2. Commission (Fixed Amount per unit)
        commission = abs(order.quantity) * self.settings.commission_rate
        
        cost = (executed_price * order.quantity) if order.side == 'BUY' else -(executed_price * order.quantity)
        
        fill = Fill(
            order=order,
            fill_price=executed_price,
            commission=commission,
            slippage=slippage,
            cost=cost, # Raw value
            timestamp=data_bar.name if isinstance(data_bar.name, datetime) else order.timestamp
        )
        self.fills.append(fill)
        self._process_trades(fill)
        return fill

    def _process_trades(self, fill: Fill):
        """
        Reconciles fills into Trades (Round-trips) for analytics.
        Uses FIFO (First-In-First-Out) matching for PnL calculation.
        """
        symbol = fill.order.symbol
        if symbol not in self._positions:
            self._positions[symbol] = []
            
        # Current fill details
        fill_qty = fill.order.quantity
        fill_price = fill.fill_price
        fill_comm = fill.commission
        fill_time = fill.timestamp
        remaining_qty = fill.order.quantity # Magnitude
        
        # Determine side (1 for Buy, -1 for Sell)
        side = 1 if fill.order.side == 'BUY' else -1
        
        # We iterate through open positions to match
        # Open positions list contains fills that are NOT yet fully closed
        # We need to modify the list in place, so index management is key.
        # Or we rebuild the list.
        
        new_open_positions = []
        
        for open_fill in self._positions[symbol]:
            if remaining_qty == 0:
                new_open_positions.append(open_fill)
                continue
                
            open_qty = open_fill.order.quantity # REMAINING magnitude
            open_side = 1 if open_fill.order.side == 'BUY' else -1
            
            # If same side, cannot match (it increases position)
            if side == open_side:
                new_open_positions.append(open_fill)
                continue
                
            # Opposite side -> Match!
            # Determine matchable quantity
            match_qty = min(abs(remaining_qty), abs(open_qty))
            
            # Update the open fill's quantity (reduce it)
            # We need to perform the reduction on the open_fill object or create a new partial one.
            # Since Fill is dataclass, let's just modify the quantity for internal tracking?
            # Or better, track 'remaining_qty' separately if we don't want to mutate history.
            # For simplicity, we assume 'execute_order' creates a NEW fill object, 
            # so we can mutate the one stored in '_positions' as it represents the 'Open Component'.
            
            # Pnl Calculation: (Exit Price - Entry Price) * Qty * Direction
            # Direction is defined by the ENTRY.
            # If Open was Long (Buy), then we are Selling. Pnl = (SellPrice - BuyPrice) * Qty
            # If Open was Short (Sell), then we are Buying. Pnl = (SellPrice - BuyPrice) * Qty :: (Entry - Exit) * Qty ?
            # Standard: (Exit - Entry) * Qty (where Qty is + for Long, but here we track matches)
            # Long: (Exit - Entry) * Qty
            # Short: (Entry - Exit) * Qty
            
            # Long: (Exit - Entry) * Qty * Multiplier
            # Short: (Entry - Exit) * Qty * Multiplier
            
            entry_price = open_fill.fill_price
            spec = self.settings.get_instrument_spec(symbol)
            multiplier = spec["multiplier"]
            
            if open_side == 1: # Long
                # Closing Long
                pnl = (fill_price - entry_price) * match_qty * multiplier
                direction = 'LONG'
            else: # Short
                # Closing Short
                pnl = (entry_price - fill_price) * match_qty * multiplier
                direction = 'SHORT'
            
            # Pro-rate commission for this trade chunk? 
            # Commission is paid on Entry and Exit.
            # Entry Comm: open_fill.commission * (match_qty / original_fill_qty)
            # Exit Comm: fill_comm * (match_qty / original_total_fill_qty)
            # This gets complex. Simplified: Attribute FULL exit commission to the trade? 
            # Or just accumulate PnL and subtract total commissions at portfolio level?
            # 'Trade' PnL usually is Net PnL.
            # Let's approximate: 
            # Comm = (EntryCommPerShare + ExitCommPerShare) * MatchQty
            entry_comm_per_share = open_fill.commission / abs(open_fill.order.quantity) if open_fill.order.quantity != 0 else 0
            exit_comm_per_share = fill_comm / abs(fill_qty) if fill_qty != 0 else 0
            
            trade_comm = (entry_comm_per_share + exit_comm_per_share) * match_qty
            net_pnl = pnl - trade_comm
            
            # Exit Reason comes from the CLOSING order (the current fill)
            exit_reason = fill.order.reason

            self.trades.append(Trade(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=fill_price,
                quantity=match_qty,
                direction=direction,
                entry_time=open_fill.timestamp,
                exit_time=fill_time,
                pnl=net_pnl,
                commission=trade_comm,
                exit_reason=exit_reason,
                entry_signal_time=open_fill.order.timestamp
            ))
            
            # Reduce quantities
            if abs(remaining_qty) >= abs(open_qty):
                # This open fill is fully closed
                remaining_qty = (abs(remaining_qty) - abs(open_qty)) * side # Keep sign
                # Open fill is consumed, do not append to new list
            else:
                # Open fill is partially closed
                # Update open fill qty
                residue = (abs(open_qty) - abs(remaining_qty)) * open_side
                open_fill.order.quantity = residue # Mutating for tracking
                new_open_positions.append(open_fill)
                remaining_qty = 0
        
        # If we still have remaining quantity, it's a new open position
        if remaining_qty != 0:
            # Create a clone/copy to store as open position with accurate remaining qty
            # We can't deepcopy easily without imports, but Fill is simple.
            # We re-use the fill object but set quantity to remaining.
            # CAUTION: 'fill' arg is the historical record. We should store a copy.
            from dataclasses import replace
            new_fill_tracker = replace(fill) # Create copy
            new_fill_tracker.order = replace(fill.order) # Deep copy order too just in case
            new_fill_tracker.order.quantity = remaining_qty
            new_open_positions.append(new_fill_tracker)
            
        self._positions[symbol] = new_open_positions

