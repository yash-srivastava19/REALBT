from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple


class Strategy(ABC):
    """
    Base class for all trading strategies in REALBT.
    
    A strategy determines when and how to place orders based on market data.
    Users should subclass this and implement the required methods.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize the strategy.
        
        Args:
            symbols: List of asset symbols this strategy will trade
        """
        self.symbols = symbols or []
        self.position = {}  # Current position sizes
        self.context = {}   # Can store strategy-specific state
        
    def initialize(self) -> None:
        """
        Called once at the start of the backtest.
        Override this method to set up the strategy.
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Generate trading signals for the current market data.
        
        Args:
            data: Market data up to the current timestamp
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping symbols to target position sizes or weights
        """
        pass
    
    def handle_filled_order(self, order_details: Dict) -> None:
        """
        Called when an order is filled.
        Override to implement custom logic for handling filled orders.
        
        Args:
            order_details: Details of the filled order
        """
        symbol = order_details["symbol"]
        self.position[symbol] = order_details["new_position"]
    
    def teardown(self) -> None:
        """
        Called at the end of the backtest.
        Override to implement any cleanup or final calculations.
        """
        pass


class ExampleMACrossover(Strategy):
    """
    Example moving average crossover strategy.
    
    Buys when fast MA crosses above slow MA, sells when it crosses below.
    """
    
    def __init__(self, symbols: List[str], fast_window: int = 10, slow_window: int = 30):
        """
        Initialize the MA crossover strategy.
        
        Args:
            symbols: List of symbols to trade
            fast_window: Fast moving average window
            slow_window: Slow moving average window
        """
        super().__init__(symbols)
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def generate_signals(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Generate signals based on MA crossover.
        
        Returns:
            Dictionary mapping symbols to target positions:
              1.0 = fully long
              0.0 = flat
             -1.0 = fully short
        """
        signals = {}
        
        for symbol in self.symbols:
            # Get price data for this symbol
            symbol_data = data[data["symbol"] == symbol]
            
            if len(symbol_data) < self.slow_window:
                signals[symbol] = 0.0  # Not enough data yet
                continue
                
            # Calculate moving averages
            close_prices = symbol_data["close"].values
            fast_ma = np.mean(close_prices[-self.fast_window:])
            slow_ma = np.mean(close_prices[-self.slow_window:])
            
            # Previous day's values
            if len(close_prices) > self.slow_window:
                prev_fast_ma = np.mean(close_prices[-self.fast_window-1:-1])
                prev_slow_ma = np.mean(close_prices[-self.slow_window-1:-1])
                
                # Check for crossover
                if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
                    signals[symbol] = 1.0  # Buy signal
                elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
                    signals[symbol] = 0.0  # Sell signal
                else:
                    # No change
                    signals[symbol] = self.position.get(symbol, 0.0)
            else:
                signals[symbol] = 0.0
                
        return signals