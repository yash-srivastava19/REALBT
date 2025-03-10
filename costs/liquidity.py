# File: realbt/costs/liquidity.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union


class LiquidityModel:
    """Models realistic liquidity constraints for backtesting."""
    
    def __init__(self, 
                 max_participation_rate: float = 0.1,
                 market_impact_factor: float = 0.1,
                 execution_delay: Dict[str, float] = None):
        """
        Initialize the liquidity model.
        
        Parameters:
        -----------
        max_participation_rate: float
            Maximum percentage of market volume that can be traded (0.0-1.0)
        market_impact_factor: float
            Factor to scale price impact based on participation rate
        execution_delay: Dict[str, float]
            Dictionary mapping order types to execution delays in seconds
            Default: {'market': 1.0, 'limit': 30.0}
        """
        self.max_participation_rate = max_participation_rate
        self.market_impact_factor = market_impact_factor
        
        if execution_delay is None:
            self.execution_delay = {'market': 1.0, 'limit': 30.0}
        else:
            self.execution_delay = execution_delay
    
    def calculate_max_shares(self, volume: float) -> int:
        """
        Calculate maximum number of shares that can be traded based on volume.
        
        Parameters:
        -----------
        volume: float
            Market volume for the period
            
        Returns:
        --------
        int: Maximum number of shares that can be traded
        """
        return int(volume * self.max_participation_rate)
    
    def model_market_depth(self, 
                          price: float, 
                          volume: float, 
                          order_size: int,
                          bid_ask_spread: float = 0.01,
                          order_book_depth: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
        """
        Model market depth to determine actual execution price.
        
        Parameters:
        -----------
        price: float
            Current mid price
        volume: float
            Current market volume
        order_size: int
            Size of the order in shares
        bid_ask_spread: float
            Current bid-ask spread
        order_book_depth: Optional[pd.DataFrame]
            Detailed order book information if available
            
        Returns:
        --------
        Tuple[float, float]: (execution_price, price_impact)
        """
        # Base implementation using square-root rule when order book not available
        participation_rate = min(abs(order_size) / volume, self.max_participation_rate)
        
        # Calculate price impact using square-root model
        # Kyle's lambda model: price impact ~ sqrt(order size / market depth)
        price_impact = price * self.market_impact_factor * np.sqrt(participation_rate)
        
        # Adjust for direction (buy or sell)
        direction = 1 if order_size > 0 else -1
        half_spread = bid_ask_spread / 2
        
        # Execution price includes bid-ask spread and market impact
        execution_price = price + (direction * half_spread) + (direction * price_impact)
        
        return execution_price, price_impact
    
    def calculate_execution_time(self, 
                               order_size: int, 
                               volume: float, 
                               order_type: str = 'market') -> float:
        """
        Calculate estimated execution time for an order.
        
        Parameters:
        -----------
        order_size: int
            Size of the order in shares
        volume: float
            Average volume per second
        order_type: str
            Type of order ('market', 'limit', etc.)
            
        Returns:
        --------
        float: Estimated execution time in seconds
        """
        # Base delay from order type
        base_delay = self.execution_delay.get(order_type, 1.0)
        
        # Calculate participation-adjusted delay
        if volume > 0:
            participation_rate = min(abs(order_size) / volume, self.max_participation_rate)
            # Larger orders take longer to execute
            size_factor = 1 + np.log1p(participation_rate * 10) 
        else:
            # If no volume, assume very slow execution
            size_factor = 5.0
            
        return base_delay * size_factor


class VWAP:
    """
    Volume-Weighted Average Price execution algorithm.
    Splits a large order into smaller chunks based on historical volume profile.
    """
    
    def __init__(self, 
                 target_participation_rate: float = 0.05,
                 trading_periods: int = 6):
        """
        Initialize VWAP execution model.
        
        Parameters:
        -----------
        target_participation_rate: float
            Target participation rate as a fraction of volume (0.0-1.0)
        trading_periods: int
            Number of periods to split the order into
        """
        self.target_participation_rate = target_participation_rate
        self.trading_periods = trading_periods
        
    def generate_schedule(self, 
                         total_shares: int, 
                         volume_profile: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Generate VWAP execution schedule.
        
        Parameters:
        -----------
        total_shares: int
            Total shares to execute
        volume_profile: Union[pd.Series, np.ndarray]
            Historical volume profile as series or array
            
        Returns:
        --------
        pd.Series: VWAP execution schedule with shares per period
        """
        # Normalize volume profile
        if isinstance(volume_profile, pd.Series):
            norm_profile = volume_profile / volume_profile.sum()
        else:
            norm_profile = volume_profile / np.sum(volume_profile)
            
        # Calculate shares per period based on volume profile
        shares_per_period = (norm_profile * total_shares).round().astype(int)
        
        # Adjust for rounding errors
        difference = total_shares - shares_per_period.sum()
        if difference != 0:
            # Add/subtract the difference from the highest volume period
            if isinstance(shares_per_period, pd.Series):
                idx = shares_per_period.idxmax()
                shares_per_period[idx] += difference
            else:
                idx = np.argmax(shares_per_period)
                shares_per_period[idx] += difference
                
        return shares_per_period if isinstance(shares_per_period, pd.Series) else pd.Series(shares_per_period)