# File: realbt/costs/position_management.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta


class BorrowingCostModel:
    """Models borrowing costs for short positions."""
    
    def __init__(self, base_fee_rate: float = 0.0025, hard_to_borrow_multiplier: Dict[str, float] = None):
        """
        Initialize borrowing cost model.
        
        Parameters:
        -----------
        base_fee_rate: float
            Annual base fee rate for borrowing shares (e.g., 0.0025 = 0.25%)
        hard_to_borrow_multiplier: Dict[str, float]
            Multiplier for hard-to-borrow securities by symbol
        """
        self.base_fee_rate = base_fee_rate
        self.hard_to_borrow_multiplier = hard_to_borrow_multiplier or {}
        
    def get_daily_fee_rate(self, symbol: str) -> float:
        """
        Get daily fee rate for a symbol.
        
        Parameters:
        -----------
        symbol: str
            Security symbol
            
        Returns:
        --------
        float: Daily fee rate
        """
        annual_rate = self.base_fee_rate * self.hard_to_borrow_multiplier.get(symbol, 1.0)
        daily_rate = annual_rate / 252  # Assuming 252 trading days per year
        return daily_rate
    
    def calculate_borrowing_cost(self, 
                               symbol: str, 
                               position_value: float, 
                               days_held: int) -> float:
        """
        Calculate borrowing cost for a short position.
        
        Parameters:
        -----------
        symbol: str
            Security symbol
        position_value: float
            Absolute dollar value of the short position
        days_held: int
            Number of days the position is held
            
        Returns:
        --------
        float: Borrowing cost in dollars
        """
        if position_value <= 0:  # Not a short position
            return 0.0
            
        daily_rate = self.get_daily_fee_rate(symbol)
        cost = position_value * daily_rate * days_held
        return cost


class DividendHandler:
    """Handles dividends for long and short positions."""
    
    def __init__(self, dividend_data: Optional[pd.DataFrame] = None):
        """
        Initialize dividend handler.
        
        Parameters:
        -----------
        dividend_data: Optional[pd.DataFrame]
            DataFrame containing dividend information
            Expected columns: ['symbol', 'ex_date', 'payment_date', 'amount']
        """
        self.dividend_data = dividend_data or pd.DataFrame(
            columns=['symbol', 'ex_date', 'payment_date', 'amount'])
        
    def load_dividend_data(self, dividend_data: pd.DataFrame):
        """
        Load dividend data.
        
        Parameters:
        -----------
        dividend_data: pd.DataFrame
            DataFrame containing dividend information
        """
        self.dividend_data = dividend_data
        
    def get_dividends_for_period(self, 
                               symbol: str, 
                               start_date: Union[str, datetime], 
                               end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get dividends for a specific period.
        
        Parameters:
        -----------
        symbol: str
            Security symbol
        start_date: Union[str, datetime]
            Start date of the period
        end_date: Union[str, datetime]
            End date of the period
            
        Returns:
        --------
        pd.DataFrame: Dividends in the specified period
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Filter dividends for the symbol and period
        mask = (
            (self.dividend_data['symbol'] == symbol) &
            (self.dividend_data['ex_date'] >= start_date) &
            (self.dividend_data['ex_date'] <= end_date)
        )
        
        return self.dividend_data[mask].copy()
    
    def calculate_dividend_impact(self, 
                                positions: Dict[str, int], 
                                start_date: Union[str, datetime], 
                                end_date: Union[str, datetime],
                                prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate dividend impact on positions.
        
        Parameters:
        -----------
        positions: Dict[str, int]
            Dictionary of positions {symbol: quantity}
            Negative quantities represent short positions
        start_date: Union[str, datetime]
            Start date of the period
        end_date: Union[str, datetime]
            End date of the period
        prices: Dict[str, float]
            Dictionary of prices {symbol: price} for calculating short dividend payments
            
        Returns:
        --------
        Dict[str, float]: Dictionary of dividend amounts by symbol
        """
        dividend_impact = {}
        
        for symbol, quantity in positions.items():
            dividends = self.get_dividends_for_period(symbol, start_date, end_date)
            
            if dividends.empty:
                continue
                
            total_dividend = 0.0
            
            for _, div_row in dividends.iterrows():
                div_amount = div_row['amount']
                
                if quantity > 0:  # Long position
                    total_dividend += quantity * div_amount
                else:  # Short position - pay the dividend
                    total_dividend -= abs(quantity) * div_amount
                    
            if total_dividend != 0:
                dividend_impact[symbol] = total_dividend
                
        return dividend_impact


class CorporateActionProcessor:
    """Processes corporate actions like splits, mergers, and spinoffs."""
    
    def __init__(self, corporate_actions: Optional[pd.DataFrame] = None):
        """
        Initialize corporate action processor.
        
        Parameters:
        -----------
        corporate_actions: Optional[pd.DataFrame]
            DataFrame containing corporate action information
            Expected columns: ['symbol', 'date', 'action_type', 'value', 'new_symbol']
            where action_type can be 'split', 'merger', 'spinoff', etc.
        """
        self.corporate_actions = corporate_actions or pd.DataFrame(
            columns=['symbol', 'date', 'action_type', 'value', 'new_symbol'])
        
    def load_corporate_actions(self, corporate_actions: pd.DataFrame):
        """
        Load corporate action data.
        
        Parameters:
        -----------
        corporate_actions: pd.DataFrame
            DataFrame containing corporate action information
        """
        self.corporate_actions = corporate_actions
        
    def get_actions_for_period(self, 
                             start_date: Union[str, datetime], 
                             end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get corporate actions for a specific period.
        
        Parameters:
        -----------
        start_date: Union[str, datetime]
            Start date of the period
        end_date: Union[str, datetime]
            End date of the period
            
        Returns:
        --------
        pd.DataFrame: Corporate actions in the specified period
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Filter actions for the period
        mask = (
            (self.corporate_actions['date'] >= start_date) &
            (self.corporate_actions['date'] <= end_date)
        )
        
        return self.corporate_actions[mask].copy()
    
    def process_actions(self, 
                      positions: Dict[str, int], 
                      start_date: Union[str, datetime], 
                      end_date: Union[str, datetime]) -> Tuple[Dict[str, int], List[Dict]]:
        """
        Process corporate actions for the specified period and update positions.
        
        Parameters:
        -----------
        positions: Dict[str, int]
            Dictionary of positions {symbol: quantity}
        start_date: Union[str, datetime]
            Start date of the period
        end_date: Union[str, datetime]
            End date of the period
            
        Returns:
        --------
        Tuple[Dict[str, int], List[Dict]]: 
            - Updated positions
            - List of processed actions with details
        """
        actions = self.get_actions_for_period(start_date, end_date)
        updated_positions = positions.copy()
        processed_actions = []
        
        if actions.empty:
            return updated_positions, processed_actions
            
        # Sort actions by date
        actions = actions.sort_values('date')
        
        for _, action in actions.iterrows():
            symbol = action['symbol']
            
            # Skip if we don't have a position in this symbol
            if symbol not in updated_positions:
                continue
                
            action_type = action['action_type']
            quantity = updated_positions[symbol]
            
            if action_type == 'split':
                # Handle stock split
                split_ratio = action['value']
                new_quantity = int(quantity * split_ratio)
                updated_positions[symbol] = new_quantity
                
                processed_actions.append({
                    'date': action['date'],
                    'symbol': symbol,
                    'action': 'split',
                    'old_quantity': quantity,
                    'new_quantity': new_quantity,
                    'ratio': split_ratio
                })
                
            elif action_type == 'merger':
                # Handle merger: convert to new symbol
                new_symbol = action['new_symbol']
                conversion_ratio = action['value']
                new_quantity = int(quantity * conversion_ratio)
                
                # Remove old symbol
                del updated_positions[symbol]
                
                # Add or update new symbol
                if new_symbol in updated_positions:
                    updated_positions[new_symbol] += new_quantity
                else:
                    updated_positions[new_symbol] = new_quantity
                    
                processed_actions.append({
                    'date': action['date'],
                    'symbol': symbol,
                    'action': 'merger',
                    'new_symbol': new_symbol,
                    'old_quantity': quantity,
                    'new_quantity': new_quantity,
                    'ratio': conversion_ratio
                })
                
            elif action_type == 'spinoff':
                # Handle spinoff: maintain original position and add new position
                new_symbol = action['new_symbol']
                spinoff_ratio = action['value']
                spinoff_quantity = int(quantity * spinoff_ratio)
                
                # Add or update spinoff symbol
                if new_symbol in updated_positions:
                    updated_positions[new_symbol] += spinoff_quantity
                else:
                    updated_positions[new_symbol] = spinoff_quantity
                    
                processed_actions.append({
                    'date': action['date'],
                    'symbol': symbol,
                    'action': 'spinoff',
                    'new_symbol': new_symbol,
                    'original_quantity': quantity,
                    'spinoff_quantity': spinoff_quantity,
                    'ratio': spinoff_ratio
                })
                
        return updated_positions, processed_actions