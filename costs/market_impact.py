"""
Market Impact Models for REAListic BackTesting.

This module provides implementations of various market impact models to simulate
how trades affect market prices in a realistic manner.
"""
import numpy as np
from typing import Dict, Optional, Union, Callable


class MarketImpactModel:
    """Base class for all market impact models."""
    
    def __init__(self):
        pass
    
    def calculate_impact(self, order_volume: float, market_data: Dict) -> float:
        """
        Calculate the price impact of an order.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            market_data: Dictionary containing market data necessary for the calculation
            
        Returns:
            Price impact in percentage or absolute terms
        """
        raise NotImplementedError("Subclasses must implement calculate_impact")


class VolumeBasedImpactModel(MarketImpactModel):
    """
    Model market impact based on order volume relative to market volume.
    Uses the square-root law: price impact ~ sign(V) * k * sqrt(|V| / ADV)
    """
    
    def __init__(self, temporary_impact_factor: float = 0.1, permanent_impact_factor: float = 0.05):
        """
        Initialize volume-based impact model.
        
        Args:
            temporary_impact_factor: Factor for temporary price impact calculation
            permanent_impact_factor: Factor for permanent price impact calculation
        """
        super().__init__()
        self.temporary_impact_factor = temporary_impact_factor
        self.permanent_impact_factor = permanent_impact_factor
    
    def calculate_impact(self, order_volume: float, market_data: Dict) -> Dict[str, float]:
        """
        Calculate both temporary and permanent price impact based on volume.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            market_data: Dictionary containing market data with at least 'adv' (average daily volume)
                         and 'price' (current asset price)
        
        Returns:
            Dictionary with 'temporary' and 'permanent' impact in price units
        """
        if 'adv' not in market_data:
            raise ValueError("Market data must contain average daily volume ('adv')")
        
        adv = market_data['adv']
        price = market_data.get('price', 1.0)
        
        # Calculate relative volume
        relative_volume = abs(order_volume) / adv
        
        # Square root model for market impact
        sign = np.sign(order_volume)
        
        # Calculate temporary impact (reverts after trade)
        temp_impact = sign * self.temporary_impact_factor * price * np.sqrt(relative_volume)
        
        # Calculate permanent impact (stays in the market)
        perm_impact = sign * self.permanent_impact_factor * price * np.sqrt(relative_volume)
        
        return {
            'temporary': temp_impact,
            'permanent': perm_impact,
            'total': temp_impact + perm_impact
        }


class ElasticityBasedImpactModel(MarketImpactModel):
    """
    Model market impact based on price elasticity principles.
    Price impact is calculated based on a demand curve with specific elasticity.
    """
    
    def __init__(self, elasticity: float = 1.5):
        """
        Initialize elasticity-based impact model.
        
        Args:
            elasticity: Price elasticity parameter (higher means less impact)
        """
        super().__init__()
        self.elasticity = elasticity
    
    def calculate_impact(self, order_volume: float, market_data: Dict) -> Dict[str, float]:
        """
        Calculate price impact based on price elasticity model.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            market_data: Dictionary containing 'adv' and 'price'
                
        Returns:
            Dictionary with impact components
        """
        if 'adv' not in market_data or 'price' not in market_data:
            raise ValueError("Market data must contain 'adv' and 'price'")
        
        adv = market_data['adv']
        price = market_data['price']
        
        # Calculate relative volume
        relative_volume = abs(order_volume) / adv
        
        # Apply elasticity model: impact ~ volume^(1/elasticity)
        impact_factor = relative_volume ** (1 / self.elasticity)
        sign = np.sign(order_volume)
        
        # Impact is higher when elasticity is lower
        impact = sign * price * impact_factor
        
        # For elasticity model, we can separate transient vs permanent differently
        temporary_portion = 0.7  # 70% of impact is temporary
        
        return {
            'temporary': impact * temporary_portion,
            'permanent': impact * (1 - temporary_portion),
            'total': impact
        }


class OrderBookImpactModel(MarketImpactModel):
    """
    Model market impact using a simulated order book to account for market depth.
    This is a more sophisticated model that requires order book data or simulation.
    """
    
    def __init__(self, depth_function: Optional[Callable] = None):
        """
        Initialize order book impact model.
        
        Args:
            depth_function: A function that simulates order book depth at different price levels
                           If None, a default exponential depth function is used
        """
        super().__init__()
        
        # Default depth function if none provided
        if depth_function is None:
            # Default exponential depth function: depth(p) = base_depth * exp(-k * price_distance)
            self.depth_function = lambda price_level, reference_price, params: (
                params.get('base_depth', 1000) * 
                np.exp(-params.get('decay', 0.1) * abs(price_level - reference_price))
            )
        else:
            self.depth_function = depth_function
    
    def calculate_impact(self, order_volume: float, market_data: Dict) -> Dict[str, float]:
        """
        Calculate price impact by walking through a simulated order book.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            market_data: Dictionary containing 'price', optional 'book_params' for the depth function
                
        Returns:
            Dictionary with impact components
        """
        if 'price' not in market_data:
            raise ValueError("Market data must contain 'price'")
        
        price = market_data['price']
        book_params = market_data.get('book_params', {'base_depth': 1000, 'decay': 0.1})
        
        # Direction of price movement (buy orders increase price, sell orders decrease)
        direction = 1 if order_volume > 0 else -1
        
        # Simulate walking through the order book
        remaining_volume = abs(order_volume)
        current_price = price
        price_levels = []
        
        # Tick size for price levels
        tick_size = market_data.get('tick_size', 0.01)
        
        # Walk through the order book until the order is filled
        while remaining_volume > 0:
            # Get available volume at this price level
            level_volume = self.depth_function(current_price, price, book_params)
            
            # If we can fill the order at this level
            if level_volume >= remaining_volume:
                price_levels.append((current_price, remaining_volume))
                remaining_volume = 0
            else:
                # Take what's available and move to the next price level
                price_levels.append((current_price, level_volume))
                remaining_volume -= level_volume
                current_price += direction * tick_size
        
        # Calculate volume-weighted average execution price
        total_volume = sum(vol for _, vol in price_levels)
        vwap = sum(price * vol for price, vol in price_levels) / total_volume
        
        # Impact is the difference between VWAP and original price
        impact = vwap - price if direction > 0 else price - vwap
        
        # Determine temporary vs permanent components
        recovery_factor = 0.7  # 70% of impact recovers (is temporary)
        
        return {
            'temporary': impact * recovery_factor,
            'permanent': impact * (1 - recovery_factor),
            'total': impact,
            'execution_details': {
                'vwap': vwap,
                'price_levels': price_levels
            }
        }


# Factory function to create impact models
def create_impact_model(model_type: str, **params) -> MarketImpactModel:
    """
    Factory function to create market impact models.
    
    Args:
        model_type: Type of model ('volume', 'elasticity', 'orderbook')
        **params: Parameters for the specific model
        
    Returns:
        Initialized market impact model
    """
    models = {
        'volume': VolumeBasedImpactModel,
        'elasticity': ElasticityBasedImpactModel,
        'orderbook': OrderBookImpactModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(models.keys())}")
    
    return models[model_type](**params)