"""
Transaction Cost Analysis for REAListic BackTesting.

This module provides models for various transaction costs including
commissions, fees, taxes, slippage, and bid-ask spreads.
"""
import numpy as np
from typing import Dict, Optional, Union, Callable, List


class TransactionCostModel:
    """Base class for all transaction cost models."""
    
    def __init__(self):
        pass
    
    def calculate_costs(self, order_volume: float, price: float, market_data: Dict) -> float:
        """
        Calculate transaction costs for an order.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            price: Execution price
            market_data: Dictionary containing additional market data
            
        Returns:
            Transaction costs in absolute terms
        """
        raise NotImplementedError("Subclasses must implement calculate_costs")


class CommissionModel(TransactionCostModel):
    """Model for broker commissions and exchange fees."""
    
    def __init__(self, 
                 rate: float = 0.001,  # 10 bps commission rate
                 min_commission: float = 1.0,  # Minimum commission per trade
                 exchange_fee_rate: float = 0.0002,  # 2 bps exchange fee
                 clearing_fee_rate: float = 0.0001,  # 1 bps clearing fee
                 tax_rate: Optional[Dict[str, float]] = None):  # Tax rates by country
        """
        Initialize commission model.
        
        Args:
            rate: Commission rate as a percentage of trade value
            min_commission: Minimum commission per trade
            exchange_fee_rate: Exchange fee rate as a percentage of trade value
            clearing_fee_rate: Clearing fee rate as a percentage of trade value
            tax_rate: Dictionary mapping country codes to tax rates
        """
        super().__init__()
        self.rate = rate
        self.min_commission = min_commission
        self.exchange_fee_rate = exchange_fee_rate
        self.clearing_fee_rate = clearing_fee_rate
        
        # Default tax rates if none provided
        if tax_rate is None:
            self.tax_rate = {
                'US': 0.0,  # No transaction tax in US
                'UK': 0.005,  # 50 bps stamp duty in UK (buy only)
                'FR': 0.003,  # 30 bps FTT in France
                'IT': 0.002,  # 20 bps FTT in Italy
                'HK': 0.001,  # 10 bps stamp duty in Hong Kong
                # Add more countries as needed
            }
        else:
            self.tax_rate = tax_rate
    
    def calculate_costs(self, order_volume: float, price: float, market_data: Dict) -> Dict[str, float]:
        """
        Calculate commission, fees, and taxes for a trade.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            price: Execution price
            market_data: Dictionary containing at least 'country_code' for tax calculation
            
        Returns:
            Dictionary with breakdown of different cost components
        """
        # Trade value (absolute)
        trade_value = abs(order_volume * price)
        
        # Calculate commission with minimum
        commission = max(trade_value * self.rate, self.min_commission)
        
        # Calculate exchange and clearing fees
        exchange_fee = trade_value * self.exchange_fee_rate
        clearing_fee = trade_value * self.clearing_fee_rate
        
        # Calculate tax if country code is provided and it's a taxable transaction
        tax = 0.0
        country_code = market_data.get('country_code', 'US')
        
        if country_code in self.tax_rate:
            tax_rate = self.tax_rate[country_code]
            
            # Some taxes only apply to buys (e.g., UK stamp duty)
            is_buy = order_volume > 0
            apply_tax = True
            
            # Special rules by country
            if country_code == 'UK' and not is_buy:
                # UK stamp duty only applies to buys
                apply_tax = False
            
            if apply_tax:
                tax = trade_value * tax_rate
        
        # Total costs
        total = commission + exchange_fee + clearing_fee + tax
        
        return {
            'commission': commission,
            'exchange_fee': exchange_fee,
            'clearing_fee': clearing_fee,
            'tax': tax,
            'total': total
        }


class SlippageModel(TransactionCostModel):
    """Model for price slippage based on order size and market conditions."""
    
    def __init__(self, base_slippage: float = 0.0001, volatility_factor: float = 2.0):
        """
        Initialize slippage model.
        
        Args:
            base_slippage: Base slippage as a percentage of price
            volatility_factor: Factor to multiply volatility for slippage calculation
        """
        super().__init__()
        self.base_slippage = base_slippage
        self.volatility_factor = volatility_factor
    
    def calculate_costs(self, order_volume: float, price: float, market_data: Dict) -> Dict[str, float]:
        """
        Calculate slippage costs.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            price: Reference price (e.g., mid price)
            market_data: Dictionary with 'adv' (avg daily volume) and 'volatility' (price volatility)
            
        Returns:
            Dictionary with slippage cost and effective price
        """
        adv = market_data.get('adv', 1e6)  # Default to 1M if not provided
        volatility = market_data.get('volatility', 0.01)  # Default to 1% if not provided
        
        # Volume-based slippage component
        volume_ratio = abs(order_volume) / adv
        volume_slippage = np.sqrt(volume_ratio) * 0.01  # Square-root model
        
        # Volatility-based slippage component
        volatility_slippage = volatility * self.volatility_factor
        
        # Base slippage plus adjustments
        total_slippage_pct = self.base_slippage + volume_slippage + volatility_slippage
        
        # Direction adjustment (buys pay more, sells receive less)
        direction = 1 if order_volume > 0 else -1
        
        # Calculate slippage in price units
        slippage_cost = direction * price * total_slippage_pct
        
        # Calculate effective execution price after slippage
        effective_price = price + slippage_cost
        
        # Total slippage cost in absolute terms
        total_cost = abs(order_volume) * abs(slippage_cost)
        
        return {
            'slippage_percentage': total_slippage_pct * 100,  # Convert to percentage
            'slippage_per_unit': slippage_cost,
            'effective_price': effective_price,
            'total_cost': total_cost
        }


class BidAskSpreadModel(TransactionCostModel):
    """Model for bid-ask spread costs."""
    
    def __init__(self, default_spread_bps: float = 10.0):
        """
        Initialize bid-ask spread model.
        
        Args:
            default_spread_bps: Default spread in basis points if not provided in market data
        """
        super().__init__()
        self.default_spread_bps = default_spread_bps / 10000.0  # Convert bps to decimal
    
    def calculate_costs(self, order_volume: float, price: float, market_data: Dict) -> Dict[str, float]:
        """
        Calculate costs due to bid-ask spread.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            price: Mid price
            market_data: Dictionary with optional 'bid' and 'ask' prices
            
        Returns:
            Dictionary with spread cost and execution price
        """
        # Use provided bid/ask if available, otherwise calculate from spread
        if 'bid' in market_data and 'ask' in market_data:
            bid_price = market_data['bid']
            ask_price = market_data['ask']
            spread = ask_price - bid_price
            spread_pct = spread / price
        else:
            # Use default spread or provided spread in market_data
            spread_pct = market_data.get('spread_pct', self.default_spread_bps)
            half_spread = price * spread_pct / 2
            bid_price = price - half_spread
            ask_price = price + half_spread
            spread = ask_price - bid_price
        
        # Execute at bid for sell orders, ask for buy orders
        is_buy = order_volume > 0
        execution_price = ask_price if is_buy else bid_price
        
        # Cost is half the spread (assuming reference price is mid)
        cost_per_unit = half_spread = spread / 2
        total_cost = abs(order_volume) * cost_per_unit
        
        return {
            'bid': bid_price,
            'ask': ask_price,
            'spread': spread,
            'spread_pct': spread_pct * 100,  # Convert to percentage
            'execution_price': execution_price,
            'cost_per_unit': cost_per_unit,
            'total_cost': total_cost
        }


class CompositeCostModel(TransactionCostModel):
    """Combines multiple cost models into a unified model."""
    
    def __init__(self, models: List[TransactionCostModel]):
        """
        Initialize composite cost model.
        
        Args:
            models: List of transaction cost models to include
        """
        super().__init__()
        self.models = models
    
    def calculate_costs(self, order_volume: float, price: float, market_data: Dict) -> Dict[str, float]:
        """
        Calculate total transaction costs by combining all models.
        
        Args:
            order_volume: Size of the order (positive for buy, negative for sell)
            price: Reference price
            market_data: Dictionary with market data for all models
            
        Returns:
            Dictionary with combined costs and breakdown by model
        """
        results = {}
        total_cost = 0.0
        effective_price = price
        
        # Apply each model in sequence
        for i, model in enumerate(self.models):
            # Update price based on previous model's impact when relevant
            if i > 0 and hasattr(results[f'model_{i-1}'], 'effective_price'):
                effective_price = results[f'model_{i-1}']['effective_price']
            
            # Calculate costs for this model
            model_result = model.calculate_costs(order_volume, effective_price, market_data)
            results[f'model_{i}'] = model_result
            
            # Add to total cost
            if 'total_cost' in model_result:
                total_cost += model_result['total_cost']
            elif 'total' in model_result:
                total_cost += model_result['total']
        
        # Add total to results
        results['total_cost'] = total_cost
        
        # Calculate total impact on execution price
        total_price_impact = abs(effective_price - price) / price * 100  # as percentage
        results['total_price_impact_pct'] = total_price_impact
        
        # Calculate execution price
        results['execution_price'] = effective_price
        
        return results


# Factory function to create standard cost models
def create_standard_cost_model(**params) -> CompositeCostModel:
    """
    Create a standard comprehensive cost model with commission, slippage, and spread components.
    
    Args:
        **params: Optional parameters for each component model
        
    Returns:
        CompositeCostModel with all standard components
    """
    # Extract parameters for each model type
    commission_params = params.get('commission', {})
    slippage_params = params.get('slippage', {})
    spread_params = params.get('spread', {})
    
    # Create individual models
    commission_model = CommissionModel(**commission_params)
    slippage_model = SlippageModel(**slippage_params)
    spread_model = BidAskSpreadModel(**spread_params)
    
    # Combine into composite model
    return CompositeCostModel([commission_model, slippage_model, spread_model])