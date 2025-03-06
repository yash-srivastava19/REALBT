import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Callable
from datetime import datetime
import logging
from tqdm import tqdm

from realbt.strategies.base import Strategy
from realbt.costs.impact import MarketImpactModel
from realbt.costs.slippage import SlippageModel
from realbt.costs.transaction import TransactionCostModel


class BacktestEngine:
    """
    Core backtesting engine that simulates strategy execution with realistic market friction.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 1_000_000.0,
        impact_model: Optional[MarketImpactModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        max_position_pct: float = 0.1,  # Maximum position size as percentage of portfolio
        max_leverage: float = 1.0,      # Maximum leverage (1.0 = no leverage)
        log_level: int = logging.INFO
    ):
        """
        Initialize the backtest engine.
        
        Args:
            data: Historical market data with columns:
                  [timestamp, symbol, open, high, low, close, volume]
            strategy: Trading strategy to backtest
            initial_capital: Starting capital
            impact_model: Model for market impact
            slippage_model: Model for execution slippage
            transaction_cost_model: Model for transaction costs
            max_position_pct: Maximum position size as percentage of portfolio
            max_leverage: Maximum leverage allowed
            log_level: Logging level
        """
        self.data = data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Market friction models
        # If not provided, we'll create default models
        self.impact_model = impact_model or self._default_impact_model()
        self.slippage_model = slippage_model or self._default_slippage_model()
        self.transaction_cost_model = transaction_cost_model or self._default_transaction_cost_model()
        
        # Position constraints
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        
        # Results tracking
        self.portfolio_history = []
        self.orders = []
        self.positions = {}
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("REALBT")
    
    def _default_impact_model(self) -> Callable:
        """Create a simple default market impact model"""
        # Simple square-root model: impact = factor * sqrt(order_size / ADV)
        def model(price, order_size, adv, side):
            impact_factor = 0.1
            normalized_size = abs(order_size) / adv if adv > 0 else 0
            impact = impact_factor * price * np.sqrt(normalized_size)
            return impact if side == "buy" else -impact
        return model
    
    def _default_slippage_model(self) -> Callable:
        """Create a simple default slippage model"""
        # Simple model: slippage = half spread + volatility component
        def model(price, order_size, adv, volatility):
            half_spread = price * 0.0005  # 1 basis point (0.01%) half-spread
            vol_component = price * volatility * 0.1 * min(1.0, abs(order_size) / adv)
            slippage = half_spread + vol_component
            return slippage
        return model
    
    def _default_transaction_cost_model(self) -> Callable:
        """Create a simple default transaction cost model"""
        # Fixed percentage + minimum fee
        def model(price, order_size):
            percentage_fee = 0.001  # 0.1%
            min_fee = 1.0  # $1 minimum
            cost = max(min_fee, abs(price * order_size * percentage_fee))
            return cost
        return model
        
    def run(self) -> pd.DataFrame:
        """
        Run the backtest.
        
        Returns:
            DataFrame with daily portfolio performance metrics
        """
        self.logger.info(f"Starting backtest with {len(self.data)} data points")
        
        # Initialize strategy
        self.strategy.initialize()
        
        # Get unique timestamps to iterate through
        timestamps = self.data["timestamp"].unique()
        
        # Main backtest loop
        for ts in tqdm(timestamps, desc="Backtesting"):
            # Get data up to current timestamp
            current_data = self.data[self.data["timestamp"] <= ts]
            latest_data = self.data[self.data["timestamp"] == ts]
            
            # Generate signals from strategy
            signals = self.strategy.generate_signals(current_data, ts)
            
            # Execute orders based on signals
            self._execute_orders(signals, latest_data, ts)
            
            # Update portfolio value
            self._update_portfolio_value(latest_data, ts)
        
        # Strategy teardown
        self.strategy.teardown()
        
        # Compile results
        results = pd.DataFrame(self.portfolio_history)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(results)
        
        self.logger.info(f"Backtest completed. Final portfolio value: ${results['portfolio_value'].iloc[-1]:,.2f}")
        
        return results
    
    def _execute_orders(self, signals: Dict[str, float], data: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        """
        Execute orders based on strategy signals.
        
        Args:
            signals: Strategy signals (target positions)
            data: Latest market data
            timestamp: Current timestamp
        """
        for symbol, target_position in signals.items():
            # Get current position
            current_position = self.positions.get(symbol, 0.0)
            
            # Skip if no change in position
            if target_position == current_position:
                continue
            
            # Get latest price data for this symbol
            symbol_data = data[data["symbol"] == symbol]
            if len(symbol_data) == 0:
                self.logger.warning(f"No data for {symbol} at {timestamp}, skipping order")
                continue
                
            price = symbol_data["close"].iloc[0]
            volume = symbol_data["volume"].iloc[0]
            
            # Calculate volatility (for slippage model)
            # Use 20-day standard deviation of returns as a simple volatility estimate
            symbol_history = self.data[
                (self.data["symbol"] == symbol) & 
                (self.data["timestamp"] <= timestamp)
            ].tail(21)
            
            volatility = 0.01  # Default if we don't have enough history
            if len(symbol_history) > 5:
                returns = symbol_history["close"].pct_change().dropna()
                volatility = returns.std()
            
            # Calculate order size in shares
            portfolio_value = self.portfolio_history[-1]["portfolio_value"] if self.portfolio_history else self.initial_capital
            position_value = price * current_position
            target_value = portfolio_value * target_position * self.max_position_pct
            
            # Limit by max leverage
            total_exposure = sum(abs(self.positions.get(s, 0) * data[data["symbol"] == s]["close"].iloc[0]) 
                                for s in self.positions if s in data["symbol"].values)
            total_exposure -= abs(position_value)  # Remove current position from calculation
            
            # Calculate maximum additional exposure allowed
            max_additional_exposure = portfolio_value * self.max_leverage - total_exposure
            target_value = min(target_value, max_additional_exposure)
            
            # Calculate order size in shares
            order_size = (target_value - position_value) / price
            
            # Skip tiny orders
            if abs(order_size) < 1:
                continue
                
            # Determine order direction
            side = "buy" if order_size > 0 else "sell"
            
            # Calculate average daily volume (ADV)
            adv_history = self.data[
                (self.data["symbol"] == symbol) & 
                (self.data["timestamp"] < timestamp)
            ].tail(20)
            
            adv = adv_history["volume"].mean() if len(adv_history) > 0 else volume
            
            # Apply market impact
            impact = self.impact_model(price, order_size, adv, side)
            
            # Apply slippage
            slippage = self.slippage_model(price, order_size, adv, volatility)
            
            # Calculate execution price
            execution_price = price + (impact + slippage if side == "buy" else -impact - slippage)
            
            # Calculate transaction costs
            transaction_cost = self.transaction_cost_model(execution_price, order_size)
            
            # Execute order
            new_position = current_position + order_size
            order_value = order_size * execution_price
            total_cost = abs(order_value) + transaction_cost
            
            # Update capital
            self.current_capital -= order_value + transaction_cost
            
            # Update positions
            self.positions[symbol] = new_position
            
            # Record order
            order = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "size": abs(order_size),
                "price": price,
                "execution_price": execution_price,
                "impact": impact,
                "slippage": slippage,
                "transaction_cost": transaction_cost,
                "total_cost": total_cost,
                "previous_position": current_position,
                "new_position": new_position
            }
            self.orders.append(order)
            
            # Notify strategy
            self.strategy.handle_filled_order(order)
    
    def _update_portfolio_value(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        """
        Update the portfolio value based on current positions and prices.
        
        Args:
            data: Latest market data
            timestamp: Current timestamp
        """
        # Calculate position values
        position_value = 0.0
        for symbol, shares in self.positions.items():
            # Skip if position is zero
            if shares == 0:
                continue
                
            # Get latest price
            symbol_data = data[data["symbol"] == symbol]
            if len(symbol_data) == 0:
                # Use last known price if not available for this timestamp
                symbol_history = self.data[
                    (self.data["symbol"] == symbol) & 
                    (self.data["timestamp"] < timestamp)
                ]
                if len(symbol_history) > 0:
                    price = symbol_history.iloc[-1]["close"]
                else:
                    self.logger.warning(f"No price data for {symbol}, assuming zero value")
                    price = 0.0
            else:
                price = symbol_data.iloc[0]["close"]
                
            # Update position value
            position_value += shares * price
        
        # Calculate total portfolio value
        portfolio_value = self.current_capital + position_value
        
        # Record portfolio state
        portfolio_state = {
            "timestamp": timestamp,
            "cash": self.current_capital,
            "position_value": position_value,
            "portfolio_value": portfolio_value,
            "returns": 0.0  # Will calculate after we have history
        }
        
        # Calculate return if we have history
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]["portfolio_value"]
            portfolio_state["returns"] = (portfolio_value / prev_value) - 1.0
        
        self.portfolio_history.append(portfolio_state)
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> None:
        """
        Calculate performance metrics for the backtest.
        
        Args:
            results: DataFrame with portfolio history
        """
        # Basic metrics
        results["cumulative_returns"] = (1 + results["returns"]).cumprod() - 1
        
        # Annualized return
        days = (results["timestamp"].max() - results["timestamp"].min()).days
        years = days / 365.25
        if years > 0:
            total_return = results["portfolio_value"].iloc[-1] / self.initial_capital - 1
            results.attrs["annualized_return"] = (1 + total_return) ** (1 / years) - 1
        else:
            results.attrs["annualized_return"] = 0.0
        
        # Volatility
        if len(results) > 1:
            daily_vol = results["returns"].std()
            results.attrs["annualized_volatility"] = daily_vol * np.sqrt(252)
            
            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            if daily_vol > 0:
                results.attrs["sharpe_ratio"] = results.attrs["annualized_return"] / results.attrs["annualized_volatility"]
            else:
                results.attrs["sharpe_ratio"] = 0.0
        else:
            results.attrs["annualized_volatility"] = 0.0
            results.attrs["sharpe_ratio"] = 0.0
        
        # Drawdown analysis
        results["high_watermark"] = results["portfolio_value"].cummax()
        results["drawdown"] = (results["portfolio_value"] / results["high_watermark"]) - 1
        results.attrs["max_drawdown"] = results["drawdown"].min()
        
        # Transaction analysis
        if self.orders:
            orders_df = pd.DataFrame(self.orders)
            results.attrs["total_trades"] = len(orders_df)
            results.attrs["total_volume"] = orders_df["size"].sum()
            results.attrs["total_transaction_costs"] = orders_df["transaction_cost"].sum()
            results.attrs["total_slippage"] = (orders_df["slippage"] * orders_df["size"]).sum()
            results.attrs["total_market_impact"] = (orders_df["impact"] * orders_df["size"]).sum()