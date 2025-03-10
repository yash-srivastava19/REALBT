# File: realbt/strategies/risk_metrics.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Container for strategy risk metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    monte_carlo_var_95: float = 0.0
    monte_carlo_expected_shortfall: float = 0.0
    

class RiskAnalyzer:
    """Analyzes strategy risk beyond simple Sharpe ratio."""
    
    def __init__(self, 
                risk_free_rate: float = 0.0, 
                trading_days_per_year: int = 252):
        """
        Initialize risk analyzer.
        
        Parameters:
        -----------
        risk_free_rate: float
            Annual risk-free rate (default: 0.0)
        trading_days_per_year: int
            Number of trading days per year (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate returns from equity curve.
        
        Parameters:
        -----------
        equity_curve: pd.Series
            Series of portfolio values over time
            
        Returns:
        --------
        pd.Series: Returns series
        """
        return equity_curve.pct_change().dropna()
    
    def calculate_drawdowns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdowns from equity curve.
        
        Parameters:
        -----------
        equity_curve: pd.Series
            Series of portfolio values over time
            
        Returns:
        --------
        pd.DataFrame: DataFrame with drawdown information
        """
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown in percentage terms
        drawdown = (equity_curve - running_max) / running_max
        
        # Create DataFrame
        drawdown_df = pd.DataFrame({
            'equity': equity_curve,
            'previous_peak': running_max,
            'drawdown': drawdown
        })
        
        # Calculate drawdown duration
        drawdown_df['drawdown_duration'] = 0
        in_drawdown = False
        current_duration = 0
        
        for i in range(len(drawdown_df)):
            if drawdown_df.iloc[i]['drawdown'] < 0:
                in_drawdown = True
                current_duration += 1
                drawdown_df.iloc[i, drawdown_df.columns.get_loc('drawdown_duration')] = current_duration
            else:
                in_drawdown = False
                current_duration = 0
                
        return drawdown_df
    
    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tail risk metrics including VaR and CVaR.
        
        Parameters:
        -----------
        returns: pd.Series
            Series of portfolio returns
            
        Returns:
        --------
        Dict[str, float]: Dictionary of tail risk metrics
        """
        # Historical Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 95% VaR is the 5th percentile of returns
        var_99 = np.percentile(returns, 1)  # 99% VaR is the 1st percentile of returns
        
        # Conditional Value at Risk (CVaR) / Expected Shortfall (ES)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Tail ratio: ratio of average positive return to average negative return
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        tail_ratio = abs(positive_returns.mean() / negative_returns.mean()) if len(negative_returns) > 0 else np.nan
        
        # Calculate omega ratio
        threshold = 0  # Can be changed to risk-free rate
        omega_ratio = len(returns[returns > threshold]) / len(returns[returns < threshold]) if len(returns[returns < threshold]) > 0 else np.nan
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'tail_ratio': tail_ratio,
            'omega_ratio': omega_ratio
        }
    
    def monte_carlo_simulation(self, 
                             returns: pd.Series, 
                             num_simulations: int = 1000, 
                             time_horizon: int = 252,
                             initial_value: float = 10000) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform Monte Carlo simulation to assess strategy robustness.
        
        Parameters:
        -----------
        returns: pd.Series
            Series of historical returns
        num_simulations: int
            Number of simulations to run
        time_horizon: int
            Number of periods to simulate
        initial_value: float
            Initial portfolio value
            
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, float]]:
            - Array of simulated paths
            - Dictionary of simulation metrics
        """
        # Estimate parameters for simulation
        mu = returns.mean()
        sigma = returns.std()
        
        # Create simulations using geometric Brownian motion
        dt = 1  # Time step (1 day)
        S0 = initial_value
        
        # Initialize simulation array
        simulation = np.zeros((time_horizon, num_simulations))
        
        # Run simulations
        for i in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mu, sigma, time_horizon)
            
            # Calculate cumulative path
            path = [S0]
            for r in random_returns:
                path.append(path[-1] * (1 + r))
                
            simulation[:, i] = path[1:]  # Store path excluding initial value
            
        # Calculate metrics from simulations
        final_values = simulation[-1, :]
        
        # Calculate 95% VaR from simulations
        monte_carlo_var_95 = np.percentile(final_values, 5)
        
        # Calculate expected shortfall
        threshold = np.percentile(final_values, 5)
        expected_shortfall = final_values[final_values <= threshold].mean()
        
        # Calculate probability of loss
        prob_loss = np.sum(final_values < initial_value) / num_simulations
        
        metrics = {
            'monte_carlo_var_95': monte_carlo_var_95,
            'monte_carlo_expected_shortfall': expected_shortfall,
            'probability_of_loss': prob_loss,
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values)
        }
        
        return simulation, metrics
    
    def compute_all_metrics(self, equity_curve: pd.Series) -> RiskMetrics:
        """
        Compute all risk metrics for a strategy.
        
        Parameters:
        -----------
        equity_curve: pd.Series
            Series of portfolio values over time
            
        Returns:
        --------
        RiskMetrics: Object containing all risk metrics
        """
        # Calculate returns
        returns = self.calculate_returns(equity_curve)
        
        # Calculate drawdowns
        drawdown_df = self.calculate_drawdowns(equity_curve)
        
        # Calculate annualized metrics
        annualized_return = (1 + returns.mean()) ** self.trading_days_per_year - 1
        annualized_vol = returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Sharpe ratio
        excess_returns = returns - self.daily_risk_free_rate
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.trading_days_per_year) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Max drawdown and duration
        max_drawdown = drawdown_df['drawdown'].min()
        max_drawdown_duration = drawdown_df['drawdown_duration'].max()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Tail risk metrics
        tail_metrics = self.calculate_tail_risk_metrics(returns)
        
        # Run Monte Carlo simulation
        _, simulation_metrics = self.monte_carlo_simulation(returns)
        
        # Create and return risk metrics object
        metrics = RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            var_95=tail_metrics['var_95'],
            var_99=tail_metrics['var_99'],
            cvar_95=tail_metrics['cvar_95'],
            cvar_99=tail_metrics['cvar_99'],
            calmar_ratio=calmar_ratio,
            omega_ratio=tail_metrics['omega_ratio'],
            tail_ratio=tail_metrics['tail_ratio'],
            monte_carlo_var_95=simulation_metrics['monte_carlo_var_95'],
            monte_carlo_expected_shortfall=simulation_metrics['monte_carlo_expected_shortfall']
        )
        
        return metrics
    
    def plot_drawdowns(self, equity_curve: pd.Series, top_n: int = 5, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the equity curve and highlight major drawdowns.
        
        Parameters:
        -----------
        equity_curve: pd.Series
            Series of portfolio values over time
        top_n: int
            Number of largest drawdowns to highlight
        figsize: Tuple[int, int]
            Figure size (width, height)
        """
        drawdown_df = self.calculate_drawdowns(equity_curve)
        
        # Find the start and end of drawdown periods
        drawdown_starts = []
        drawdown_ends = []
        drawdown_sizes = []
        
        in_drawdown = False
        start_idx = 0
        current_dd = 0
        
        for i in range(len(drawdown_df)):
            dd = drawdown_df.iloc[i]['drawdown']
            
            if not in_drawdown and dd < 0:
                # Start of a drawdown
                in_drawdown = True
                start_idx = i
                current_dd = dd
            elif in_drawdown:
                if dd < current_dd:
                    current_dd = dd
            elif in_drawdown:
                if dd < current_dd:
                    # Deeper drawdown
                    current_dd = dd
                elif dd == 0:
                    # End of drawdown
                    in_drawdown = False
                    drawdown_starts.append(start_idx)
                    drawdown_ends.append(i)
                    drawdown_sizes.append(current_dd)
        
        # Handle case where we're still in a drawdown at the end
        if in_drawdown:
            drawdown_starts.append(start_idx)
            drawdown_ends.append(len(drawdown_df) - 1)
            drawdown_sizes.append(current_dd)
        
        # Sort drawdowns by size
        sorted_indices = np.argsort(drawdown_sizes)
        top_drawdown_indices = sorted_indices[:min(top_n, len(sorted_indices))]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot equity curve
        plt.plot(equity_curve.index, equity_curve, label='Equity Curve', color='blue')
        
        # Highlight drawdown periods
        colors = plt.cm.Reds(np.linspace(0.5, 0.9, top_n))
        
        for i, idx in enumerate(top_drawdown_indices):
            start_date = equity_curve.index[drawdown_starts[idx]]
            end_date = equity_curve.index[drawdown_ends[idx]]
            dd_size = drawdown_sizes[idx]
            
            # Highlight the drawdown period
            plt.fill_between(
                equity_curve.index[drawdown_starts[idx]:drawdown_ends[idx]+1],
                0, 
                equity_curve.iloc[drawdown_starts[idx]:drawdown_ends[idx]+1],
                color=colors[i],
                alpha=0.3,
                label=f'Drawdown {i+1}: {dd_size:.2%} ({start_date.date()} - {end_date.date()})'
            )
        
        plt.title('Equity Curve with Major Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_return_distribution(self, equity_curve: pd.Series, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the distribution of returns with fitted normal distribution.
        
        Parameters:
        -----------
        equity_curve: pd.Series
            Series of portfolio values over time
        figsize: Tuple[int, int]
            Figure size (width, height)
        """
        returns = self.calculate_returns(equity_curve)
        
        plt.figure(figsize=figsize)
        
        # Plot histogram of returns
        n, bins, patches = plt.hist(returns, bins=50, density=True, alpha=0.7, label='Returns')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(returns)
        x = np.linspace(min(returns), max(returns), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
                 label=f'Normal Fit (μ={mu:.4f}, σ={sigma:.4f})')
        
        # Add VaR and CVaR lines
        tail_metrics = self.calculate_tail_risk_metrics(returns)
        plt.axvline(x=tail_metrics['var_95'], color='orange', linestyle='--', 
                    label=f'95% VaR: {tail_metrics["var_95"]:.4f}')
        plt.axvline(x=tail_metrics['var_99'], color='red', linestyle='--', 
                    label=f'99% VaR: {tail_metrics["var_99"]:.4f}')
        
        plt.title('Return Distribution')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_monte_carlo_simulation(self, 
                                  returns: pd.Series, 
                                  num_simulations: int = 1000, 
                                  time_horizon: int = 252,
                                  initial_value: float = 10000,
                                  figsize: Tuple[int, int] = (12, 8)):
        """
        Plot Monte Carlo simulation results.
        
        Parameters:
        -----------
        returns: pd.Series
            Series of historical returns
        num_simulations: int
            Number of simulations to run
        time_horizon: int
            Number of periods to simulate
        initial_value: float
            Initial portfolio value
        figsize: Tuple[int, int]
            Figure size (width, height)
        """
        simulation, metrics = self.monte_carlo_simulation(
            returns, num_simulations, time_horizon, initial_value
        )
        
        plt.figure(figsize=figsize)
        
        # Plot some sample paths
        num_paths_to_show = min(100, num_simulations)
        for i in range(num_paths_to_show):
            plt.plot(range(time_horizon), simulation[:, i], 'b-', alpha=0.1)
            
        # Plot percentiles
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        
        for i, p in enumerate(percentiles):
            percentile_path = np.percentile(simulation, p, axis=1)
            plt.plot(range(time_horizon), percentile_path, color=colors[i], lw=2, 
                     label=f'{p}th Percentile')
        
        plt.axhline(y=initial_value, color='black', linestyle='--', label='Initial Value')
        
        plt.title(f'Monte Carlo Simulation ({num_simulations} paths)')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join([
            f'Mean Final Value: ${metrics["mean_final_value"]:.2f}',
            f'Median Final Value: ${metrics["median_final_value"]:.2f}',
            f'95% VaR: ${initial_value - metrics["monte_carlo_var_95"]:.2f}',
            f'Expected Shortfall: ${initial_value - metrics["monte_carlo_expected_shortfall"]:.2f}',
            f'Probability of Loss: {metrics["probability_of_loss"]:.2%}'
        ])
        
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        return plt.gcf()