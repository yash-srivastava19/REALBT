import click
import pandas as pd
import yaml
import importlib
import os
import sys
import logging
from datetime import datetime
from src.engine import BacktestEngine
import plotly.graph_objects as go
import plotly.express as px
from data.fetch_data import fetch_stock_data

@click.group()
def cli():
    """REALBT - REAListic BackTesting framework with accurate market friction modeling."""
    pass


def _generate_report(results, orders, config, output_file):
    """
    Generate an interactive HTML report summarizing the backtest results.
    
    :param results: DataFrame containing the backtest results
    :param orders: List of orders executed during the backtest
    :param config: Configuration dictionary used for the backtest
    :param output_file: Path to the output HTML file
    """
    # Create interactive plot
    print(results.columns)
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.25, 0.1, 0.45, 0.2], subplot_titles=("Equity Curve", "Profit/Loss", "OHLC Chart", "Drawdown"))
    
    trade_dates = results["timestamp"]
    trade_profits = results["returns"]
    trade_colors = ["green" if x > 0 else "red" for x in trade_profits]
    
    # Equity Curve
    fig.add_trace(go.Scatter(x=trade_dates, y=results["portfolio_value"], mode="lines", name="Portfolio Value", line=dict(color="blue", width=2)),row=1, col=1)

    #Profit Loss Curves
    fig.add_trace(go.Scatter(x=trade_dates, y=trade_profits, mode="markers", marker = dict(size=10, color=trade_colors, opacity=0.6), name="Trades"), row=2, col=1)
    
    #OHLC Candlestick Chart
    fig.add_trace(go.Candlestick(x=trade_dates, 
            open=results["portfolio_value"] - results["position_value"] * 0.02, 
            high=results["portfolio_value"] + results["position_value"] * 0.03, 
            low=results["portfolio_value"] - results["position_value"] * 0.03, 
            close=results["portfolio_value"],
            increasing_line_color= 'green', decreasing_line_color= 'red', 
            name="OHLC"), row=3, col=1)
    
    #Moving Averages
    fig.add_trace(go.Scatter(
        x=trade_dates, y=results["portfolio_value"].rolling(window=10).mean(),
        mode="lines", name="SMA 10", line=dict(color="orange", width=1.5, dash="dot")
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=trade_dates, y=results["portfolio_value"].rolling(window=20).mean(),
        mode="lines", name="SMA 20", line=dict(color="blue", width=1, dash="dash")
    ), row=3, col=1)

    #Drawdown
    fig.add_trace(go.Scatter(x=trade_dates, y=results["drawdown"], mode="lines", name="Drawdown", line=dict(color="red", width=2), fill="tozeroy"), row=4, col=1)
    
    fig.update_layout(height=1000, width=1400, title="Backtest Results",
                  xaxis_rangeslider_visible=False, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    # fig = px.line(results, x='timestamp', y='portfolio_value', title='Portfolio Value Over Time')
    # fig.update_layout(
    #     xaxis_title='Time',
    #     yaxis_title='Portfolio Value',
    #     template='plotly_white',
    #     title_font_size=18,
    #     font = dict(size=14),
    # )

    # fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='green')
    # fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='green')

    # fig.update_traces(line=dict(width=2))
    # Save plot to HTML file
    plot_file = f"{output_file}_portfolio_value.html"
    fig.write_html(plot_file)
    src_path_html = plot_file.replace("results/", "") 

    # Create HTML report
    report_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backtest Report</title>
    </head>
    <body>
        <h1>Backtest Report</h1>
        <h2>Configuration</h2>
        <pre>{yaml.dump(config, default_flow_style=False)}</pre>
        <h2>Key Metrics</h2>
        <ul>
            <li>Initial Capital: ${config['initial_capital']:,.2f}</li>
            <li>Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}</li>
            <li>Total Return: {results['portfolio_value'].iloc[-1] / config['initial_capital'] - 1:.2%}</li>
            <li>Annualized Return: {results.attrs.get('annualized_return', 0):.2%}</li>
            <li>Annualized Volatility: {results.attrs.get('annualized_volatility', 0):.2%}</li>
            <li>Sharpe Ratio: {results.attrs.get('sharpe_ratio', 0):.2f}</li>
            <li>Maximum Drawdown: {results.attrs.get('max_drawdown', 0):.2%}</li>
            {f"<li>Total Trades: {results.attrs['total_trades']}</li>" if 'total_trades' in results.attrs else ''}
            {f"<li>Total Transaction Costs: ${results.attrs.get('total_transaction_costs', 0):,.2f}</li>" if 'total_transaction_costs' in results.attrs else ''}
            {f"<li>Total Slippage: ${results.attrs.get('total_slippage', 0):,.2f}</li>" if 'total_slippage' in results.attrs else ''}
            {f"<li>Total Market Impact: ${results.attrs.get('total_market_impact', 0):,.2f}</li>" if 'total_market_impact' in results.attrs else ''}
        </ul>
        <h2>Portfolio Value Over Time</h2>
        <iframe src="{src_path_html}" width="100%" height="600"></iframe>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(report_html)

@cli.command("new")
@click.argument("project_name")
@click.option("--directory", "-d", default=".", help="Target directory")
def create_new_project(project_name, directory):
    """
    Create a new REALBT project with sample files.
    
    PROJECT_NAME is the name of the project.
    """
    project_dir = os.path.join(directory, project_name)
    
    # Create project directories
    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "strategies"), exist_ok=True)
    
    # Create sample strategy file
    with open(os.path.join(project_dir, "strategies", "sample_strategy.py"), "w") as f:
        f.write("""from""")

###
@cli.command("fetch-data")
@click.argument("ticker")
@click.argument("start_date")
@click.argument("end_date")
@click.argument("output_file", type=click.Path())
def fetch_data(ticker, start_date, end_date, output_file):
    """
    Fetch stock data from Yahoo Finance.
    
    TICKER is the stock ticker symbol.
    START_DATE is the start date for fetching data (YYYY-MM-DD).
    END_DATE is the end date for fetching data (YYYY-MM-DD).
    OUTPUT_FILE is the path to the output CSV file.
    """
    fetch_stock_data(ticker, start_date, end_date, output_file)

@cli.command("run")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="results", help="Output directory for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def run_backtest(config_file, output, verbose):
    """
    Run a backtest using a configuration file.
    
    CONFIG_FILE should be a YAML file with the backtest configuration.
    """
    # Configure logging
    print("Running backtest")
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("REALBT-CLI")
    
    logger.info(f"Loading configuration from {config_file}")
    
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    required_keys = ["data_file", "strategy", "initial_capital"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from {config['data_file']}")
    try:
        data = pd.read_csv(config["data_file"], parse_dates=["timestamp"])
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Load strategy
    logger.info(f"Loading strategy {config['strategy']['class']}")
    try:
        # Parse strategy class path
        module_path, class_name = config["strategy"]["class"].rsplit(".", 1)
        
        # Import module
        module = importlib.import_module(module_path)
        
        # Get strategy class
        strategy_class = getattr(module, class_name)
        
        # Create strategy instance
        strategy_args = config["strategy"].get("args", {})
        strategy = strategy_class(**strategy_args)
    except Exception as e:
        logger.error(f"Failed to load strategy: {e}")
        sys.exit(1)
    
    # Create backtest engine
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=config["initial_capital"],
        max_position_pct=config.get("max_position_pct", 0.1),
        max_leverage=config.get("max_leverage", 1.0),
        log_level=log_level
    )
    
    # Run backtest
    logger.info("Running backtest")
    results = engine.run()
    
    # Create output directory
    os.makedirs(output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    results.to_csv(f"{output}/results_{timestamp}.csv", index=False)
    
    # Save orders
    if engine.orders:
        orders_df = pd.DataFrame(engine.orders)
        orders_df.to_csv(f"{output}/orders_{timestamp}.csv", index=False)
    
    # Save performance metrics
    metrics = {k: v for k, v in results.attrs.items()}
    with open(f"{output}/metrics_{timestamp}.yaml", "w") as f:
        yaml.dump(metrics, f)
    
    # Generate performance report
    _generate_report(results, engine.orders, config, f"{output}/report_{timestamp}.html")
    
    # Display key metrics
    click.echo("\nBacktest Results:")
    click.echo(f"Initial Capital: ${config['initial_capital']:,.2f}")
    click.echo(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    click.echo(f"Total Return: {results['portfolio_value'].iloc[-1] / config['initial_capital'] - 1:.2%}")
    click.echo(f"Annualized Return: {results.attrs.get('annualized_return', 0):.2%}")
    click.echo(f"Annualized Volatility: {results.attrs.get('annualized_volatility', 0):.2%}")
    click.echo(f"Sharpe Ratio: {results.attrs.get('sharpe_ratio', 0):.2f}")
    click.echo(f"Maximum Drawdown: {results.attrs.get('max_drawdown', 0):.2%}")
    
    if "total_trades" in results.attrs:
        click.echo(f"Total Trades: {results.attrs['total_trades']}")
        click.echo(f"Total Transaction Costs: ${results.attrs.get('total_transaction_costs', 0):,.2f}")
        click.echo(f"Total Slippage: ${results.attrs.get('total_slippage', 0):,.2f}")
        click.echo(f"Total Market Impact: ${results.attrs.get('total_market_impact', 0):,.2f}")
    
    click.echo(f"\nResults saved to {output}/ directory")

###

if __name__ == "__main__":
    cli()