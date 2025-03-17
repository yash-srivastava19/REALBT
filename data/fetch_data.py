import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date, output_file):
    """
    Fetch stock data from Yahoo Finance and save it to a CSV file.
    
    :param ticker: Stock ticker symbol
    :param start_date: Start date for fetching data (YYYY-MM-DD)
    :param end_date: End date for fetching data (YYYY-MM-DD)
    :param output_file: Path to the output CSV file
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    # we need data in the format: timestamp,symbol,price,o, h, l, c, v
    # we can get this by using the following code
    data["symbol"] = ticker
    data = data[["Date", "symbol", "Close", "Open", "High", "Low", "Close", "Volume"]]
    data.columns = ["timestamp", "symbol", "price", "open", "high", "low", "close", "volume"]
    data.to_csv(output_file, index=False)
    print(f"Data for {ticker} saved to {output_file}")