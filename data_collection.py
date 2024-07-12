import yfinance as yf
import pandas as pd

def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    data = data['Adj Close']
    return data

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Example tickers
    start_date = "2018-01-01"
    end_date = "2023-01-01"
    
    stock_data = download_stock_data(tickers, start_date, end_date)
    stock_data.to_csv("stock_data.csv")
    print("Data downloaded and saved to stock_data.csv")
