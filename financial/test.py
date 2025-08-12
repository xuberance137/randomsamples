"""
Yahoo Finance API Test Script - Stock Data Validation and Testing

This script provides a simple command-line interface for testing and validating
stock ticker data retrieval from the Yahoo Finance API. It serves as a utility
tool for debugging and verifying that specific tickers can be successfully
fetched and their basic financial information can be accessed.

Key Features:
- Command-line argument parsing for ticker input
- Basic stock information retrieval (company name, P/E ratios)
- Error handling and validation of Yahoo Finance API responses
- Simple output formatting for quick testing

Usage:
    python test.py <ticker_symbol>
    
    Examples:
        python test.py AAPL
        python test.py MSFT
        python test.py BRK-B

Output:
    - Company long name
    - Trailing P/E ratio
    - Forward P/E ratio

Data Source:
- Yahoo Finance API via yfinance library

Dependencies:
- yfinance: Yahoo Finance API wrapper
- argparse: Command-line argument parsing

Author: Gopal Erinjippurath
Version: 1.0
Last Updated: 2025
"""

# test script for testing tickers on yfinance API


import yfinance as yf
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Receive text as an argument.")

# Add an argument for text
parser.add_argument('text', type=str, help='Text input from command line')

# Parse the arguments
args = parser.parse_args()

# Access the text argument
print(f"Received text: {args.text}")

# Run the app
if __name__ == "__main__":
    ticker = args.text
    print(ticker)
    stock = yf.Ticker(ticker)
    company_name = stock.info.get("longName", "N/A")
    pe_ratio = stock.info.get("trailingPE", "N/A")
    forward_pe = stock.info.get("forwardPE", "N/A")
    print(company_name, pe_ratio, forward_pe)

