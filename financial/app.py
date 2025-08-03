import dash
from dash import dcc, html, Input, Output, dash_table
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from pytrends.request import TrendReq
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from keys import NEWSAPI_KEY
import time
import json
import os
from pytz import timezone

REFRESH_CYCLE = 60 # in minutes
FEAR_AND_GREED_SMOOTHING_WINDOW = 10 # number of days to get smoother sentiment
CACHE_FILE = "./data/fomo_index_cache.json"
PACIFIC_TIME = timezone('US/Pacific')
# Path to ChromeDriver (adjust based on your system setup)
CHROME_DRIVER_PATH = '/usr/local/bin/chromedriver'

def should_run_fomo_computation():
    """
    Check if we should run the FOMO index computation based on the current time.
    Returns True if it's 7 AM Pacific Time and we haven't run today, False otherwise.
    """
    # Get current time in Pacific timezone
    now = datetime.now(PACIFIC_TIME)
    
    # Check if cache file exists
    if not os.path.exists(CACHE_FILE):
        return True
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            # Parse the ISO format string directly into a timezone-aware datetime
            last_run_time = datetime.fromisoformat(cache['last_run_time'])
            
            # Check if we've run today
            if last_run_time.date() == now.date():
                return False
            
            # Check if it's 7 AM
            return now.hour == 7
    except Exception as e:
        print(f"Error reading cache: {e}")
        return True

def save_fomo_results(results):
    """
    Save FOMO index results to cache file.
    """
    try:
        cache_data = {
            'last_run_time': datetime.now(PACIFIC_TIME).isoformat(),
            'results': results
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

def load_fomo_results():
    """
    Load FOMO index results from cache file.
    Returns None if cache doesn't exist or is invalid.
    """
    try:
        if not os.path.exists(CACHE_FILE):
            return None
        
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            return cache['results']
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None

def fetch_put_call_ratio(start="2022-01-01", end="2023-01-01"):
    """
    Fetches CBOE Put/Call Ratio data directly from a web source.
    Replace this implementation with scraping or APIs as needed.
    """
    # For demonstration, returning a placeholder time series
    # Replace this with actual data fetching logic
    dates = pd.date_range(start=start, end=end, freq='D')
    put_call_ratios = np.random.uniform(0.8, 1.2, len(dates))  # Simulated values
    return pd.DataFrame({'Date': dates, 'Put/Call': put_call_ratios}).set_index('Date')


# Define the function
def fetch_put_call_ratio(start="2022-01-01", end="2023-01-01"):
    """
    Fetches CBOE Put/Call Ratio data dynamically using Selenium and returns it as a time series.

    Returns:
    - DataFrame with Date as index and Put/Call Ratio as a column.
    """
    # Configure Selenium WebDriver (ensure you have ChromeDriver installed)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    print("Loading Driver")
    service = Service(CHROME_DRIVER_PATH)  # Update with your ChromeDriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("Loaded Driver")
    
    try:
        # Open the CBOE page
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        driver.get(url)
        time.sleep(5)  # Allow time for JavaScript to render the page

        # Locate the table containing the Put/Call Ratio
        table_element = driver.find_element(By.XPATH, '//table[contains(@class, "bds-table")]')  # Adjusted XPath
        html_content = table_element.get_attribute('outerHTML')

        print("Loaded Page")
        # Parse the table using pandas
        table = pd.read_html(html_content)[0]
        table = table.rename(columns={"Date": "Date", "Equity Put/Call Ratio": "Put/Call"})
        table["Date"] = pd.to_datetime(table["Date"])
        table.set_index("Date", inplace=True)

        print("Loaded Table")
        # Return the Put/Call Ratio as a DataFrame
        put_call_data = table[["Put/Call"]].copy()
        return put_call_data[(put_call_data.index >= start) & (put_call_data.index <= end)]
    except Exception as e:
        print(f"Error fetching Put/Call Ratio: {e}")
        return pd.DataFrame()
    finally:
        driver.quit()

def compute_momentum_score(ticker):
    """
    Calculate the Momentum Score based on 7-day price change.
    
    Parameters:
    - ticker: str, the stock ticker symbol
    
    Returns a momentum score between 0-20.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="7d")
        
        # Check if data is empty
        if data.empty:
            print(f"No data available for {ticker}")
            momentum_score = 0
        else:
            # Get close prices
            close_prices = data['Close'].dropna()
            
            # Check if we have enough valid data points
            if len(close_prices) < 2:
                print(f"Not enough valid data points for {ticker}")
                momentum_score = 0
            else:
                # Get first and last valid prices
                start_price = float(close_prices.iloc[0])
                end_price = float(close_prices.iloc[-1])
                
                # Calculate percentage change
                change = (end_price - start_price) / start_price
                
                # Normalize to [-100, 100] range
                momentum_score = min(max(change*100, -100), 100) 
                print(f"Momentum score for {ticker}: {momentum_score:.2f}")
    except Exception as e:
        print(f"Error calculating momentum for {ticker}: {str(e)}")
        momentum_score = 10  # Neutral fallback

    return momentum_score

def compute_fomo_index(ticker, keyword):
    """
    Approximate the FOMO index based on:
    1. Asset momentum (7-day price change)
    2. News sentiment (headline polarity)
    3. Google Trends interest in trading behavior
    
    Parameters:
    - ticker: str, the stock ticker symbol
    - keyword: str, the keyword to use for sentiment and trends analysis
    
    Returns a composite score between 0-100.
    """
    # --- 1. Momentum ---
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="7d")
        
        # Check if data is empty
        if data.empty:
            print(f"No data available for {ticker}")
            momentum_score = 10
        else:
            # Get close prices
            close_prices = data['Close'].dropna()
            
            # Check if we have enough valid data points
            if len(close_prices) < 2:
                print(f"Not enough valid data points for {ticker}")
                momentum_score = 10
            else:
                # Get first and last valid prices
                start_price = float(close_prices.iloc[0])
                end_price = float(close_prices.iloc[-1])
                
                # Calculate percentage change
                change = (end_price - start_price) / start_price
                
                # Normalize to 0-20 range
                momentum_score = min(max((change * 100), -10), 10) + 10
                print(f"Momentum score for {ticker}: {momentum_score:.2f}")
    except Exception as e:
        print(f"Error calculating momentum for {ticker}: {str(e)}")
        momentum_score = 10  # Neutral fallback

    # --- 2. News Sentiment (last 5 headlines) ---
    try:
        api_key = NEWSAPI_KEY
        # Wrap keyword in quotation marks for exact phrase matching
        quoted_keyword = f'"{keyword}"'
        url = f"https://newsapi.org/v2/everything?q={quoted_keyword}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}"
        headlines = requests.get(url).json().get("articles", [])
        sentiment = [TextBlob(article['title']).sentiment.polarity for article in headlines]
        avg_sentiment = sum(sentiment) / len(sentiment) if sentiment else 0
        sentiment_score = min(max((avg_sentiment + 1) * 10, 0), 20)  # Normalize to 0-20
        print(f"Sentiment score for {ticker}: {sentiment_score:.2f}")
    except Exception as e:
        print(f"Error calculating sentiment for {ticker}: {str(e)}")
        sentiment_score = 10  # Neutral fallback

    # --- 3. Google Trends for keyword ---
    max_retries = 3
    base_delay = 5  # Base delay in seconds
    trends_score = 10  # Default value
    
    for attempt in range(max_retries):
        try:
            # Add exponential backoff delay
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"Retrying Google Trends request for {ticker} after {delay} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            
            # Initialize pytrends with timeout and proper parameters
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
            
            # Build payload
            pytrends.build_payload([keyword], timeframe='now 7-d')
            
            # Get interest over time
            interest = pytrends.interest_over_time()
            
            if interest.empty:
                print(f"No trends data available for {ticker}")
                break
                
            # Calculate trends score
            avg_interest = interest[keyword].mean()
            trends_score = avg_interest / 5  # Normalize approx to 0-20
            print(f"Trends score for {ticker}: {trends_score:.2f}")
            break  # Success, exit retry loop
            
        except Exception as e:
            if "429" in str(e):
                if attempt < max_retries - 1:
                    continue  # Try again with backoff
                else:
                    print(f"Max retries reached for {ticker}. Using default trends score.")
            else:
                print(f"Error processing trends for {ticker}: {str(e)}")
                break  # Other errors, don't retry

    # --- Composite FOMO Index (0-100 scale) ---
    fomo_index = round((momentum_score + sentiment_score + trends_score) * (100 / 60), 2)
    return {
        "ticker": ticker,
        "keyword": keyword,
        "momentum_score": momentum_score,
        "sentiment_score": sentiment_score,
        "trends_score": trends_score,
        "fomo_index": fomo_index
    }

def calculate_fear_and_greed(start_date="2022-01-01", end_date="2023-01-01"):
    """
    Calculates the Fear and Greed Index based on various metrics
    including stock price momentum, strength, breadth, put/call ratio,
    market volatility, safe haven demand, and junk bond demand.

    Parameters:
    - start_date: str, start date of analysis
    - end_date: str, end date of analysis

    Returns:
    - DataFrame with Date and Fear and Greed Index
    """
    data = pd.DataFrame()

    # 1. Stock Price Momentum (S&P 500 vs. 125-day moving average)
    sp500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)
    sp500.index = sp500.index.tz_localize(None)  # Remove timezone information
    sp500['Momentum'] = sp500['Close'] - sp500['Close'].rolling(125).mean()
    data['Momentum'] = sp500['Momentum']

    # 2. Stock Price Strength (Net new highs vs. lows)
    data['Strength'] = sp500['Close'].diff()

    # 3. Stock Price Breadth (Advancing vs. Declining stocks)
    data['Breadth'] = np.sign(sp500['Close'].pct_change())

    # 4. Put/Call Ratio (Options market sentiment)
    # put_call_data = fetch_put_call_ratio(start=start_date, end=end_date)
    # #put_call_data.index = put_call_data.index.tz_localize(None)  # Remove timezone
    # data = data.merge(put_call_data, left_index=True, right_index=True, how='left')

    # 5. Market Volatility (VIX index)
    vix = yf.Ticker("^VIX").history(start=start_date, end=end_date)
    vix.index = vix.index.tz_localize(None)  # Remove timezone
    data['Volatility'] = vix['Close']

    # 6. Safe Haven Demand (Bonds vs. Stocks)
    tlt = yf.Ticker("TLT").history(start=start_date, end=end_date)
    tlt.index = tlt.index.tz_localize(None)  # Remove timezone
    data['Safe Haven'] = tlt['Close'].pct_change() - sp500['Close'].pct_change()

    # 7. Junk Bond Demand (Spread between junk and investment-grade bonds)
    hyg = yf.Ticker("HYG").history(start=start_date, end=end_date)
    hyg.index = hyg.index.tz_localize(None)  # Remove timezone
    lqd = yf.Ticker("LQD").history(start=start_date, end=end_date)
    lqd.index = lqd.index.tz_localize(None)  # Remove timezone
    data['Junk Spread'] = hyg['Close'] - lqd['Close']

    # Normalize all metrics to scale 0-1
    for column in ['Momentum', 'Strength', 'Breadth', 'Volatility', 'Safe Haven', 'Junk Spread']:
        if column in data:
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    # Compute the Fear and Greed Index as an average of all components
    data['Fear and Greed Index'] = data[['Momentum', 'Strength', 'Breadth',
                                         'Volatility', 'Safe Haven', 'Junk Spread']].mean(axis=1)

    data = data.reset_index().rename(columns={'index': 'Date'})
    return data[['Date', 'Fear and Greed Index']]

def calculate_smoothed_fear_and_greed(fear_greed_data, window=5):
    """
    Calculates a smoothed version of the Fear and Greed Index using a moving average.

    Parameters:
    - fear_greed_data: DataFrame containing the Fear and Greed Index time series.
    - window: int, the window size for the moving average (default is 10).

    Returns:
    - DataFrame with an additional column for the smoothed Fear and Greed Index.
    """
    # Add a smoothed column using a rolling mean
    fear_greed_data['Smoothed Fear and Greed Index'] = (
        fear_greed_data['Fear and Greed Index']
        .rolling(window=window, min_periods=1)
        .mean()
    )
    return fear_greed_data


# Define a function to fetch data
def fetch_market_data():
    """
    Fetches market data for tickers in 'tickers-shortlist.csv' and returns a DataFrame.
    """
    # Read tickers and keywords from CSV
    try:
        tickers_df = pd.read_csv("./data/tickers.csv")
        tickers = tickers_df["Ticker"].tolist()
        keywords = tickers_df["Keyword"].tolist()
    except Exception as e:
        print(f"Error reading tickers file: {e}")
        return pd.DataFrame()

    # Check if we need to compute momentum scores
    if should_run_fomo_computation():
        print("Computing new momentum scores...")
        momentum_results = []
        for ticker, keyword in zip(tickers, keywords):
            max_retries = 3
            base_delay = 5  # Base delay in seconds
            for attempt in range(max_retries):
                try:
                    result = compute_momentum_score(ticker)
                    momentum_results.append({
                        "ticker": ticker,
                        "momentum_score": result
                    })
                    # Add delay between requests
                    time.sleep(2)  # 2 second delay between momentum calculations
                    break  # Success, exit retry loop
                except Exception as e:
                    if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limited for {ticker}. Retrying after {delay} seconds (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        print(f"Error computing momentum for {ticker}: {e}")
                        momentum_results.append({
                            "ticker": ticker,
                            "momentum_score": 10
                        })
                        time.sleep(2)  # Delay even on error
                        break
        save_fomo_results(momentum_results)
    else:
        print("Using cached momentum scores...")
        momentum_results = load_fomo_results()
        if momentum_results is None:
            print("No cached results available, computing new momentum scores...")
            momentum_results = []
            for ticker, keyword in zip(tickers, keywords):
                max_retries = 3
                base_delay = 5  # Base delay in seconds
                for attempt in range(max_retries):
                    try:
                        result = compute_momentum_score(ticker)
                        momentum_results.append({
                            "ticker": ticker,
                            "momentum_score": result
                        })
                        time.sleep(2)  # 2 second delay between momentum calculations
                        break  # Success, exit retry loop
                    except Exception as e:
                        if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            print(f"Rate limited for {ticker}. Retrying after {delay} seconds (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            print(f"Error computing momentum for {ticker}: {e}")
                            momentum_results.append({
                                "ticker": ticker,
                                "momentum_score": 10
                            })
                            time.sleep(2)  # Delay even on error
                            break
            save_fomo_results(momentum_results)

    # Create momentum dictionary using tickers and momentum_results
    momentum_dict = {}
    for ticker, result in zip(tickers, momentum_results):
        momentum_dict[ticker] = result["momentum_score"]

    # Initialize an empty list to store data
    data = []

    # Collect data for each company
    for ticker, keyword in zip(tickers, keywords):
        max_retries = 3
        base_delay = 5  # Base delay in seconds
        for attempt in range(max_retries):
            try:
                # Add delay between ticker requests
                time.sleep(2)  # 2 second delay between ticker data fetches
                
                stock = yf.Ticker(ticker)

                # Basic stock information
                company_name = stock.info.get("longName", "N/A")
                pe_ratio = stock.info.get("trailingPE", "N/A")
                forward_pe = stock.info.get("forwardPE", "N/A")
                revenue_growth = stock.info.get("revenueGrowth", "N/A")
                beta = stock.info.get("beta", "N/A")

                # Fetch trailing 12-month high and low
                try:
                    historical_data = stock.history(period="1y")
                    if not historical_data.empty:
                        trailing_high = historical_data["High"].max()
                        trailing_low = historical_data["Low"].min()
                    else:
                        trailing_high = "N/A"
                        trailing_low = "N/A"
                except Exception as e:
                    print(f"Error fetching historical data for {ticker}: {e}")
                    trailing_high = "N/A"
                    trailing_low = "N/A"

                # Fetch last price and calculate percentage change since market open
                try:
                    last_data = stock.history(period="1d")  # Fetch data for today
                    if not last_data.empty:
                        last_price = last_data["Close"].iloc[-1]
                    else:
                        last_price = "N/A"
                    
                    # Fetch historical data (ensure enough data is retrieved)
                    hist = stock.history(period="5d")  # Fetch last 5 days to ensure a valid close price

                    # Ensure there is enough data and get the last valid close price
                    if "Close" in hist.columns and len(hist) > 1:
                        last_close_price = hist["Close"].dropna().iloc[-2]  # Get the most recent valid close price
                    else:
                        last_close_price = None  # Handle missing data case

                    # Set the open price using the last close price
                    open_price = last_close_price if last_close_price is not None else "N/A"

                    if last_price != "N/A" and open_price != "N/A" and open_price != 0:
                        percent_change = ((last_price - open_price) / open_price) * 100
                    else:
                        percent_change = "N/A"
                except Exception as e:
                    print(f"Error fetching price data for {ticker}: {e}")
                    last_price = "N/A"
                    percent_change = "N/A"

                try:
                    # Fetch historical data for the past 10 trading days
                    hist = stock.history(period="1mo")  # Fetch 2 weeks of data to ensure at least 10 trading days
                    if hist.empty:
                        return {"error": f"No trading data available for {ticker}"}
                    
                    # Current trading volume (real-time)
                    current_volume = int(stock.info.get('volume', 'N/A')/1000000.0)
                    
                    # Calculate average volume over the past 10 trading days
                    avg_volume_10d = int(hist['Volume'].tail(10).mean()/1000000.0)
                except Exception as e:
                    print(f"Error fetching price data for {ticker}: {e}")
                    current_volume = "N/A"
                    avg_volume_10d = "N/A"

                # Fetch quarterly diluted EPS
                eps_values = ["N/A"] * 5  # Default if no data available
                try:
                    # Get quarterly financial data
                    quarterly_financials = stock.quarterly_financials

                    # Check if diluted EPS is available in the data
                    if "Diluted EPS" in quarterly_financials.index:
                        if len(quarterly_financials.loc["Diluted EPS"]) < 5:
                            esp_values_short = quarterly_financials.loc["Diluted EPS"].tolist()
                            eps_values =  esp_values_short + ["N/A"] * (5 - len(esp_values_short))
                        else:
                            eps_values = quarterly_financials.loc["Diluted EPS"].iloc[:5].tolist()
                except Exception as e:
                    print(f"Error fetching quarterly EPS for {ticker}: {e}")

                # Get momentum score from cache
                momentum_score = momentum_dict.get(ticker, 10)

                # Append all data
                data.append({
                    "Ticker": ticker,
                    "Keyword": keyword,
                    "LAST": last_price,
                    "12M Low": trailing_low,
                    "12M High": trailing_high,
                    "% CHG": percent_change,
                    "Vol": current_volume,
                    "10D Vol": avg_volume_10d,
                    "P/E": pe_ratio,
                    "fP/E": forward_pe,
                    "REV GRW": revenue_growth,
                    "BETA": beta,
                    "EPS Q-4": eps_values[4],
                    "EPS Q-3": eps_values[3],
                    "EPS Q-2": eps_values[2],
                    "EPS Q-1": eps_values[1],
                    "EPS Q0": eps_values[0],
                    "Momentum Score": momentum_score
                })
                break  # Success, exit retry loop
            except Exception as e:
                if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited for {ticker}. Retrying after {delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"Error fetching data for {ticker}: {e}")
                    # Add extra delay after an error
                    time.sleep(5)  # 5 second delay after an error
                    break

    # Load data into a DataFrame
    df = pd.DataFrame(data)

    # Ensure numeric columns are properly formatted
    eps_columns = ["EPS Q0", "EPS Q-1", "EPS Q-2", "EPS Q-3", "EPS Q-4"]
    df[eps_columns] = df[eps_columns].apply(pd.to_numeric, errors="coerce")

    # Add a computed column for max EPS of past quarters
    df["Max EPS Past"] = df[["EPS Q-1", "EPS Q-2", "EPS Q-3", "EPS Q-4"]].max(axis=1)

    # Convert Column_A to numeric type
    numeric_columns = ["12M Low", 
                    "12M High", 
                    "Momentum Score", 
                    "P/E", 
                    "fP/E", 
                    "REV GRW", 
                    "BETA", 
                    "EPS Q-4", 
                    "EPS Q-3", 
                    "EPS Q-2", 
                    "EPS Q-1", 
                    "EPS Q0"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1(
        "Equities Dashboard",
        style={
            'textAlign': 'center',
            'fontFamily': 'Tahoma',
            'color': '#333',
            'marginBottom': '20px'
        }
    ),
    html.Div(
        id="last-updated",
        style={
            'textAlign': 'center',
            'fontFamily': 'Tahoma',
            'fontSize': '16px',
            'marginBottom': '20px'
        }
    ),
    html.Div(
        id="plots-container",
        style={
            "display": "flex",
            "justifyContent": "space-around",
            "marginBottom": "30px"
        },
        children=[
            dcc.Graph(id="time-series-plot", style={"width": "48%"}),
            dcc.Graph(id="histogram-plot", style={"width": "48%"})
        ]
    ),
    html.Div(id="table-container"),
    dcc.Interval(
        id="interval-component",
        interval= REFRESH_CYCLE * 60 * 1000,  # REFRESH_CYCLE minutes in milliseconds
        n_intervals=0  # Start immediately
    )
])

# Callback to update the table and timestamp
@app.callback(
    [
        Output("table-container", "children"),
        Output("last-updated", "children"),
        Output("time-series-plot", "figure"),
        Output("histogram-plot", "figure")
    ],
    [Input("interval-component", "n_intervals")]
)
def update_table(n_intervals):
    # Fetch the latest data
    df = fetch_market_data()

    # Calculate momentum scores for each ticker
    momentum_scores = {}
    for ticker in df['Ticker']:
        momentum_scores[ticker] = compute_momentum_score(ticker)

    # Update the DataFrame with momentum scores
    df['Momentum'] = df['Ticker'].map(momentum_scores)

    # Reorder columns to place Momentum Score after 12M High
    column_order = [
        "Ticker", "Keyword", "LAST", 
        "% CHG", "Vol", "Momentum", 
        "12M Low", "12M High", "P/E", "fP/E", "REV GRW", "BETA", "10D Vol", 
        "EPS Q-4", "EPS Q-3", "EPS Q-2", "EPS Q-1", "EPS Q0"
    ]
    df = df[column_order]

    # Prepare the Fear and Greed Index data
    fear_greed_data = calculate_fear_and_greed(start_date="2021-12-01", end_date=datetime.now().strftime("%Y-%m-%d"))
    smoothed_data = calculate_smoothed_fear_and_greed(fear_greed_data, window=FEAR_AND_GREED_SMOOTHING_WINDOW)

    # Create a time series plot for the smoothed Fear and Greed Index
    time_series_fig = {
        "data": [
            {
                "x": smoothed_data["Date"],
                "y": smoothed_data["Smoothed Fear and Greed Index"],
                "type": "scatter",
                "mode": "lines",
                "name": "Smoothed Fear and Greed Index",
                "line": {"color": "orange"}
            }
        ],
        "layout": {
            "title": "Smoothed Fear and Greed Index (Past 2 Years)",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Fear and Greed Index"},
            "template": "plotly_white",
            "shapes": [  # Add horizontal reference lines
                {
                    "type": "line",
                    "x0": smoothed_data["Date"].min(),
                    "x1": smoothed_data["Date"].max(),
                    "y0": 0.5,
                    "y1": 0.5,
                    "line": {"dash": "dash", "color": "gray"},
                    "xref": "x",
                    "yref": "y",
                    "name": "Neutral Zone (0.5)"
                },
                {
                    "type": "line",
                    "x0": smoothed_data["Date"].min(),
                    "x1": smoothed_data["Date"].max(),
                    "y0": 0.45,
                    "y1": 0.45,
                    "line": {"dash": "dash", "color": "red"},
                    "xref": "x",
                    "yref": "y",
                    "name": "Fear Threshold (0.45)"
                },
                {
                    "type": "line",
                    "x0": smoothed_data["Date"].min(),
                    "x1": smoothed_data["Date"].max(),
                    "y0": 0.55,
                    "y1": 0.55,
                    "line": {"dash": "dash", "color": "green"},
                    "xref": "x",
                    "yref": "y",
                    "name": "Greed Threshold (0.55)"
                }
            ]
        }
    }

    # Create a histogram plot for daily Fear and Greed Index
    current_value = fear_greed_data["Fear and Greed Index"].iloc[-1]
    
    histogram_fig = {
        "data": [
            {
                "x": fear_greed_data["Fear and Greed Index"],
                "type": "histogram",
                "name": "Daily Fear and Greed Index",
                "opacity": 0.7,
                "marker": {
                    "color": "lightblue", # Set histogram color to light blue
                    "line": {
                        "color": "white",  # Set the border color to white
                        "width": 1  # Set the border width
                    }
                }
            }
        ],
        "layout": {
            "title": "Daily Fear and Greed Index Distribution",
            "xaxis": {"title": "Fear and Greed Index"},
            "yaxis": {"title": "Frequency"},
            "template": "plotly_white",
            "shapes": [  # Add a vertical line for the current day's value
                {
                    "type": "line",
                    "x0": current_value,
                    "x1": current_value,
                    "y0": 0,
                    "y1": 1,
                    "xref": "x",
                    "yref": "paper",  # Use "paper" to span the full y-axis
                    "line": {"color": "red", "width": 4},
                    "name": f"Today: {current_value:.2f}"
                }
            ],
            "annotations": [
                {
                    "x": current_value,
                    "y": 1,
                    "xref": "x",
                    "yref": "paper",
                    "text": f"Today: {current_value:.2f}",
                    "showarrow": True,
                    "arrowhead": 2,
                    "ax": 0,
                    "ay": -30,
                    "font": {
                        "color": "red",
                        "size": 16,  # Increase font size
                        "family": "Tahoma",
                        "weight": "bold"  # Make the text bold
                    }
                }
            ]
        }
    }

    # Identify numeric columns and set their format
    formatted_columns = []
    for col in df.columns:
        if col != "Max EPS Past":  # Exclude "Max EPS Past" from the table
            if pd.api.types.is_numeric_dtype(df[col]) and col !="Vol" and col != "10D Vol":
                formatted_columns.append(
                    {"name": col, "id": col, "type": "numeric", "format": {"specifier": ".2f"}}
                )
            else:
                formatted_columns.append({"name": col, "id": col})

    # Create a Dash DataTable
    table = dash_table.DataTable(
        columns=formatted_columns,
        data=df.to_dict("records"),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'minWidth': '100px',
            'width': '150px',
            'maxWidth': '200px',
            'fontFamily': 'Tahoma',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {
                    'column_id': '% CHG',
                    'filter_query': '{% CHG} < 0',
                },
                'backgroundColor': 'red',
                'color': 'white',
            },
            {
                'if': {
                    'column_id': '% CHG',
                    'filter_query': '{% CHG} >= 0',
                },
                'backgroundColor': 'green',
                'color': 'white',
            },
            {
                "if": {
                    "filter_query": "{EPS Q0} > {Max EPS Past}",
                    "column_id": "EPS Q0",
                },
                "backgroundColor": "green",
                "color": "white",
            },
            {
                "if": {
                    "filter_query": "{Vol} > {10D Vol}",
                    "column_id": "Vol",
                },
                "backgroundColor": "green",
                "color": "white",
            },
            {
                'if': {
                    'column_id': 'Momentum',
                    'filter_query': '{Momentum} >= 20',
                },
                'backgroundColor': 'green',
                'color': 'white',
            },
            {
                'if': {
                    'column_id': 'Momentum',
                    'filter_query': '{Momentum} <= -20',
                },
                'backgroundColor': 'red',
                'color': 'white',
            }
        ]
    )

    # Get the current timestamp
    timestamp = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return table, timestamp, time_series_fig, histogram_fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
