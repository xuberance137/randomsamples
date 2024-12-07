import dash
from dash import dcc, html, Input, Output
import dash_table
import yfinance as yf
import pandas as pd
from datetime import datetime

REFRESH_CYCLE = 30 # in minutes

# Define a function to fetch data
def fetch_market_data():
    """
    Fetches market data for tickers in 'tickers-shortlist.txt' and returns a DataFrame.
    """
    # Read tickers from file
    with open("tickers.txt", "r") as file:
        tickers = [line.strip() for line in file]

    # Initialize an empty list to store data
    data = []

    # Collect data for each company
    for ticker in tickers:
        try:
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
                open_price = stock.info.get("regularMarketOpen", "N/A")
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
                    eps_values = quarterly_financials.loc["Diluted EPS"].iloc[:5].tolist()
            except Exception as e:
                print(f"Error fetching quarterly EPS for {ticker}: {e}")

            # Append all data
            data.append({
                "Ticker": ticker,
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
            })

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # Load data into a DataFrame
    df = pd.DataFrame(data)

    # Ensure numeric columns are properly formatted
    eps_columns = ["EPS Q0", "EPS Q-1", "EPS Q-2", "EPS Q-3", "EPS Q-4"]
    df[eps_columns] = df[eps_columns].apply(pd.to_numeric, errors="coerce")

    # Add a computed column for max EPS of past quarters
    df["Max EPS Past"] = df[["EPS Q-1", "EPS Q-2", "EPS Q-3", "EPS Q-4"]].max(axis=1)

    # Convert Column_A to numeric type
    for col in ["12M Low", "12M High", "P/E", "fP/E", "REV GRW", "BETA", "EPS Q-4", "EPS Q-3", "EPS Q-2", "EPS Q-1", "EPS Q0"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1(
        "Watchlist Dashboard",
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
    html.Div(id="table-container"),
    dcc.Interval(
        id="interval-component",
        interval= REFRESH_CYCLE * 60 * 1000,  # REFRESH_CYCLE minutes in milliseconds
        n_intervals=0  # Start immediately
    )
])

# Callback to update the table and timestamp
@app.callback(
    [Output("table-container", "children"),
     Output("last-updated", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_table(n_intervals):
    # Fetch the latest data
    df = fetch_market_data()

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
            }
        ]
    )

    # Get the current timestamp
    timestamp = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return table, timestamp

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
