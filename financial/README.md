# Watchlist Tracker App

Live watchlist tracker with essential metrics on a given collection of tickers.

## Preliminary Steps

1. Create a virtual env

```
virtualenv -p python3 venv
```

2. Activate the virtual env

```
source venv/bin/activate
```

3. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt

```

### Run the app

1. Create a file `tickers.txt` with one stock ticker on each line in the txt file.

2. Run the following from the current folder.
```
python app.py
```

3. Open your browser and go to `http://127.0.0.1:8050/`

App natively refreshes every 30 mins. If you want it to be more frequently refresh, change the REFRESH_CYCLE variable (representing minutes between refresh cycles) at the top of app.py.




