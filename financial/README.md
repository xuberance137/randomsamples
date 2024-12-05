# Watchlist Tracker App

Live watchlist tracker with essential metrics on a given collection of tickers.

## Preliminary Steps

You need python3 to run this app locally. If you don't have this installed already, go to [python.org](https://www.python.org/downloads/macos/) to download and run the installer. 

1. Install virtualenv

```
pip install virtualenv
```

2. Create a virtual env

```
virtualenv -p python3 venv
```

3. Activate the virtual env

```
source venv/bin/activate
```

4. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

1. Create a file `tickers.txt` with one stock ticker on each line in the txt file and place in this folder.

2. Run the following from the current folder.
```
python app.py
```

3. Open your browser and go to `http://127.0.0.1:8050/`

App natively refreshes every 30 mins. If you want it to be more frequently refresh, change the REFRESH_CYCLE variable (representing minutes between refresh cycles) at the top of app.py.


# Running Scripts

Some scripts like `topAssetOwners.py` use the selenium package that drives off a headless browser. This requires the installation of the chrome driver. These instructions will enable to get started on these when using mac OSX.

1. Install homebrew

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install the chrome driver

This will place the chrome driver at `/usr/local/bin`:

```
brew install chromedriver
```
3. Test the chrome driver (enable it to run even when not verified)

```
open /usr/local/bin/chromedriver
```

4. Run the script

```
python topAssetOwners.py 084670702
```



