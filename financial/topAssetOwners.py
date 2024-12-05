"""
CUSIP-Based Institutional Holdings Scraper and Formatter

This script retrieves and formats institutional holdings data for a given CUSIP identifier using the 
13f.info website. The primary functionalities include scraping data, processing it into a tabular 
format, and customizing the output to include specific institutional managers and the top institutional 
managers based on share difference.

Key Features:
- Accepts a CUSIP identifier as a command-line argument.
- Utilizes Selenium for rendering JavaScript-based tables and BeautifulSoup for parsing the page source.
- Extracts, filters, and formats institutional holdings data into a DataFrame.
- Ensures inclusion of specific institutional managers (e.g., Vanguard Group Inc, BlackRock Inc, State Street Corp).
- Dynamically calculates and includes the top N managers based on share difference.

Dependencies:
- Selenium for web scraping and rendering JavaScript content.
- pandas for DataFrame manipulations.
- BeautifulSoup (from bs4) for HTML parsing.
- argparse for command-line argument parsing.

Input:
- A CUSIP identifier passed as a command-line argument.

Output:
- A formatted DataFrame containing:
  - Manager
  - Q2 2024 Shares
  - Q3 2024 Shares
  - Difference (Diff)
  - Change Percentage (Chg %)

Usage:
1. Ensure ChromeDriver is installed and accessible in the specified path (`CHROME_DRIVER_PATH`).
2. Run the script with the desired CUSIP as an argument:
python script_name.py <CUSIP>

"""

import argparse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import StringIO
from bs4 import BeautifulSoup

# Path to ChromeDriver (adjust based on your system setup)
CHROME_DRIVER_PATH = '/usr/local/bin/chromedriver'
NUM_MANAGERS = 10

# Initialize the Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run in headless mode
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

def arg_parse():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Store a string passed as an argument.")
    
    # Add a single string argument
    parser.add_argument("input_string", type=str, help="CUSIP string")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Store the string in a variable
    CUSIP = args.input_string
    
    # Print the stored variable
    print(f"CUSIP: {CUSIP}")
    return CUSIP

def format_table(CUSIP):

    print('Loading Chrome Driver')
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    
    # URL to fetch
    print('Loading Table')
    url = 'https://13f.info/cusip/'+str(CUSIP)+'/2024/3/compare/2024/2'
    # print(url)
    driver.get(url)
    
    # Wait for the table to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        # Extract the page source after rendering
        page_source = driver.page_source
    finally:
        driver.quit()
    
    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Locate the table and extract its content
    table = soup.find('table')
    if table:
        # Use StringIO to wrap the table string for pd.read_html
        table_html = str(table)
        df = pd.read_html(StringIO(table_html))[0]
    else:
        raise Exception("Table not found on the page.")
    
    #df.drop('DIF', axis=1, inplace=True)
    
    print('Formatting Table to custom view')
    # List of specific managers
    specific_managers = ['VANGUARD GROUP INC', 'BlackRock Inc', 'STATE STREET CORP']
    
    # Filter rows for specific managers
    df_specific = df[df['Manager'].isin(specific_managers)]
    
    # Add missing specific managers with zeros in other columns
    for manager in specific_managers:
        if manager not in df_specific['Manager'].values:
            df_specific = pd.concat([
                df_specific,
                pd.DataFrame({
                    'Manager': [manager],
                    'Q2 2024  Shares': [0],
                    'Q3 2024  Shares': [0],
                    'Diff': [0],
                    'Chg %': [0]
                })
            ])
    
    # Convert 'DIF' column to numeric for sorting
    df['Diff'] = pd.to_numeric(df['Diff'], errors='coerce')
    
    # Exclude specific managers from top 10 DIF calculation
    excluded_managers = specific_managers #['Vanguard Group Inc', 'BlackRock Inc', 'State Street Corp']
    df_filtered = df[~df['Manager'].isin(excluded_managers)]
    
    # Get top 10 managers by 'DIF' excluding specific managers
    df_top10 = df_filtered.nlargest(NUM_MANAGERS, 'Diff')
    
    # Combine the specific managers and top 10 managers
    result_df = pd.concat([df_specific, df_top10]).drop_duplicates().reset_index(drop=True)
    
    result_df = result_df[['Manager', 'Q2 2024  Shares', 'Q3 2024  Shares', 'Diff', 'Chg %']]

    return result_df


def main():

    # lite_cusips = {'BRK-B':'084670702'}
    
    # for index, (ticker, cusip) in enumerate(lite_cusips.items()):
    #     print(f"{index}: {ticker} -> {cusip}")

    #     result_df = format_table(cusip)
    
    #     # Display the resulting DataFrame
    #     print(result_df)
        
    cusip = arg_parse()
    
    result_df = format_table(cusip)

    # Display the resulting DataFrame
    print(result_df)


if __name__ == "__main__":
    main()