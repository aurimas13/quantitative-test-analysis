 import pandas as pd
import numpy as np

def detect_errors(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['P_Ticker', 'Date'], inplace=True)
    
    # Detect Price Anomalies
    df['Prev_Price'] = df.groupby('P_Ticker')['Price'].shift(1)
    df['Price_Change_%'] = ((df['Price'] - df['Prev_Price']) / df['Prev_Price']) * 100
    price_anomalies = df[abs(df['Price_Change_%']) > 20]
    
    # Detect Quantity Changes Without Trade Records
    df['Prev_Close_Quantity'] = df.groupby('P_Ticker')['Close Quantity'].shift(1)
    df['Quantity_Change'] = df['Open Quantity'] - df['Prev_Close_Quantity']
    missing_trades = df[(df['Quantity_Change'] != 0) & (df['Prev_Close_Quantity'].notna())]
    
    # Output results
    return {'price_anomalies': price_anomalies, 'missing_trades': missing_trades}

if __name__ == "__main__":
    file_path = "data/sample_test.xlsx"
    errors = detect_errors(file_path)
    for key, df in errors.items():
        df.to_csv(f"reports/{key}.csv", index=False)
    print("Error detection completed. Results saved in reports/")

