import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class PortfolioValidator:
    """
    A class to validate portfolio data and detect errors and inconsistencies.
    """

    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize the PortfolioValidator with data.

        Args:
            data_path: Path to the Excel/CSV file containing portfolio data.
            df: DataFrame containing portfolio data (alternative to data_path).
        """
        self.data_path = data_path
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.load_data(data_path)
        else:
            self.df = None

        self.error_reports = {}
        self.summary = {}

    def load_data(self, data_path: str) -> None:
        """
        Load portfolio data from file.

        Args:
            data_path: Path to the Excel/CSV file containing portfolio data
        """
        try:
            if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                self.df = pd.read_excel(data_path)
            elif data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path)
            elif data_path.endswith('.txt'):
                self.df = pd.read_csv(data_path, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            # Convert date column to datetime if present
            if 'Date' in self.df.columns:
                try:
                    self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
                except:
                    try:
                        self.df['Date'] = pd.to_datetime(self.df['Date'])
                    except:
                        print("Warning: Could not convert Date column to datetime")

            print(f"Loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = None

    def preprocess_data(self) -> None:
        """
        Preprocess the data for analysis.
        - Ensure correct data types
        - Handle missing values appropriately
        - Create derived fields if needed
        """
        if self.df is None:
            print("No data to preprocess")
            return

        # Convert columns to appropriate data types
        numeric_columns = [
            'Opening Weights', 'Open Quantity', 'Close Quantity', 'Closing Weights',
            'Price', 'Exchange Rate', 'Value in USD', 'Price Yesterday',
            'Stock Movement', 'Performance Contribution', 'Trade Price',
            'Trade Day Move', 'Trade Weight', 'Return Adjustments', 'Total Return',
            'Calculated NAV', 'NAV Yesterday', 'Close Weight ABS', 'Dollar PnL',
            'SharesOut', 'Market_Cap', 'Traded Today'
        ]

        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Convert boolean columns (if they exist)
        bool_columns = ['IS_Currency', 'IS_Future', 'Short_Pos']
        for col in bool_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(bool)

        # Create lookup dictionaries for fast access
        self.create_lookup_dictionaries()

        # Calculate expected values for certain fields for validation
        self.calculate_expected_values()

    def create_lookup_dictionaries(self) -> None:
        """Create lookup dictionaries for fast access to data."""
        self.price_lookup = {}
        self.position_lookup = {}

        for _, row in self.df.iterrows():
            date = row.get('Date')
            ticker = row.get('P_Ticker')

            if date and ticker:
                # Price lookup
                if date not in self.price_lookup:
                    self.price_lookup[date] = {}
                self.price_lookup[date][ticker] = {
                    'Price': row.get('Price'),
                    'Price Yesterday': row.get('Price Yesterday'),
                    'Trade Price': row.get('Trade Price')
                }

                # Position lookup
                if date not in self.position_lookup:
                    self.position_lookup[date] = {}
                self.position_lookup[date][ticker] = {
                    'Open Quantity': row.get('Open Quantity'),
                    'Close Quantity': row.get('Close Quantity'),
                    'Traded Today': row.get('Traded Today')
                }

    def calculate_expected_values(self) -> None:
        """Calculate expected values for certain fields for validation."""
        self.df['Expected_Value_in_USD'] = self.df.apply(
            lambda row: self._calculate_expected_value_in_usd(row), axis=1
        )
        self.df['Expected_Stock_Movement'] = self.df.apply(
            lambda row: self._calculate_expected_stock_movement(row), axis=1
        )
        self.df['Expected_Performance_Contribution'] = self.df.apply(
            lambda row: self._calculate_expected_performance_contribution(row), axis=1
        )
        self.df['Expected_Trade_Day_Move'] = self.df.apply(
            lambda row: self._calculate_expected_trade_day_move(row), axis=1
        )
        self.df['Expected_Total_Return'] = self.df.apply(
            lambda row: self._calculate_expected_total_return(row), axis=1
        )

    def _calculate_expected_value_in_usd(self, row: pd.Series) -> float:
        """Calculate expected Value in USD for a row."""
        try:
            if pd.isna(row.get('Close Quantity')) or row.get('Close Quantity') == 0:
                return 0

            value = row.get('Close Quantity') * row.get('Price') * row.get('Exchange Rate')
            return value
        except:
            return np.nan

    def _calculate_expected_stock_movement(self, row: pd.Series) -> float:
        """Calculate expected Stock Movement (%) for a row, comparing Price vs. Price Yesterday."""
        try:
            if pd.isna(row.get('Price')) or pd.isna(row.get('Price Yesterday')) or row.get('Price Yesterday') == 0:
                return np.nan

            return (row.get('Price') - row.get('Price Yesterday')) / row.get('Price Yesterday') * 100
        except:
            return np.nan

    def _calculate_expected_performance_contribution(self, row: pd.Series) -> float:
        """
        Calculate expected Performance Contribution for a row.
        Typically: (Opening Weights / 100) * (Stock Movement / 100) * 100
        i.e., Opening Weights * Stock Movement, but scaled for percentages.
        """
        try:
            if pd.isna(row.get('Opening Weights')) or pd.isna(row.get('Stock Movement')):
                return np.nan

            return (row.get('Opening Weights') / 100.0) * (row.get('Stock Movement') / 100.0) * 100.0
        except:
            return np.nan

    def _calculate_expected_trade_day_move(self, row: pd.Series) -> float:
        """Calculate expected Trade Day Move (%), comparing Trade Price vs. Price."""
        try:
            if pd.isna(row.get('Trade Price')) or pd.isna(row.get('Price')) or row.get('Price') == 0:
                return np.nan
            return (row.get('Trade Price') - row.get('Price')) / row.get('Price') * 100
        except:
            return np.nan

    def _calculate_expected_total_return(self, row: pd.Series) -> float:
        """Calculate expected Total Return as Performance Contribution + Return Adjustments."""
        try:
            return row.get('Performance Contribution', 0) + row.get('Return Adjustments', 0)
        except:
            return np.nan

    def check_for_price_consistency(self, tolerance: float = 10.0) -> List[Dict]:
        """
        Check for inconsistent prices between consecutive days.
        We interpret 'tolerance' as an allowable % difference.

        Args:
            tolerance: Allowed percentage difference between yesterday's price 
                       and previous day's price before flagging as an issue.

        Returns:
            A list of dicts describing each inconsistency found.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []
        sorted_df = self.df.sort_values(['P_Ticker', 'Date'])

        # Group by ticker
        ticker_groups = sorted_df.groupby('P_Ticker')

        for ticker, group in ticker_groups:
            group = group.reset_index(drop=True)
            for i in range(1, len(group)):
                current_row = group.iloc[i]
                prev_row = group.iloc[i - 1]

                # Skip if any price value is NaN
                if pd.isna(current_row['Price']) or pd.isna(prev_row['Price']):
                    continue

                # Calculate the % difference between today's "Price Yesterday" and previous day's "Price"
                if (not pd.isna(current_row['Price Yesterday']) 
                    and not pd.isna(prev_row['Price']) 
                    and prev_row['Price'] != 0):
                    perc_diff = ((current_row['Price Yesterday'] - prev_row['Price']) / prev_row['Price']) * 100
                    if abs(perc_diff) > tolerance:
                        issues.append({
                            'Ticker': ticker,
                            'Date': current_row['Date'],
                            'Price Yesterday': current_row['Price Yesterday'],
                            'Previous Day Price': prev_row['Price'],
                            'Difference': current_row['Price Yesterday'] - prev_row['Price'],
                            'Difference %': perc_diff
                        })

        self.error_reports['price_consistency'] = issues
        return issues

    def check_for_missing_trades(self) -> List[Dict]:
        """
        Check for changes in holdings without corresponding trade information.

        Returns:
            A list of dicts describing each missing trade found.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []
        sorted_df = self.df.sort_values(['P_Ticker', 'Date'])

        # Group by ticker
        ticker_groups = sorted_df.groupby('P_Ticker')

        for ticker, group in ticker_groups:
            group = group.reset_index(drop=True)
            for i in range(1, len(group)):
                current_row = group.iloc[i]
                prev_row = group.iloc[i - 1]

                # If open != previous day close, but no trade to explain it
                if (not pd.isna(current_row['Open Quantity']) 
                    and not pd.isna(prev_row['Close Quantity']) 
                    and current_row['Open Quantity'] != prev_row['Close Quantity']):
                    issues.append({
                        'Ticker': ticker,
                        'Date': current_row['Date'],
                        'Open Quantity': current_row['Open Quantity'],
                        'Previous Day Close Quantity': prev_row['Close Quantity'],
                        'Difference': current_row['Open Quantity'] - prev_row['Close Quantity']
                    })

        # Check for day-level changes in open/close with no 'Traded Today'
        for _, row in self.df.iterrows():
            if (not pd.isna(row['Open Quantity']) 
                and not pd.isna(row['Close Quantity']) 
                and row['Open Quantity'] != row['Close Quantity']
                and (pd.isna(row['Traded Today']) or row['Traded Today'] == 0)):
                issues.append({
                    'Ticker': row['P_Ticker'],
                    'Date': row['Date'],
                    'Open Quantity': row['Open Quantity'],
                    'Close Quantity': row['Close Quantity'],
                    'Difference': row['Close Quantity'] - row['Open Quantity'],
                    'Traded Today': row['Traded Today']
                })

        self.error_reports['missing_trades'] = issues
        return issues

    def check_for_calculation_errors(self, tolerance: float = 0.1) -> List[Dict]:
        """
        Check for errors in calculation of weights, values, and other ratios.
        
        Args:
            tolerance: Allowed fractional difference (0.1 => 10%) before flagging as an error.

        Returns:
            A list of dicts describing calculation errors.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []

        # Check Value in USD calculations
        for _, row in self.df.iterrows():
            actual_val = row.get('Value in USD')
            expected_val = row.get('Expected_Value_in_USD')
            if not pd.isna(actual_val) and not pd.isna(expected_val):
                if abs(expected_val) < 1e-8 and abs(actual_val) < 1e-8:
                    # both near zero, skip
                    continue
                if expected_val != 0:
                    relative_diff = abs((actual_val - expected_val) / expected_val)
                    if relative_diff > tolerance:
                        issues.append({
                            'Type': 'Value in USD',
                            'Ticker': row.get('P_Ticker'),
                            'Date': row.get('Date'),
                            'Actual': actual_val,
                            'Expected': expected_val,
                            'Difference': actual_val - expected_val,
                            'Difference %': relative_diff * 100
                        })

        # Check Stock Movement calculations
        for _, row in self.df.iterrows():
            actual_sm = row.get('Stock Movement')
            expected_sm = row.get('Expected_Stock_Movement')
            if not pd.isna(actual_sm) and not pd.isna(expected_sm):
                diff = abs(actual_sm - expected_sm)
                if diff > tolerance:
                    issues.append({
                        'Type': 'Stock Movement',
                        'Ticker': row.get('P_Ticker'),
                        'Date': row.get('Date'),
                        'Actual': actual_sm,
                        'Expected': expected_sm,
                        'Difference': actual_sm - expected_sm
                    })

        # Check Performance Contribution calculations
        for _, row in self.df.iterrows():
            actual_pc = row.get('Performance Contribution')
            expected_pc = row.get('Expected_Performance_Contribution')
            if not pd.isna(actual_pc) and not pd.isna(expected_pc):
                diff = abs(actual_pc - expected_pc)
                if diff > tolerance:
                    issues.append({
                        'Type': 'Performance Contribution',
                        'Ticker': row.get('P_Ticker'),
                        'Date': row.get('Date'),
                        'Actual': actual_pc,
                        'Expected': expected_pc,
                        'Difference': actual_pc - expected_pc
                    })

        # Check Total Return calculations
        for _, row in self.df.iterrows():
            actual_tr = row.get('Total Return')
            expected_tr = row.get('Expected_Total_Return')
            if not pd.isna(actual_tr) and not pd.isna(expected_tr):
                diff = abs(actual_tr - expected_tr)
                if diff > tolerance:
                    issues.append({
                        'Type': 'Total Return',
                        'Ticker': row.get('P_Ticker'),
                        'Date': row.get('Date'),
                        'Actual': actual_tr,
                        'Expected': expected_tr,
                        'Difference': actual_tr - expected_tr
                    })

        self.error_reports['calculation_errors'] = issues
        return issues

    def check_for_trade_price_inconsistencies(self, tolerance: float = 10.0) -> List[Dict]:
        """
        Check for inconsistencies between trade prices and holding prices.
        We interpret 'tolerance' as an allowable % difference.

        Args:
            tolerance: Allowed percentage difference before flagging as an error.

        Returns:
            A list of dicts describing trade price inconsistencies.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []

        for _, row in self.df.iterrows():
            # Skip if no trade occurred
            traded_today = row.get('Traded Today', 0)
            if pd.isna(traded_today) or traded_today == 0:
                continue

            if pd.isna(row.get('Trade Price')) or pd.isna(row.get('Price')):
                continue

            if row['Price'] != 0:
                diff_pct = abs((row['Trade Price'] - row['Price']) / row['Price']) * 100
                if diff_pct > tolerance:
                    issues.append({
                        'Ticker': row.get('P_Ticker'),
                        'Date': row.get('Date'),
                        'Trade Price': row.get('Trade Price'),
                        'Holding Price': row.get('Price'),
                        'Difference': row['Trade Price'] - row['Price'],
                        'Difference %': diff_pct
                    })

        self.error_reports['trade_price_inconsistencies'] = issues
        return issues

    def check_for_weight_consistency(self, tolerance: float = 0.1) -> List[Dict]:
        """
        Check if portfolio weights sum to 100% (+/- tolerance).

        Args:
            tolerance: e.g., 0.1 => 0.1 absolute difference from 100. 
                       If your data is in percentages, we interpret 
                       this as "do they sum to 100 +/- 0.1" ?

        Returns:
            A list of dicts describing weight consistency issues.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []

        date_groups = self.df.groupby('Date')
        for date, group in date_groups:
            if 'Opening Weights' in group.columns:
                opening_sum = group['Opening Weights'].sum()
                if abs(opening_sum - 100) > tolerance:
                    issues.append({
                        'Date': date,
                        'Type': 'Opening Weights',
                        'Sum': opening_sum,
                        'Expected': 100,
                        'Difference': opening_sum - 100
                    })
            if 'Closing Weights' in group.columns:
                closing_sum = group['Closing Weights'].sum()
                if abs(closing_sum - 100) > tolerance:
                    issues.append({
                        'Date': date,
                        'Type': 'Closing Weights',
                        'Sum': closing_sum,
                        'Expected': 100,
                        'Difference': closing_sum - 100
                    })

        self.error_reports['weight_consistency'] = issues
        return issues

    def check_for_cash_flow_consistency(self) -> List[Dict]:
        """
        Check for consistency in cash flows (total NAV should account for performance).

        Returns:
            A list of dicts describing cash flow consistency issues.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []
        sorted_df = self.df.sort_values('Date')
        date_groups = sorted_df.groupby('Date')

        prev_date = None
        prev_nav = None

        for date, group in date_groups:
            curr_nav = group['Calculated NAV'].iloc[0] if 'Calculated NAV' in group.columns else None
            nav_yesterday = group['NAV Yesterday'].iloc[0] if 'NAV Yesterday' in group.columns else None

            # Compare with previous day
            if prev_date is not None and prev_nav is not None and not pd.isna(nav_yesterday):
                if abs(nav_yesterday - prev_nav) > 0.1:
                    issues.append({
                        'Date': date,
                        'NAV Yesterday': nav_yesterday,
                        'Previous Day NAV': prev_nav,
                        'Difference': nav_yesterday - prev_nav
                    })

            # Compare the current NAV to expected (if we have total performance or sum of performance)
            # We sum the day's performance across all rows in that date group.
            if 'Performance Contribution' in group.columns and not pd.isna(nav_yesterday):
                total_perf = group['Performance Contribution'].sum()
                if not pd.isna(curr_nav):
                    expected_nav = nav_yesterday * (1 + total_perf / 100.0)
                    # Compare current nav to expected
                    if abs(curr_nav - expected_nav) > (0.1 * abs(curr_nav)):
                        issues.append({
                            'Date': date,
                            'Current NAV': curr_nav,
                            'NAV Yesterday': nav_yesterday,
                            'Total Performance (%)': total_perf,
                            'Expected NAV': expected_nav,
                            'Difference': curr_nav - expected_nav
                        })

            prev_date = date
            prev_nav = curr_nav

        self.error_reports['cash_flow_consistency'] = issues
        return issues

    def check_for_currency_issues(self) -> List[Dict]:
        """
        Check for inconsistencies in currency exchange rates.
        
        Returns:
            A list of dicts describing currency issues.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []
        if not {'Currency', 'Exchange Rate'}.issubset(self.df.columns):
            self.error_reports['currency_issues'] = issues
            return issues

        # Group by (Currency, Date) to check if multiple rates exist
        curr_df = self.df.dropna(subset=['Currency', 'Exchange Rate'])
        curr_groups = curr_df.groupby(['Currency', 'Date'])

        for (currency, date), group in curr_groups:
            # If the same currency has multiple distinct exchange rates on the same day
            rates = group['Exchange Rate'].unique()
            if len(rates) > 1:
                issues.append({
                    'Currency': currency,
                    'Date': date,
                    'Exchange Rates': rates.tolist(),
                    'Min Rate': min(rates),
                    'Max Rate': max(rates),
                    'Max Difference': max(rates) - min(rates)
                })

        # Check for suspicious rates
        for idx, row in self.df.iterrows():
            currency = row.get('Currency')
            rate = row.get('Exchange Rate')
            if pd.isna(currency) or pd.isna(rate):
                continue

            # Non-USD with rate 1.0 => suspicious
            if currency != 'USD' and abs(rate - 1.0) < 0.001:
                issues.append({
                    'Ticker': row.get('P_Ticker'),
                    'Date': row.get('Date'),
                    'Currency': currency,
                    'Exchange Rate': rate,
                    'Issue': 'Non-USD currency with exchange rate of 1.0'
                })

            # USD with rate not ~1.0 => suspicious
            if currency == 'USD' and abs(rate - 1.0) > 0.001:
                issues.append({
                    'Ticker': row.get('P_Ticker'),
                    'Date': row.get('Date'),
                    'Currency': currency,
                    'Exchange Rate': rate,
                    'Issue': 'USD with exchange rate not equal to 1.0'
                })

        self.error_reports['currency_issues'] = issues
        return issues

    def check_for_negative_prices_and_rates(self) -> List[Dict]:
        """
        Example of an additional check to highlight negative or zero prices or FX rates.
        """
        if self.df is None:
            return [{"error": "No data available"}]

        issues = []
        # Price or Exchange Rate should not be negative or zero (unless there's some short or special case)
        for idx, row in self.df.iterrows():
            price = row.get('Price')
            exch = row.get('Exchange Rate')
            ticker = row.get('P_Ticker')
            date = row.get('Date')

            if not pd.isna(price) and price <= 0:
                issues.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Field': 'Price',
                    'Value': price,
                    'Issue': 'Price is negative or zero'
                })

            if not pd.isna(exch) and exch <= 0:
                issues.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Field': 'Exchange Rate',
                    'Value': exch,
                    'Issue': 'Exchange rate is negative or zero'
                })

        self.error_reports['negative_prices_or_rates'] = issues
        return issues

    def run_all_checks(self) -> Dict:
        """
        Run all validation checks and compile results.

        Returns:
            Dict containing all validation results
        """
        if self.df is None:
            return {"error": "No data available"}

        # Preprocess data first
        self.preprocess_data()

        # Run all checks
        price_issues = self.check_for_price_consistency()
        trade_issues = self.check_for_missing_trades()
        calc_issues = self.check_for_calculation_errors()
        trade_price_issues = self.check_for_trade_price_inconsistencies()
        weight_issues = self.check_for_weight_consistency()
        cash_flow_issues = self.check_for_cash_flow_consistency()
        currency_issues = self.check_for_currency_issues()
        negative_prices_issues = self.check_for_negative_prices_and_rates()

        # Compile summary
        self.summary = {
            'price_consistency': len(price_issues),
            'missing_trades': len(trade_issues),
            'calculation_errors': len(calc_issues),
            'trade_price_inconsistencies': len(trade_price_issues),
            'weight_consistency': len(weight_issues),
            'cash_flow_consistency': len(cash_flow_issues),
            'currency_issues': len(currency_issues),
            'negative_prices_or_rates': len(negative_prices_issues),
            'total_issues': (
                len(price_issues) + len(trade_issues) + len(calc_issues) +
                len(trade_price_issues) + len(weight_issues) +
                len(cash_flow_issues) + len(currency_issues) + len(negative_prices_issues)
            )
        }

        return {
            'summary': self.summary,
            'price_consistency': price_issues,
            'missing_trades': trade_issues,
            'calculation_errors': calc_issues,
            'trade_price_inconsistencies': trade_price_issues,
            'weight_consistency': weight_issues,
            'cash_flow_consistency': cash_flow_issues,
            'currency_issues': currency_issues,
            'negative_prices_or_rates': negative_prices_issues
        }

    def generate_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive report of all issues found in Markdown format.

        Args:
            save_path: Path to save the report (optional)

        Returns:
            Report in markdown format
        """
        if not self.error_reports:
            self.run_all_checks()

        report = "# Portfolio Validation Report\n\n"

        # Add summary section
        report += "## Summary\n\n"
        report += "| Issue Type | Count |\n"
        report += "| --- | --- |\n"

        for issue_type, count in self.summary.items():
            if issue_type != 'total_issues':
                report += f"| {issue_type.replace('_', ' ').title()} | {count} |\n"
        report += f"| **Total Issues** | **{self.summary.get('total_issues', 0)}** |\n\n"

        # Detailed sections
        for issue_type, issues in self.error_reports.items():
            if not issues:
                continue
            report += f"## {issue_type.replace('_', ' ').title()}\n\n"
            report += f"Found {len(issues)} issues.\n\n"
            report += "### Examples\n\n"
            for i, issue in enumerate(issues[:5]):
                report += f"**Issue {i+1}**\n\n"
                for key, value in issue.items():
                    report += f"- {key}: {value}\n"
                report += "\n"

        # Optionally save
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")

        return report

    def visualize_issues(self, save_path: str = None) -> None:
        """
        Create visualizations of the detected issues.
        Args:
            save_path: Path to save the visualizations (optional)
        """
        if not self.error_reports:
            self.run_all_checks()

        # Summaries for bar/pie charts
        plt.figure(figsize=(14, 8))

        issue_types = [k for k in self.summary.keys() if k != 'total_issues']
        counts = [self.summary[k] for k in issue_types]

        # Bar Chart
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(issue_types)), counts)
        plt.xticks(range(len(issue_types)), [t.replace('_', '\n').title() for t in issue_types], rotation=45)
        plt.title('Number of Issues by Type')
        plt.ylabel('Count')
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count),
                     ha='center', va='bottom')

        # Pie Chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=[t.replace('_', ' ').title() for t in issue_types], autopct='%1.1f%%')
        plt.title('Distribution of Issues by Type')

        plt.tight_layout()

        # Optionally save
        if save_path:
            plt.savefig(save_path)
            print(f"Visualizations saved to {save_path}")

        plt.show()
