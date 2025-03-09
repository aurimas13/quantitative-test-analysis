#!/usr/bin/env python
"""
Command-line script to run the PortfolioValidator checks.
"""

import argparse
import sys
from validator import PortfolioValidator

def main():
    parser = argparse.ArgumentParser(description="Run portfolio validation checks.")
    parser.add_argument('--data', required=True, help="Path to the Excel/CSV file containing the portfolio data.")
    parser.add_argument('--report', default=None, help="Optional path to save a Markdown report.")
    parser.add_argument('--fig', default=None, help="Optional path to save the summary plots (PNG).")

    args = parser.parse_args()

    # Create validator instance
    validator = PortfolioValidator(data_path=args.data)

    # Run checks
    results = validator.run_all_checks()

    # Print summary to stdout
    if 'summary' in results:
        print("\n===== VALIDATION SUMMARY =====")
        for k, v in results['summary'].items():
            print(f"{k}: {v}")
        print("==============================\n")

    # Optionally save a report
    if args.report:
        _ = validator.generate_report(save_path=args.report)

    # Optionally save a figure
    if args.fig:
        validator.visualize_issues(save_path=args.fig)
    else:
        # Show on screen if not saving
        validator.visualize_issues()

if __name__ == '__main__':
    sys.exit(main())
