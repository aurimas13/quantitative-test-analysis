# Portfolio Data Validation

This repository provides a **step-by-step** solution to identify errors in a hypothetical global stock portfolio dataset. 

**Key Features**:
- Uses Python (Pandas, NumPy) for data analysis.
- Deploys multiple validation checks (price consistency, trade reconciliation, calculation errors, and more).
- Offers a command-line script (`run_validator.py`) plus a Jupyter notebook (`portfolio_validation.ipynb`) for interactive exploration.
- Generates a Markdown report summarizing detected issues, and optionally saves visual charts.

---

## What Errors Are Checked?

1. **Price Consistency**  
   - Compares `Price Yesterday` in the current row to the previous day’s `Price` (for the same ticker).  
   - Flags large discrepancies.

2. **Missing Trades**  
   - Detects changes in `Open Quantity` vs. prior day’s `Close Quantity` without a recorded trade.  
   - Also flags rows where `Open Quantity` ≠ `Close Quantity` but no `Traded Today` is present.

3. **Calculation Errors**  
   - Validates that `Value in USD` ≈ `Quantity * Price * Exchange Rate`, and that `Stock Movement`, `Performance Contribution`, etc., align with expected formulas.

4. **Trade Price Inconsistencies**  
   - Checks if `Trade Price` deviates excessively from the day’s `Price`.

5. **Weight Summation**  
   - Ensures each day’s `Opening Weights` and `Closing Weights` sum up to ~100%.

6. **Cash Flow Consistency (NAV)**  
   - Confirms that day-to-day `NAV Yesterday` aligns with the previous day’s `Calculated NAV`, and that `NAV` changes match daily performance.

7. **Currency Issues**  
   - Flags suspicious or inconsistent exchange rates (e.g., non-USD currency with a rate of 1, or multiple rates for the same currency on the same day).

8. **Negative Prices/Rates**  
   - Detects any negative or zero `Price` or `Exchange Rate` values.

---

## Step-by-Step Guide: How the Validation Works

1. **Load Data**  
   - Data is read from an Excel/CSV file (e.g., `Test.xlsx`) via `PortfolioValidator.load_data()` or by passing a `DataFrame`.  
   - Dates, numeric columns, and booleans are standardized.

2. **Preprocess & Compute Expected Values**  
   - The script calculates derived fields like `Expected_Value_in_USD`, `Expected_Stock_Movement`, etc.  
   - These expected fields are used as reference points for the checks.

3. **Run Checks**  
   - Each validation method runs on the data (price consistency, missing trades, etc.).  
   - Issues are stored in `self.error_reports`.

4. **Generate Summary & Report**  
   - A dictionary summarizing the counts of each issue type is stored in `self.summary`.  
   - Optionally produce a Markdown report with details of the flagged rows.

5. **Visualize**  
   - Generate bar and pie charts showing issue distribution.  
   - Optionally produce a heatmap by ticker if you want deeper analysis (already partly in the code).

6. **Interpret & Fix**  
   - Check flagged rows, correct the data if needed (in your actual system), then re-run the validator.

---

## Scripts Overview

- **`scripts/validator.py`**  
  - Main validation class, `PortfolioValidator`, containing:
    - `load_data()`: Loads data from Excel/CSV.
    - `preprocess_data()`: Ensures correct data types and calculates expected fields.
    - `run_all_checks()`: Runs all validation methods (price consistency, missing trades, etc.).
    - `generate_report()`: Produces a Markdown summary.
    - `visualize_issues()`: Creates bar and pie charts for the summarized issues.

- **`scripts/run_validator.py`**  
  - A command-line script to run the validator:
    1. Takes `--data` (path to your file).
    2. (Optional) `--report` to save a Markdown file.
    3. (Optional) `--fig` to save a chart image.
    4. Prints a summary to console, can show or save visualizations.

---

## How to Run the Validation

1. **Clone** this repo (if you haven’t already):
   ```bash
   git clone https://github.com/yourusername/my-portfolio-validator.git
   cd my-portfolio-validator

2. Create & Activate a Virtual Environment (optional but recommended):

python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
# or .venv\\Scripts\\activate on Windows

3. Install required packages:

pip install -r requirements.txt

4. Run the Validator from the command line:

```bash 
python bash python scripts/run_validator.py --data data/Test.xlsx --report results/report.md --fig results/plots.png python scripts/run_validator.py --data data/Test.xlsx ```
 
- `--data data/Test.xlsx`: path to your Excel/CSV file.
- `--report results/report.md`: saves a Markdown file detailing all flagged issues.
- `--fig results/plots.png`: saves a bar/pie chart visualization of the issues.

If you omit `--report` or `--fig`, the script will print the summary in the console and display any plots interactively.


5. Review the findings:

- Open `results/report.md` to see the detailed list of issues (if you used `--report`).
- Check `results/plots.png` for the distribution of issue types (if you used `--fig`).
- If you see errors that reflect actual data problems, correct your data and run again.

6. Optional: Jupyter Notebook
    - Open `notebooks/portfolio_validation.ipynb` in Jupyter for an interactive exploration and the ability to fix data inline, then re-validate.
