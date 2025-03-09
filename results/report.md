# Portfolio Validation Report

## Summary

| Issue Type | Count |
| --- | --- |
| Price Consistency | 139 |
| Missing Trades | 38 |
| Calculation Errors | 6817 |
| Trade Price Inconsistencies | 31 |
| Weight Consistency | 516 |
| Cash Flow Consistency | 260 |
| Currency Issues | 10 |
| Negative Prices Or Rates | 0 |
| **Total Issues** | **7811** |

## Price Consistency

Found 139 issues.

### Examples

**Issue 1**

- Ticker: ADBE
- Date: 2022-10-13 00:00:00
- Price Yesterday: 286.1499938964844
- Previous Day Price: 554.0800170898438
- Difference: -267.9300231933594
- Difference %: -48.35583578714673

**Issue 2**

- Ticker: AMC
- Date: 2022-05-11 00:00:00
- Price Yesterday: 11.84500026702881
- Previous Day Price: 22.9950008392334
- Difference: -11.150000572204588
- Difference %: -48.488802632182484

**Issue 3**

- Ticker: AMC
- Date: 2022-06-13 00:00:00
- Price Yesterday: 12.4350004196167
- Previous Day Price: 14.42500019073486
- Difference: -1.9899997711181605
- Difference %: -13.795492165028408

**Issue 4**

- Ticker: AMC
- Date: 2022-08-09 00:00:00
- Price Yesterday: 23.95999908447266
- Previous Day Price: 12.76000022888184
- Difference: 11.19999885559082
- Difference %: 87.77428412767574

**Issue 5**

- Ticker: AMR
- Date: 2022-10-11 00:00:00
- Price Yesterday: 151.4600067138672
- Previous Day Price: 128.4600067138672
- Difference: 23.0
- Difference %: 17.904405105030374

## Missing Trades

Found 38 issues.

### Examples

**Issue 1**

- Ticker: AMC
- Date: 2022-08-23 00:00:00
- Open Quantity: 0
- Previous Day Close Quantity: -983
- Difference: 983

**Issue 2**

- Ticker: AMC
- Date: 2022-08-23 00:00:00
- Open Quantity: -983
- Previous Day Close Quantity: -1310
- Difference: 327

**Issue 3**

- Ticker: AMC
- Date: 2022-08-24 00:00:00
- Open Quantity: -1310
- Previous Day Close Quantity: -1964
- Difference: 654

**Issue 4**

- Ticker: AMC
- Date: 2022-08-24 00:00:00
- Open Quantity: -1964
- Previous Day Close Quantity: 0
- Difference: -1964

**Issue 5**

- Ticker: AMC
- Date: 2022-08-25 00:00:00
- Open Quantity: 0
- Previous Day Close Quantity: -1964
- Difference: 1964

## Calculation Errors

Found 6817 issues.

### Examples

**Issue 1**

- Type: Value in USD
- Ticker: RHM
- Date: 2022-03-04 00:00:00
- Actual: 121400.2212372896
- Expected: -94936.95775232186
- Difference: 216337.17898961145
- Difference %: 227.87456445993027

**Issue 2**

- Type: Value in USD
- Ticker: RHM
- Date: 2022-03-07 00:00:00
- Actual: 126674.9550448656
- Expected: -99061.88582527639
- Difference: 225736.84087014198
- Difference %: 227.87456445993027

**Issue 3**

- Type: Value in USD
- Ticker: PFE
- Date: 2022-03-14 00:00:00
- Actual: 25891.66939544678
- Expected: 28860.34
- Difference: -2968.670604553219
- Difference %: 10.28633274782355

**Issue 4**

- Type: Value in USD
- Ticker: 2GB
- Date: 2022-05-17 00:00:00
- Actual: 5879.582552230222
- Expected: 23518.330208920892
- Difference: -17638.74765669067
- Difference %: 75.0

**Issue 5**

- Type: Value in USD
- Ticker: 2GB
- Date: 2022-05-18 00:00:00
- Actual: 5875.739919662476
- Expected: 23502.9596786499
- Difference: -17627.219758987423
- Difference %: 75.0

## Trade Price Inconsistencies

Found 31 issues.

### Examples

**Issue 1**

- Ticker: RSX
- Date: 2022-03-02 00:00:00
- Trade Price: 8.13
- Holding Price: 7.190000057220459
- Difference: 0.9399999427795418
- Difference %: 13.073712591080715

**Issue 2**

- Ticker: RIVN
- Date: 2022-03-07 00:00:00
- Trade Price: 48.0
- Holding Price: 42.43000030517578
- Difference: 5.569999694824219
- Difference %: 13.127503310776003

**Issue 3**

- Ticker: CLB_US
- Date: 2022-03-08 00:00:00
- Trade Price: 31.40875638723373
- Holding Price: 35.04000091552734
- Difference: -3.6312445282936068
- Difference %: 10.363140506324836

**Issue 4**

- Ticker: HSBK
- Date: 2022-03-09 00:00:00
- Trade Price: 8.5
- Holding Price: 9.444999694824219
- Difference: -0.9449996948242188
- Difference %: 10.005290898443022

**Issue 5**

- Ticker: KAP
- Date: 2022-03-09 00:00:00
- Trade Price: 32.841136819
- Holding Price: 26.54999923706055
- Difference: 6.291137581939449
- Difference %: 23.695434134543365

## Weight Consistency

Found 516 issues.

### Examples

**Issue 1**

- Date: 2022-01-03 00:00:00
- Type: Opening Weights
- Sum: 0.44913374592088723
- Expected: 100
- Difference: -99.55086625407911

**Issue 2**

- Date: 2022-01-03 00:00:00
- Type: Closing Weights
- Sum: 0.42867082130081924
- Expected: 100
- Difference: -99.57132917869919

**Issue 3**

- Date: 2022-01-04 00:00:00
- Type: Opening Weights
- Sum: 0.42845569467488703
- Expected: 100
- Difference: -99.57154430532512

**Issue 4**

- Date: 2022-01-04 00:00:00
- Type: Closing Weights
- Sum: 0.24950793939351387
- Expected: 100
- Difference: -99.75049206060649

**Issue 5**

- Date: 2022-01-05 00:00:00
- Type: Opening Weights
- Sum: 0.2499141363851887
- Expected: 100
- Difference: -99.75008586361481

## Cash Flow Consistency

Found 260 issues.

### Examples

**Issue 1**

- Date: 2022-01-04 00:00:00
- NAV Yesterday: 929490.4375
- Previous Day NAV: 929315.622804415
- Difference: 174.8146955850534

**Issue 2**

- Date: 2022-01-05 00:00:00
- NAV Yesterday: 918195.25
- Previous Day NAV: 919771.2149885213
- Difference: -1575.9649885213003

**Issue 3**

- Date: 2022-01-06 00:00:00
- NAV Yesterday: 911905.1875
- Previous Day NAV: 913007.459533798
- Difference: -1102.2720337980427

**Issue 4**

- Date: 2022-01-07 00:00:00
- NAV Yesterday: 911075.75
- Previous Day NAV: 912573.7128464646
- Difference: -1497.962846464594

**Issue 5**

- Date: 2022-01-10 00:00:00
- NAV Yesterday: 908683.5625
- Previous Day NAV: 909766.3288944616
- Difference: -1082.7663944616215

## Currency Issues

Found 10 issues.

### Examples

**Issue 1**

- Ticker: CON
- Date: 2022-08-29 00:00:00
- Currency: EURUSD
- Exchange Rate: 0.9998000264167786
- Issue: Non-USD currency with exchange rate of 1.0

**Issue 2**

- Ticker: EXS1
- Date: 2022-08-29 00:00:00
- Currency: EURUSD
- Exchange Rate: 0.9998000264167786
- Issue: Non-USD currency with exchange rate of 1.0

**Issue 3**

- Ticker: HLAG
- Date: 2022-08-29 00:00:00
- Currency: EURUSD
- Exchange Rate: 0.9998000264167786
- Issue: Non-USD currency with exchange rate of 1.0

**Issue 4**

- Ticker: LDO_IT
- Date: 2022-08-29 00:00:00
- Currency: EURUSD
- Exchange Rate: 0.9998000264167786
- Issue: Non-USD currency with exchange rate of 1.0

**Issue 5**

- Ticker: MT
- Date: 2022-08-29 00:00:00
- Currency: EURUSD
- Exchange Rate: 0.9998000264167786
- Issue: Non-USD currency with exchange rate of 1.0

