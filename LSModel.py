#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt

# 1.1 Importing the Dataset and Initializing Variables -----------------------:

df = pd.read_csv("stock_data.csv",
                 index_col = 0, parse_dates = True)

print(df)
'''
            Adj Close
Date                 
2004-06-10   6.116667
2004-06-14   6.066667
2004-06-15   6.250000
              ...
2020-04-21  39.970001
2020-04-22  39.770000
2020-04-23  39.990002

[3995 rows x 1 columns]
'''
# Initial Plot 
plt.plot(df['Adj Close'])
plt.title("CBRE Adj Closing Price from 2004/06/10 - 2020/04/23")
plt.legend(['Price'])
plt.show()

# Initializing Variables
N_long = 252 # Long-term MA time frame - Long-term Look back period
N_short = 42 # Short-term MA time frame - Short-term Look back period 
N = 252 #  Number of business days in a year

# Defining the shape of the data 
rr, cc = df.shape# (3995, 1)


# 1.2 Computing Simple Moving Averages ---------------------------------------:

df['MA_Short'] = df['Adj Close'].rolling(N_short).mean() # 42 day MA
df['MA_Long'] = df['Adj Close'].rolling(N_long).mean() # 252 day MA 

print(df)

'''
            Adj Close   MA_Short    MA_Long
Date                                       
2004-06-10   6.116667        NaN        NaN
2004-06-14   6.066667        NaN        NaN
2004-06-15   6.250000        NaN        NaN
2004-06-16   6.333333        NaN        NaN
2004-06-17   6.350000        NaN        NaN
              ...        ...        ...
2020-04-17  44.500000  46.336428  52.790437
2020-04-20  41.959999  45.825952  52.755635
2020-04-21  39.970001  45.260952  52.713810
2020-04-22  39.770000  44.714286  52.667302
2020-04-23  39.990002  44.236190  52.620635

[3995 rows x 3 columns]
'''

# Note: There are a lot of NaN values at the start of the MA_Short and MA_Long
# columns, because you would need at least 41 values of data to calculate 
# a 42 day moving average, for example. For now, we will let these NaN values
# be, as they will be removed at a later point (d)

# 1.3 Computing Exponentially-Weighted Moving Averages -----------------------:

df['EMA_Short'] = df['Adj Close'].ewm(span = N_short, adjust=False).mean()
df['EMA_Long'] = df['Adj Close'].ewm(span = N_long, adjust=False).mean()

print(df)

'''
            Adj Close   MA_Short    MA_Long  EMA_Short   EMA_Long
Date                                                             
2004-06-10   6.116667        NaN        NaN   6.116667   6.116667
2004-06-14   6.066667        NaN        NaN   6.114341   6.116272
2004-06-15   6.250000        NaN        NaN   6.120651   6.117329
2004-06-16   6.333333        NaN        NaN   6.130543   6.119036
2004-06-17   6.350000        NaN        NaN   6.140751   6.120862
              ...        ...        ...        ...        ...
2020-04-17  44.500000  46.336428  52.790437  45.676528  51.691902
2020-04-20  41.959999  45.825952  52.755635  45.503666  51.614970
2020-04-21  39.970001  45.260952  52.713810  45.246287  51.522915
2020-04-22  39.770000  44.714286  52.667302  44.991576  51.430007
2020-04-23  39.990002  44.236190  52.620635  44.758944  51.339572

[3995 rows x 5 columns]
'''

# 1.4 Dropping Missing Values ------------------------------------------------:

df.dropna(inplace = True)
print(df.head(3))
      
'''
            Adj Close   MA_Short   MA_Long  EMA_Short   EMA_Long
Date                                                            
2005-06-09  13.303333  11.915476  9.500767  12.190587  10.031323
2005-06-10  12.983334  11.946905  9.528016  12.227459  10.054660
2005-06-13  12.966666  11.991190  9.555397  12.261841  10.077679
                            .........
'''

# 1.5 Plotting SMAs and EMAs -------------------------------------------------:

plt.figure(figsize = (10, 8))
plt.plot(df)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('CBRE Price Data and Moving Averages')
plt.legend(['Adj Close', 'MA_Short', 'MA_Long', 'EMA_Short', 'EMA_Long'])
plt.show()

# If you look at the plot, where the price is the blue line, the Long SMA
# is the green line and the long EMA is the purple line, you can see that the
# purple line almost always moves quicker than the green line in response to 
# price fluctuations - thus, the exponential moving average (EMA_Long) is 
# quicker to repond to price changes than the simple moving average (SMA_Long)

# 1.6 Finding the Positions based on EMA Rules -------------------------------:

df['Position'] = 1
df['Position'].where(cond = df['EMA_Short'] > df['EMA_Long']
                                       , other = -1, inplace = True)

# Because we can only make a trade the day after we obtain the signal from 
# our EMA strategy (at market close), we need to introduce a lag to our 
# position variable, and shift it 1 day forward throughout. This can be 
# accomplshed by using the .shift(1) function available in pandas. 

df['Position'] = df['Position'].shift(1)

print(df)

'''
            Adj Close   MA_Short    MA_Long  EMA_Short   EMA_Long  Position
Date                                                                       
2005-06-09  13.303333  11.915476   9.500767  12.190587  10.031323       NaN
2005-06-10  12.983334  11.946905   9.528016  12.227459  10.054660       1.0
2005-06-13  12.966666  11.991190   9.555397  12.261841  10.077679       1.0
2005-06-14  12.906667  12.030556   9.581812  12.291833  10.100043       1.0
2005-06-15  13.036667  12.081111   9.608413  12.326476  10.123257       1.0
              ...        ...        ...        ...        ...       ...
2020-04-17  44.500000  46.336428  52.790437  45.676528  51.691902      -1.0
2020-04-20  41.959999  45.825952  52.755635  45.503666  51.614970      -1.0
2020-04-21  39.970001  45.260952  52.713810  45.246287  51.522915      -1.0
2020-04-22  39.770000  44.714286  52.667302  44.991576  51.430007      -1.0
2020-04-23  39.990002  44.236190  52.620635  44.758944  51.339572      -1.0

[3744 rows x 6 columns]
'''

# 1.7 Plotting the Positions on a Secondary Axis -----------------------------:

plt.plot(df)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('CBRE Price, Moving Averages, and EMA Positions')
plt.legend(['Adj Close', 'MA_Short', 'MA_Long', 
            'EMA_Short', 'EMA_Long', 'Position'])
df.plot(figsize = (10, 8), secondary_y = 'Position')
plt.show()

# Here, the brown line represents the position to be taken. As you can see, 
# whenever the short-term EMA (red line) is higher than the long-term EMA
# (purple line), you should go long (brown line = + 1), and whenever the short
# -term EMA (red line) is lower than the long-term EMA (purple line), you
# should go short (brown line = -1) - this is primarily a momentum trade, and 
# whenever there is a crossover, you should change your strategy.    


# 2. BACKTESTING THE STRATEGY --------------------------------------------------:
    
# Assumption: The transaction cost is incurred in the buy and hold strategy 
# as well, at the first period, when you do your buy. This is a reasonable 
# assumption and the only way to fairly compare the 2 strategies, as it can't
# be costless for one (where you do indeed buy at one point) and have a cost 
# for the other. 
    
# 2.1 Calculating the returns of the Benchmark (buy and hold strategy)-------:

tcost = 0.005 # 0.5% transaction cost
dfbench = df["Adj Close"].copy()
dfbench[0] = dfbench[0] * (1 + tcost) # Increasing the buying price in the
# first period, to now include the transaction cost too
print(dfbench)
    
'''
Date
2005-06-09    13.369850
2005-06-10    12.983334
2005-06-13    12.966666
2005-06-14    12.906667
2005-06-15    13.036667
   
2020-04-17    44.500000
2020-04-20    41.959999
2020-04-21    39.970001
2020-04-22    39.770000
2020-04-23    39.990002
'''

Final = dfbench
Initial = dfbench.shift(1) # One period prior (t-1)
df['Benchmark'] = (Final - Initial) / Initial

print(df['Benchmark'])

'''
Date
2005-06-09         NaN
2005-06-10   -0.028909
2005-06-13   -0.001284
2005-06-14   -0.004627
2005-06-15    0.010072
  
2020-04-17    0.046074
2020-04-20   -0.057079
2020-04-21   -0.047426
2020-04-22   -0.005004
2020-04-23    0.005532
'''

# Formally putting this, what we are doing, is INCREASING the price by 
# (1 + transaction percentage) at every point of buying. Now, if we were to 
# generalize this, we know that when we buy, the INITAL PRICE (which is 
# your buying price), which is the previous period's price, i.e., the 
# adj price. shift(1), should rise, and then the final price, which is 
# this period's price, should stay the same. 

# Now, when you go short, then, you sell the stock at the price today, but 
# first thing in the morning - thus, at yesterdays closing price. 
# So the price in t-1 will be reduced as we deduct the transaction cost from 
# it. Thus, we will REDUCE the INITIAL PRICE  by (1 - transaction percentage)

# 2.2 Calculating the returns of the Strategy --------------------------------:


dfstrategy = df[["Adj Close", "Position"]].copy()

dfstrategy["Diff"] = dfstrategy["Position"]-dfstrategy["Position"].shift()
# At every point where Diff = 2, a shift in strategy has occurred 

# If diff yields a 2.0, it means 1--1, the strategy has changed from short 
# to long - thus increase INITIAL price by (1 + transaction percentage)

# # If diff yields a -2.0, it means -1-1, the strategy has changed from long 
# to short - thus reduce INITIAL price by (1-transaction percentage)

# A key thing to note here, is that if you're going long, a larger absolute 
# value of the return is obviously good. 
# But if you're going short, a larger absolute value of the return is bad.
# This is because it will be multiplied by -1 as that is our position. 

# So, mathematically speaking, in the return = (final-initial/initial)
# calculation, it makes sense that the initial rises (and thus the return 
# falls) when you switch to going long and incurring a transaction cost. 

# It also makes sense that the initial falls (although this initially seems
# counter-intuitive) when you switch to going short and incur a transaction 
# cost, as although the abolsute value of the return will rise, since that 
# is multipied by -1 in our strategy, this will actually reduce the strategy
# return. 

# Now, to illustrate this and aid future work, lets try and find a long and
# short position to test if our loops work later on. 

print(dfstrategy.tail(25)) # On 2020-03-23, we change from a long to a short 
# Thus, we have a Diff value of -2. 
'''
            Adj Close  Position  Diff
Date                                 
2020-03-19  39.439999       1.0   0.0
2020-03-20  34.259998       1.0   0.0
2020-03-23  29.680850      -1.0  -2.0
2020-03-24  33.860001      -1.0   0.0
.....................................
2020-04-20  41.959999      -1.0   0.0
2020-04-21  39.970001      -1.0   0.0
2020-04-22  39.770000      -1.0   0.0
2020-04-23  39.990002      -1.0   0.0
'''

print(dfstrategy.iloc[2900:2915]) # On 2016-12-16, we change from a short to 
# a long. Thus we have a Diff value of 2. 

'''
            Adj Close  Position  Diff
Date                                 
2016-12-14  31.820000      -1.0   0.0
2016-12-15  31.760000      -1.0   0.0
2016-12-16  31.760000       1.0   2.0
2016-12-19  31.799999       1.0   0.0
2016-12-20  32.070000       1.0   0.0
.....................................
2016-12-30  31.490000       1.0   0.0
2017-01-03  31.570000       1.0   0.0
2017-01-04  32.080002       1.0   0.0
2017-01-05  31.809999       1.0   0.0
'''

# Now one thing we want to avoid doing is simply changing the closing prices
# in the dataframe itself (which was what I initially did, until I thought 
# about it a bit more),  Because then itll affect the previous periods return 
# too - where a transaction cost is not incurred. Instead, we will run a 
# for loop to periodically calculate returns. 

df["SReturns"] = np.nan # Creating an empty strategy returns column in the 
# dataframe to later populate with returns (not taking position into account)

length = len(dfstrategy["Diff"])

for ii in range(1, length):
    Final = dfstrategy["Adj Close"][ii]
    Initial = dfstrategy["Adj Close"][ii-1] # One period prior (t-1)
    if dfstrategy["Diff"][ii] == 0.0: # No change in strategy
        df['SReturns'][ii] = (Final - Initial) / Initial
    elif dfstrategy["Diff"][ii] == 2.0: # changed to long
        df['SReturns'][ii] = (Final - (Initial*(1+tcost)))/(Initial*(1+tcost))
        # The return will reduce due to tranaction costs raising buying price
    elif dfstrategy["Diff"][ii] == -2.0: # changed to short
        df['SReturns'][ii] = (Final - (Initial*(1-tcost)))/(Initial*(1-tcost))
        # The return will rise, but remember, this is bad when you go short (-)
        # This will be clear later, when we take positions into account
    else: # When NaN, such as the second row, because of the Diff calculation
        df['SReturns'][ii] = (Final - Initial) / Initial

# Now, lets see if everything worked. (Spot Checks) 

print(df.tail(25)) # On 2020-03-23, we change from a long to a short 
# Here, we see that the SReturns are slifghtly lower than that of the Benchmark
# returns, as it should be, as we are paying transaction costs. 
'''
            Adj Close   MA_Short    MA_Long  ...  Position  Benchmark  SReturns
Date                                         ...                               
2020-03-19  39.439999  57.278095  53.638492  ...       1.0   0.110360  0.110360
2020-03-20  34.259998  56.639048  53.577143  ...       1.0  -0.131339 -0.131339
2020-03-23  29.830000  55.899048  53.504048  ...      -1.0  -0.129305 -0.124930
2020-03-24  33.860001  55.256667  53.446786  ...      -1.0   0.135099  0.135099
...............................................................................
2020-04-21  39.970001  45.260952  52.713810  ...      -1.0  -0.047426 -0.047426
2020-04-22  39.770000  44.714286  52.667302  ...      -1.0  -0.005004 -0.005004
2020-04-23  39.990002  44.236190  52.620635  ...      -1.0   0.005532  0.005532
'''

print(df.iloc[2900:2915]) # On 2016-12-16, we change from a short to 
# a long. Here, we see that the SReturns are again slightly lower than the 
# Benchmark, which is exactly what we want, as we are paying transaction costs.  

'''
            Adj Close   MA_Short    MA_Long  ...  Position  Benchmark  SReturns
Date                                         ...                               
2016-12-14  31.820000  28.631667  28.706587  ...      -1.0  -0.017295 -0.017295
2016-12-15  31.760000  28.759286  28.688651  ...      -1.0  -0.001886 -0.001886
2016-12-16  31.760000  28.880952  28.674286  ...       1.0   0.000000 -0.004975
2016-12-19  31.799999  28.978572  28.665754  ...       1.0   0.001259  0.001259
...............................................................................
2017-01-03  31.570000  29.953810  28.560873  ...       1.0   0.002540  0.002540
2017-01-04  32.080002  30.106429  28.551706  ...       1.0   0.016155  0.016155
2017-01-05  31.809999  30.243333  28.547341  ...       1.0  -0.008417 -0.008417
'''  

# Now, we will adjust for the positions in our Strategy. 

df['Strategy'] = df['SReturns'] * df['Position']

print(df)

'''
            Adj Close   MA_Short    MA_Long  ...  Benchmark  SReturns  Strategy
Date                                         ...                               
2005-06-09  13.303333  11.915476   9.500767  ...        NaN       NaN       NaN
2005-06-10  12.983334  11.946905   9.528016  ...  -0.028909 -0.024054 -0.024054
2005-06-13  12.966666  11.991190   9.555397  ...  -0.001284 -0.001284 -0.001284
2005-06-14  12.906667  12.030556   9.581812  ...  -0.004627 -0.004627 -0.004627
2005-06-15  13.036667  12.081111   9.608413  ...   0.010072  0.010072  0.010072
              ...        ...        ...  ...        ...       ...       ...
2020-04-17  44.500000  46.336428  52.790437  ...   0.046074  0.046074 -0.046074
2020-04-20  41.959999  45.825952  52.755635  ...  -0.057079 -0.057079  0.057079
2020-04-21  39.970001  45.260952  52.713810  ...  -0.047426 -0.047426  0.047426
2020-04-22  39.770000  44.714286  52.667302  ...  -0.005004 -0.005004  0.005004
2020-04-23  39.990002  44.236190  52.620635  ...   0.005532  0.005532 -0.005532

[3744 rows x 9 columns]
'''


# But there is also a transaction cost at the start, when you buy, even for 
# the strategy - So add that in. By equating the benchmark return for that 
# date only. 

df["SReturns"][1] = df["Benchmark"][1]
df['Strategy'] = df['SReturns'] * df['Position']

print(df)

'''
            Adj Close   MA_Short    MA_Long  ...  Benchmark  SReturns  Strategy
Date                                         ...                               
2005-06-09  13.303333  11.915476   9.500767  ...        NaN       NaN       NaN
2005-06-10  12.983334  11.946905   9.528016  ...  -0.028909 -0.028909 -0.028909
2005-06-13  12.966666  11.991190   9.555397  ...  -0.001284 -0.001284 -0.001284
2005-06-14  12.906667  12.030556   9.581812  ...  -0.004627 -0.004627 -0.004627
2005-06-15  13.036667  12.081111   9.608413  ...   0.010072  0.010072  0.010072
              ...        ...        ...  ...        ...       ...       ...
2020-04-17  44.500000  46.336428  52.790437  ...   0.046074  0.046074 -0.046074
2020-04-20  41.959999  45.825952  52.755635  ...  -0.057079 -0.057079  0.057079
2020-04-21  39.970001  45.260952  52.713810  ...  -0.047426 -0.047426  0.047426
2020-04-22  39.770000  44.714286  52.667302  ...  -0.005004 -0.005004  0.005004
2020-04-23  39.990002  44.236190  52.620635  ...   0.005532  0.005532 -0.005532

[3744 rows x 9 columns]
'''

# 2.3 Dropping Nans -----------------------------------------------------------:

df.dropna(inplace = True)
print(df)

'''
            Adj Close   MA_Short    MA_Long  ...  Benchmark  SReturns  Strategy
Date                                         ...                               
2005-06-10  12.983334  11.946905   9.528016  ...  -0.028909 -0.028909 -0.028909
2005-06-13  12.966666  11.991190   9.555397  ...  -0.001284 -0.001284 -0.001284
2005-06-14  12.906667  12.030556   9.581812  ...  -0.004627 -0.004627 -0.004627
2005-06-15  13.036667  12.081111   9.608413  ...   0.010072  0.010072  0.010072
2005-06-16  13.073334  12.133254   9.635093  ...   0.002813  0.002813  0.002813
              ...        ...        ...  ...        ...       ...       ...
2020-04-17  44.500000  46.336428  52.790437  ...   0.046074  0.046074 -0.046074
2020-04-20  41.959999  45.825952  52.755635  ...  -0.057079 -0.057079  0.057079
2020-04-21  39.970001  45.260952  52.713810  ...  -0.047426 -0.047426  0.047426
2020-04-22  39.770000  44.714286  52.667302  ...  -0.005004 -0.005004  0.005004
2020-04-23  39.990002  44.236190  52.620635  ...   0.005532  0.005532 -0.005532

[3743 rows x 9 columns]
'''

# 2.4 Computing the Average Returns for the Benchmark and for the Strategy ---:

benchravg = df["Benchmark"].mean() # 0.0009386688493611099
strategyravg = df["Strategy"].mean() # 7.864768106503228e-06 (A very small, 
# but positive return)

# Since the next question asks for annualized volatility, as well as the fact 
# that the Sharpe Ratio usually considers the annualized returns, we will 
# also calculate annualized returns (for future usage). Fpr this, we simply
# just multiply by the number of trading days in a year - i.e., 252. 

benchravg_ann = benchravg * 252 # 0.23654455003899968
strategyravg_ann = strategyravg * 252 # 0.0019819215628388135 


# 2.5 Computing Volatility and Annualized Volatility -------------------------:

benchrvol = df["Benchmark"].std() # 0.0366037417022739
strategyrvol = df["Strategy"].std() # 0.03660948379458992 (Very close)

# And Annualised Volatility = Daily Volatility * (252)^(1/2)

benchrvol_ann = benchrvol * (252 ** 0.5) # 0.5810663855919649
strategyrvol_ann = strategyrvol * (252 ** 0.5) # 0.581157538481605

# 2.6 Sharpe Ratios ----------------------------------------------------------:

rf = 0.01

# Sharpe Ratios make the most sense when using annualized returns and 
# volatility. But let's illustrate this first. 

# First, using Daily Returns

# Benchmark SR: 
benchSR = (benchravg - rf) / (benchrvol) # -0.2475520460269225

# Strategy SR: 
strategySR = (strategyravg - rf) / (strategyrvol) # -0.27293843551462793

# As you can see, the SRs come out to be negative, and really bad, despite us 
# actually making a positive return overall, on average. This is because the 
# risk-free rate deduction makes a lot more sense (numerator) when looking 
# at annualized parameters. 

# Thus let's use the Annualized Returns and Volatility to Calculate the 
# Annualized Sharpe Ratio: 
    
benchSR_ann = (benchravg_ann - rf) / (benchrvol_ann) # 0.38987722514391543

strategySR_ann = (strategyravg_ann - rf) / (strategyrvol_ann) 
# -0.01379673824434952

# Thus, while buying and holding would have done well, our stratey has
# performed badly, on a risk-adjusted basis, and thus, is not a great 
# strategy, when backtested. 

# 2.7 Computing Cumulative Returns -------------------------------------------:

# We will achieve this using .cumprod()

benchr_cum = ((df["Benchmark"] + 1).cumprod() - 1)[-1] # 1.9910584637826587
# Final value of the series of numbers cumulatively multiplied and calculated
# This represents a 199% return over the period for this stock (non-dividend 
# paying), so this will be the total return = absolute return (dividend = 0). 

strategyr_cum = ((df["Strategy"] + 1).cumprod() - 1)[-1] # -0.9371708218142443
# So you would have losr 93% of your money over the period - not a great 
# strategy at all. 

# 2.8 Reporting the results --------------------------------------------------:

'''
Benchmark average return: 0.0009386688493611099
Strategy average return: 7.864768106503228e-06
Benchmark SR: 0.38987722514391543
Strategy SR: -0.01379673824434952
Benchmark cumulative return: 1.9910584637826587
Strategy cumulative return: -0.9371708218142443
'''

# In general we see that the effect of Transaction costs has reduced the 
# returns of the benchmark and thus, its Sharpe Ratio, slightly, but the 
# returns of the Strategy have been severely affected negatively due to the 
# introduction of transaction costs, as many trades were made throughout the 
# course of the sample period.  


#%%
