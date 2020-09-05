#pip install pandas_datareader
#pip install PyPortfolioOpt
#pip install pulp
# Description: This program attempts to optimize a users portfolio using the Efficient Frontier & Python.
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#----------定義變數---------------
assets =  ['2884.TW', "2885.TW", "2881.TW", "2891.TW", "2887.TW","2886.TW",'00635U.TW','0050.TW']
#assets =  ['2884.TW'"2886.TW",'00635U.TW']
#weights=np.array([0.3333,0.33333,0.333333])
#----------投資比例---------------
# Assign weights to the stocks. Weights must = 1 so 0.2 for each
weights=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])
#----------設定日期---------------
#Get the stock starting date
StockStartDate='2015-04-01'
# Get the stocks ending date aka todays date and format it in the form YYYY-MM-DD
today=datetime.today().strftime('%Y-%m-%d')
#Create a dataframe to store the adjusted close price of the stocks

#----------灌入pd---------------
df=pd.DataFrame()
#Store the adjusted close price of stock into the data frame
for stock in assets:
    df[stock]=web.DataReader(stock,data_source='yahoo',start=StockStartDate,end=today)['Adj Close']
#----------作圖---------------
# Create the title 'Portfolio Adj Close Price History    
title='Profolio Adj. Close Price History'
#Get the stocks
my_stocks=df
#Create and plot the graph
plt.figure(figsize=(18,6)) #width = 12.2in, height = 4.5
# Loop through each stock and plot the Adj Close for each day
for c in my_stocks.columns.values :
    plt.plot(my_stocks[c],label=c)
plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj Price USD($)',fontsize=18)
plt.legend(my_stocks.columns.values, loc='upper left')
plt.show()


#----------算報酬---------------
#Show the daily simple returns, NOTE: Formula = new_price/old_price - 1
returns=df.pct_change()
#returns

cov_matrix_annual=returns.cov()*252
#cov_matrix_annual

#Expected portfolio variance= WT * (Covariance Matrix) * W
port_variance = np.dot(weights.T,np.dot(cov_matrix_annual,weights))
#port_variance

#Expected portfolio volatility= SQRT (WT * (Covariance Matrix) * W)
port_volatility=np.sqrt(port_variance)
#port_volatility
#calculate the portfolio annual simple return.
portfolioSimpleAnnualReturn=np.sum(returns.mean()*weights)*252
#portfolioSimpleAnnualReturn

percent_var=str(round(port_variance,2)*100)+'%'
percent_vols=str(round(port_volatility,2)*100)+'%'
percent_ret=str(round(portfolioSimpleAnnualReturn,2)*100)+'%'
print("Expected annual return : "+ percent_ret)
print('Annual volatility/standard deviation/risk : '+percent_vols)
print('Annual variance : '+percent_var)

#----------計算效率前緣---------------
#Next, I will import the necessary libraries.
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
#Calculate the expected returns and the annualised sample covariance matrix of daily asset returns.
mu = expected_returns.mean_historical_return(df)#returns.mean() * 252
S = risk_models.sample_cov(df) #Get the sample covariance matrix
#Optimize for maximal Sharpe ration .
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe(risk_free_rate=0.008) #Maximize the Sharpe ratio, and get the raw weights
cleaned_weights = ef.clean_weights() 
print(cleaned_weights) #Note the weights may have some rounding error, meaning they may not add up exactly to 1 but should be close
ef.portfolio_performance(verbose=True)




#----------投組分配---------------
#Now it’s time to get the discrete allocation of each stock.
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
weights = cleaned_weights 
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)#一千萬
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))