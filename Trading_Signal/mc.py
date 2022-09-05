import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import *
import time
from tqdm import tqdm
import seaborn
from scipy.stats import norm

def data_loader(symbols_list, start_time, end_time):
	df = pd.DataFrame()

	for ticker in symbols_list:
		data = get_data(ticker,'1d', start_time, end_time)
		df[ticker] = data["Close"]
	return df


def log_returns(data):
    return (np.log(1+data.pct_change())).dropna()

def simple_returns(data):
    return ((data/data.shift(1))-1)

def drift_calc(data, return_type='log'):
    if return_type=='log':
        lr = log_returns(data)
    elif return_type=='simple':
        lr = simple_returns(data)
    u = lr.mean()
    var = lr.var()
    drift = u-(0.5*var)
    try:
        return drift.values
    except:
        return drift

def daily_returns(data, days, iterations, return_type='log'):
    ft = drift_calc(data, return_type)

    if return_type == 'log':
    	stv = log_returns(data).std()
    elif return_type=='simple':
    	stv = simple_returns(data).std()
    #Oftentimes, we find that the distribution of returns is a variation of the normal distribution where it has a fat tail
    # This distribution is called cauchy distribution
    dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))
    
    return dr


def beta_sharpe(df,riskfree = 0.025):
    # Beta
    #dd, mark_ret = market_data_combination(df, mark_ticker = mark_ticker)
    mark_ticker = '^GSPC'
    lr  = log_returns(df)
    mark_ret = lr.iloc[:,-1:]
    mark_ret = np.exp(mark_ret.mean()*252) - 1

    covar = lr.cov()*252  # Annualized
    covar = pd.DataFrame(covar.iloc[:-1,-1])
    mark_var = lr.iloc[:,-1].var()*252 #Annualized of market
    beta = covar/mark_var

    stdev_ret = pd.DataFrame(((lr.std()*250**0.5)[:-1]), columns=['STD'])
    beta = beta.merge(stdev_ret, left_index=True, right_index=True)

    # CAPM
    for i, row in beta.iterrows():
        beta.at[i,'CAPM'] = riskfree + (row[mark_ticker] * float((mark_ret-riskfree) ) )


    # Sharpe
    for i, row in beta.iterrows():
        beta.at[i,'Sharpe'] = ((row['CAPM']-riskfree)/(row['STD']))
    beta.rename(columns={"^GSPC":"Beta"}, inplace=True)
    #print(beta)
    #time.sleep(100)
    return beta

def probs_find(predicted, higherthan, ticker = None, on = 'value'):
    """
    This function calculated the probability of a stock being above a certain threshhold, which can be defined as a value (final stock price) or return rate (percentage change)
    Input:
    1. predicted: dataframe with all the predicted prices (days and simulations)
    2. higherthan: specified threshhold to which compute the probability (ex. 0 on return will compute the probability of at least breakeven)
    3. on: 'return' or 'value', the return of the stock or the final value of stock for every simulation over the time specified
    4. ticker: specific ticker to compute probability for
    """
    if ticker == None:
        if on == 'return':
            predicted0 = predicted.iloc[0,0]
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
            less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
        elif on == 'value':
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [i for i in predList if i >= higherthan]
            less = [i for i in predList if i < higherthan]
        else:
            print("'on' must be either value or return")
    else:
        if on == 'return':
            predicted = predicted[predicted['ticker'] == ticker]
            predicted0 = predicted.iloc[0,0]
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
            less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
        elif on == 'value':
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [i for i in predList if i >= higherthan]
            less = [i for i in predList if i < higherthan]
        else:
            print("'on' must be either value or return")
    return (len(over)/(len(over)+len(less)))



def simulate_mc(df, days = 250, iterations = int(1e4), start_time = '2010-1-1', \
				end_time = date.today(), return_type='log', plot=False):
    # Generate daily returns
    returns = daily_returns(df, days, iterations, return_type)

    # Create empty matrix
    price_list = np.zeros_like(returns)
    # Put the last actual price in the first row of matrix. 
    price_list[0] = df.iloc[-1]


    # Calculate the price of each day
    for t in range(1,days):
        price_list[t] = price_list[t-1]*returns[t]

    # Plot Option
    if plot == True:
        x = pd.DataFrame(price_list).iloc[-1]
        fig, ax = plt.subplots(1,2, figsize=(14,4))
        seaborn.distplot(x, ax=ax[0])
        seaborn.distplot(x, hist_kws={'cumulative':True},kde_kws={'cumulative':True},ax=ax[1])
        plt.xlabel("Stock Price")
        plt.show()

    #CAPM and Sharpe Ratio

    # Printing information about stock
    try:
        [print(nam) for nam in df.columns]
    except:
        print(df.name)
    
    #print(pd.DataFrame(price_list))
    #time.sleep(100)
    print(f"Days: {days-1}")

    # Take last value of each iteration and make a mean to compute the expected value
    print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(),2)}")
    # Take the last actual value and compute the expected return
    print(f"Return: {round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)}%")
    print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list),0, on='return')}")


    return pd.DataFrame(price_list)

def monte_carlo(symbols_list, days, iterations, start_time = '2010-1-1', end_time=date.today(), \
				return_type = 'log', plotten=False):
    df = data_loader(symbols_list, start_time, end_time)
    inform = beta_sharpe(df)
    simulatedDF = []


    for t in tqdm(range(len(symbols_list))):
    	y = simulate_mc(df.iloc[:,t], (days+1), iterations, return_type)
    	
    	if plotten == True:
    		forplot = y.iloc[:,0:10]
    		forplot.plot(figsize=(15,4))

    	#Skip beta, sharpe, CAPM return of market
    	if t == len(symbols_list)-1:
    		pass

    	else:
    		print(f"Beta: {round(inform.iloc[t,inform.columns.get_loc('Beta')],2)}")
    		print(f"Sharpe: {round(inform.iloc[t,inform.columns.get_loc('Sharpe')],2)}")
	    	print(f"CAPM Return: {round(100*inform.iloc[t,inform.columns.get_loc('CAPM')],2)}%")
	    	y['ticker'] = symbols_list[t]
	    	cols = y.columns.tolist()
	    	cols = cols[-1:] + cols[:-1]
	    	y = y[cols]
	    	simulatedDF.append(y)
	    	print("--------------------")
	    	
    #simulatedDF = pd.concat(simulatedDF)
    #print(simulatedDF)
    return simulatedDF

