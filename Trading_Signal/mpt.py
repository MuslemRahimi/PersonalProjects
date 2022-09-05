import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import *
import time
import multiprocessing as mp
from tqdm import tqdm
import seaborn
import asyncio


def test_mode():
	df = pd.DataFrame()
	symbol = "EVOK"
	data = get_data(symbol, start_time='2000-1-1', end_time='2018-1-3')
	df[symbol] = data['Close']
	#print(df)
	ind_er = df.resample('Y').last().pct_change().mean()
	ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
	message = "Expected return: %s | std: %s " % (ind_er*100, ann_sd * 100)
	print(message)
	data = get_data(symbol, start_time='2018-1-2', end_time='2018-12-31')
	#print(data['Close'][0])
	print("Actual return:", (data['Close'][-1]/data['Close'][0] -1 ) * 100)

def mpt(symbols_list):
	df = pd.DataFrame()

	for ticker in symbols_list:
		data = get_data(ticker)
		df[ticker] = data["Close"]

	cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
	corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
	print("----------Correlation-Matrix----------")
	print(corr_matrix)
	fig, ax = plt.subplots(figsize=(13, 8))
	seaborn.heatmap(corr_matrix, annot=True, cmap= "RdYlGn")
	fig.savefig("corr_fig.png")

	#resample to get yearly returns if not it gets daily returns
	ind_er = df.resample('Y').last().pct_change().mean()
	#print(ind_er)


	ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
	assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
	assets.columns = ['Returns', 'Volatility']



	p_ret = [] # Define an empty array for portfolio returns
	p_vol = [] # Define an empty array for portfolio volatility
	p_weights = [] # Define an empty array for asset weights

	num_assets = len(df.columns)
	#print(num_assets)

	num_portfolios = int(1e4)

	print("----------Portfolio simulation starts----------")

	def background(f):
	    def wrapped(*args, **kwargs):
	        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

	    return wrapped

	@background
	def parallel_simulate():

		#for portfolio in tqdm(range(num_portfolios)):
		weights = np.random.random(num_assets)

		weights = weights/np.sum(weights)
		p_weights.append(weights)
		returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
		                                      # weights
		p_ret.append(returns)
		var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
		sd = np.sqrt(var) # Daily standard deviation
		ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
		p_vol.append(ann_sd)

	for portfolio in tqdm(range(num_portfolios)):
		parallel_simulate()




	data = {'Returns':p_ret, 'Volatility':p_vol}

	for counter, symbol in enumerate(df.columns.tolist()):
	    #print(counter, symbol)
	    data[symbol+' weight'] = [w[counter] for w in p_weights]


	portfolios  = pd.DataFrame(data)
	#print(portfolios)

	min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
	print("----------Least volatility portfolio----------")
	print(min_vol_port)

	max_return_port = portfolios.iloc[portfolios['Returns'].idxmax()]
	print("----------Max Return portfolio----------")
	print(max_return_port)
	"""
	plt.subplots(figsize=[10,10])
	plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
	plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
	plt.show()
	"""

	# Finding the optimal portfolio
	rf = 0.01 # risk factor
	optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
	print("----------Optimal portfolio----------")
	print(optimal_risky_port)


	fig, ax = plt.subplots()

	ax.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
	plt.xlabel('Volatility')
	plt.ylabel("Return")
	ax.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
	ax.scatter(max_return_port[1], max_return_port[0], color='b', marker='*', s=500)
	ax.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
	fig.savefig("ef_frontier.png")

