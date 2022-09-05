from data import *
from datetime import date
from termcolor import colored
import pandas as pd 


# Todo List
# Add metric function to measure performance of strategy
# Add order function for buy and sell
# Add portfolio
# Add summary for all important information as output
'''
INITIAL_BUDGET = 1000

current_budget= INITIAL_BUDGET

today = date.today()
data = get_data("AMD", start_time='2010-1-1', end_time = today)
data= round(data,2)

calendar= pd.Series(data.index).iloc[1:]


print("Initial budget:", initial_budget)

shares= int(initial_budget/data["Close"][1])

current_budget -=  data["Close"][1]*shares

print("current budget", round(current_budget,2))

print("Entry price: %s | Current price: %s | Number of shares: %s" %(data["Close"][1],data["Close"][-1], shares) )
print("Return: %s %%" % (round(data["Close"][-1]/data["Close"][1]*100,2)) )

'''

def orderBuy(current_budget,data):

	shares = int(current_budget/data[1])
	current_budget -=  data[1]*shares
	print("Entry price: %s | Shares bought: %s" % (data[1], shares) )
	return current_budget, shares


def orderSell(current_budget, data, shares, weight):
	# para: weight how many shares should be sold

	current_budget +=  data[-1]*shares*weight
	print("Exit price: %s | Shares sold: %s" % (data[-1], round(shares*weight,2)) )
	print("ROI: %s %%" % (round(data[-1]/data[1]*100,2)) )
	shares = int(shares*(1-weight))
	return current_budget, shares

def portfolio(initial_budget, current_budget, data, shares):
	print("Initial Budget:", round(initial_budget,2))
	print("Current budget:", round(current_budget,2))
	print("Holding shares:", shares)
	print("Asset value:", round(data[-1]*shares,2))
	print("Total budget:", round(current_budget,2) + round(data[-1]*shares,2) )

def daily_pct_change(data):
	return data/data.shift(1) -1

def SP500(symbol_roi, start_time,end_time):
	# Benchmark test to see if stock outperforms the market
	raw_data = round(get_data("^GSPC",start_time=start_time, end_time=end_time),2)
	data= raw_data["Close"]
	sp500_roi= (round(data[-1]/data[1]*100,2))

	if symbol_roi >= sp500_roi:
		performance = colored('Outperforms the market, ROI: ','green')
	else:
		performance = colored('Underperforms the market','red')
	return sp500_roi, performance

def main():
	# CONSTANTS
	start_time = '2010-1-1'
	end_time = date.today()
	INITIAL_BUDGET = 1000
	weight = 1
	current_budget= INITIAL_BUDGET
	symbol_list = ['AAPL','NVDA','AMD']
	data = pd.DataFrame()
	for symbol in symbol_list:
		print("=======%s=======" % symbol)
		raw_data = round(get_data(symbol, start_time=start_time, end_time = end_time),2)
		data[symbol] = raw_data["Close"]
		current_budget, shares = orderBuy(current_budget, data[symbol])
		current_budget, shares = orderSell(current_budget, data[symbol], shares, weight)
		#===Benchmark===#
		roi_sp500,market_performs = SP500((round(data[symbol][-1]/data[symbol][1]*100,2)), \
										   start_time, end_time)
		#===============#
		print(market_performs)
		portfolio(INITIAL_BUDGET,current_budget, data[symbol], shares)
		'''
		dpc = daily_pct_change(data[symbol])
		dpc = dpc.fillna(0)
		print((1+dpc).cumprod())
		'''
if __name__ == "__main__":
	main()
	