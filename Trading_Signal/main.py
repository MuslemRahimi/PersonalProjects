import yfinance as yf
import pandas as pd
from talib import *
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import csv
from statistics import stdev
from Signal import Get_Signal
from data import *
from correlation import *
from mpt import *
from mc import monte_carlo
import warnings
import seaborn




#define the ticker symbol




if __name__ == "__main__":

	today = date.today()
	'''
	symbol = "AMD"
	data = get_data(symbol, end_time=today)
	signal = Get_Signal(data)

	print(signal.score_list)

	#print(int(len(signal.score_list)/2)-1)
	if sum(signal.score_list) >= int(len(signal.score_list)/2):
		trend = "Bullish"
	else:
		trend = "Bearish"
	

	print("Stock: %s | Number of Indicators: %s | Trend: %s " % (symbol, len(signal.score_list), trend))


	symbols_list = ['EVOK','AMD','SNGX','SNDL','TSLA']

	get_corr(symbols_list, upper_corr=0.3, lower_corr=-0.3)
	'''
	
	symbols_list = ['AAPL','NVDA','AMD','AHLA.DU','TSLA','VOW.DE']
	for symbol in symbols_list:
		data = get_data(symbol, start_time='2000-1-1', end_time = today)
		signal = Get_Signal(data)
		print('----------Signal-Score----------')
		print(signal.score_list)
		print('----------Trend----------')
		if sum(signal.score_list) >= int(len(signal.score_list)/2):
			trend = "Bullish"
		else:
			trend = "Bearish"
		print("Stock: %s | Number of Indicators: %s | Trend: %s " % (symbol, len(signal.score_list), trend))
		print('=====================================')
	mpt(symbols_list)
	
	


	#symbols_list = ['NVDA','AMD','^GSPC']
	#monte_carlo(symbols_list, days=20, iterations = int(1e6))

