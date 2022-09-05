import yfinance as yf
import pandas as pd
from talib import *
from datetime import date
import matplotlib.pyplot as plt
import csv
from statistics import stdev
from Signal import Get_Signal
from data import *
import warnings
import seaborn

def get_corr(symbols_list, upper_corr = 0.2, lower_corr= -0.2):
	symbols = []
	for ticker in symbols_list:
		r = get_data(ticker)
		r["Symbol"] = ticker
		symbols.append(r)
	df = pd.concat(symbols)
	df = df.reset_index()
	df = df[['Date', 'Close', 'Symbol']]
	df_pivot=df.pivot('Date','Symbol','Close').reset_index()
	#print(df_pivot.head())

	corr_df = df_pivot.corr(method='pearson')
	#reset symbol as index (rather than 0-X)
	#del corr_df.index.name
	#print(corr_df.head(10))


	fig, ax = plt.subplots(figsize=(13, 8))
	seaborn.heatmap(corr_df, annot=True, cmap= "RdYlGn")
	plt.savefig("corr_fig.png")


	fig, ax = plt.subplots(figsize=(13, 8))
	dfCorr = corr_df
	filter_df = dfCorr[((dfCorr <= upper_corr) & (dfCorr >= lower_corr)) & (dfCorr !=1.000)]

	seaborn.heatmap(filter_df, annot=True, cmap= "RdYlGn")
	plt.savefig("corr_constrain.png")