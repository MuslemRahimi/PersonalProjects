import yfinance as yf
import pandas as pd
from datetime import date
import csv


def get_data(symbol, period = '1d', start_time='2000-1-1', end_time = date.today()):
	tickerData = yf.Ticker(symbol)
	tickerData = tickerData.history(period = period, start = start_time, end = end_time)
	return tickerData

def get_symbol():
	with open("symbol_list.csv",newline="") as file:
		reader = csv.reader(file)
		data = list(reader)
	symbol_list = []

	for i in range(1,len(data)):
		#print(data[i][0])
		symbol_list.append(data[i][0])

	return symbol_list