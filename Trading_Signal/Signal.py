import pandas as pd
from talib import *
import matplotlib.pyplot as plt


class Get_Signal:
	def __init__(self, data):
		self.data = data

		rsi = self.RSI_signal()
		macd = self.MACD_signal()
		ma = self.MA_signal()
		apo = self.APO_signal()
		aroon = self.AROON_signal()
		aroonosc = self.AROONOSC_signal()
		bop = self.BOP_signal()
		cci = self.CCI_signal()
		cmo = self.CMO_signal()
		dx = self.DX_signal()
		mfi = self.MFI_signal()
		mom = self.MOM_signal()
		roc = self.ROC_signal()
		stoch = self.STOCH_signal()
		stochf = self.STOCHF_signal()
		stochrsi = self.STOCHRSI_signal()
		trix = self.TRIX_signal()
		ultosc = self.ULTOSC_signal()
		willr = self.WILLR_signal()
		obv = self.OBV_signal()
		self.score_list= [rsi, macd, ma, apo, aroon,aroonosc,bop,cci,cmo,\
						  dx,mfi,mom,roc,stoch,stochf,stochrsi,trix,ultosc,\
						  willr]



	def MA_signal(self):
		signal = 0
		ma_50 = MA(self.data["Close"], timeperiod=50, matype=0)
		ma_200 = MA(self.data["Close"], timeperiod=200, matype=0)
		if  ma_50[-1] > ma_200[-1]:
			#print( ma_50[-1]/ma_200[-1]-1)
			signal = 1
		else:
			signal = 0
		return signal

	def MACD_signal(self):
		macd, macdsignal, macdhist = MACD(self.data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
		if macd[-1] > macdsignal[-1]:
			signal = 1
		else:
			signal = 0
		return signal

	def RSI_signal(self):
		rsi = RSI(self.data["Close"], timeperiod=14)

		if rsi[-1] <= 40:
			signal = 1
		else:
			signal = 0
		return signal

	def APO_signal(self):
		real = APO(self.data["Close"], fastperiod=12, slowperiod=26, matype=0)
		if real[-1] > 0:
			signal = 1
		else:
			signal = 0
		return signal

	def AROON_signal(self):
		aroondown, aroonup = AROON(self.data["High"], self.data["Low"], timeperiod=25)

		if aroonup[-1] > aroondown[-1]:
			signal = 1
		else:
			signal = 0
		return signal

	def AROONOSC_signal(self):
		real = AROONOSC(self.data["High"], self.data["Low"], timeperiod=25)

		if real[-1] > 0 and real[-1] > real[-2]:
			signal = 1
		else:
			signal = 0
		return signal

	def BOP_signal(self):
		real = BOP(self.data["Open"], self.data["High"], self.data["Low"], self.data["Close"])

		if real[-1] > 0 and real[-1] > real[-2]:
			signal = 1
		else:
			signal = 0
		return signal

	def CCI_signal(self):
		real = CCI(self.data["High"], self.data["Low"], self.data["Close"], timeperiod=14)

		if real[-1] >=0 and real[-1] > real[-2]:
			signal = 1
		else:
			signal = 0
		return signal

	def CMO_signal(self):
		real = CMO(self.data["Close"], timeperiod=14)

		if real[-1] >= -50 and real[-1] <= -35:
			signal = 1
		else:
			signal = 0
		return signal

	def DX_signal(self):
		# Combine with Minus Directional Indicator
		real = DX(self.data["High"], self.data["Low"], self.data["Close"], timeperiod=14)

		if real[-1] > real[-2] and real[-1] > real[-3]:
			signal = 1
		else:
			signal = 0
		return signal

	def MFI_signal(self):
		real = MFI(self.data["High"], self.data["Low"], self.data["Close"], self.data["Volume"], timeperiod=14)
		if real[-1] < 25:
			signal = 1
		else:
			signal = 0
		return signal

	def MOM_signal(self):
		real = MOM(self.data["Close"], timeperiod=10)
		if real[-1] > real[-2] and real[-1] > real[-3] and real[-1] > real[-4]:
			signal = 1
		else:
			signal = 0
		return signal

	def ROC_signal(self):
		signal = 0
		real = ROC(self.data["Close"], timeperiod = 14)

		if real[-1] > 0 and real[-1] > real[-2]:
			signal = 1
		return signal

	def STOCH_signal(self):
		# Take the mean of slowk slowd??
		signal = 0
		slowk, slowd = STOCH(self.data["High"], self.data["Low"],self.data["Close"], \
					fastk_period=5, slowk_period=3,slowk_matype=0, slowd_period=3, slowd_matype=0)

		if slowk[-1] <= 20 or slowd[-1] <=20:
			signal = 1
		return signal

	def STOCHF_signal(self):
		# Take the mean of fastk fastd??
		signal = 0
		fastk, fastd = STOCHF(self.data["High"], self.data["Low"],self.data["Close"], \
			fastk_period=5, fastd_period=3, fastd_matype=0)

		if fastk[-1] <= 20 or fastd[-1] <=20:
			signal = 1
		return signal

	def STOCHRSI_signal(self):
		signal = 0
		fastk, fastd = STOCHRSI(self.data["Close"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

		if fastk[-1] <= 20 or fastd[-1] <=20:
			signal = 1
		return signal

	def TRIX_signal(self):
		signal = 0
		real = TRIX(self.data["Close"], timeperiod=30)
		if real[-1] > 0 and real[-1] > real[-2]:
			signal = 1
		return signal

	def ULTOSC_signal(self):
		signal = 0
		real = ULTOSC(self.data["High"], self.data["Low"],self.data["Close"], \
						timeperiod1=7, timeperiod2=14, timeperiod3=28)
		if real[-1] <=30:
			signal = 1
		return signal

	def WILLR_signal(self):
		signal = 0
		real = WILLR(self.data["High"], self.data["Low"],self.data["Close"], timeperiod=30)
		real = real
		if real[-1] <= -80:
			signal = 1
		return signal

	def OBV_signal(self):
		signal = 0
		real = OBV(self.data["Close"], self.data["Volume"])