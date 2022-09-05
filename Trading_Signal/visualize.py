#plt.plot(data["Close"])
	#plt.title(symbol+"| Trend: "+trend)
	#plt.show()
	#================================#
	"""
	fig, axs = plt.subplots(3, 3)
	axs[0, 0].plot(data["Close"], color="black",label="Close value")
	axs[0, 0].set_title(symbol)
	axs[1, 0].plot(macd, color="blue",label="MACD")
	axs[1, 0].plot(macdsignal, color="red",label="MACD Signal")
	axs[1, 0].set_title("MACD")
	axs[1, 0].sharex(axs[0, 0])
	axs[0, 1].plot(rsi, color="blue",label="RSI")
	axs[0, 1].set_title("RSI")
	axs[0,1].axhline(y=30,xmin=0,xmax=2022, color="red")
	axs[0,1].axhline(y=70,xmin=0,xmax=2022, color="red")

	axs[0, 2].plot(slowk, color="blue")
	#axs[0, 2].plot(slowd, color="red")
	axs[0, 2].set_title("STOCH")
	axs[0,2].axhline(y=20,xmin=0,xmax=2022, color="red")
	axs[0,2].axhline(y=80,xmin=0,xmax=2022, color="red")

	axs[1, 1].plot(ma_50, color="blue",label="50 Moving Average")
	axs[1, 1].plot(ma_200, color="red",label="200 Moving Average")
	axs[1, 1].set_title("MA-50-200")

	axs[1, 2].plot(upperband, color="blue")
	axs[1, 2].plot(middleband, color="black")
	axs[1, 2].plot(lowerband, color="red")
	#axs[0, 2].plot(slowd, color="red")
	axs[1, 2].set_title("BBANDS")

	axs[2, 0].plot(data["Close"], color="black",label="Close value")
	axs[2,0].axhspan(level1,price_min, alpha=0.5, color='orange')
	axs[2,0].axhspan(level2, level1, alpha=0.5, color='red')
	axs[2,0].axhspan(level3, level2, alpha=0.5, color='green')
	axs[2,0].axhspan(price_max, level3, alpha=0.5, color='blue')

	axs[2, 1].set_title("Chaikin A/D Line")
	axs[2, 1].plot(Chaikin_Line, color="black")

	axs[2, 2].set_title("Standard deviation")
	axs[2, 2].plot(data["Close"].rolling(window=30).std(), color="red")

	fig.tight_layout()
	plt.show()
	"""