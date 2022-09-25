from datetime import date
import pandas as pd 
from plotly.subplots import make_subplots
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import streamlit as st
from stqdm import stqdm
from talib import *




@st.cache
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data = data.reset_index()
    return data


# Plot raw data
@st.cache
def plot_data(data, selected_indicator, forecast,period):
	colors = px.colors.qualitative.Plotly
	fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

	#Prediction price
	if period == 0:
		pass
	else:
		fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
					  marker=dict(color='#0072B2',line=dict(width=1)),name='Prediction'),1,1)
		fig.add_trace(go.Scatter(x = forecast['ds'],y = forecast['yhat_lower'],
  								fill='tonexty',marker={'color':'#0072B2'},opacity = 0.001,showlegend = False,hoverinfo = 'none'))
		fig.add_trace(go.Scatter(x = forecast['ds'],y = forecast['yhat_upper'],
  								fill='tonexty',marker={'color':'#0072B2'},opacity = 0.001,showlegend = False,hoverinfo = 'none'))
		
	# Real stock price
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], marker={'color': 'blue'},
				 line = {'width': 3}, name="Close Price"),1,1)
	# Indicators
	for i in selected_indicator:
		if i == 'RSI':
			rsi =  RSI(data['Close'], timeperiod=14).fillna(0)
			fig.add_trace(go.Scatter(x=data['Date'], y=rsi, name="RSI"),2,1)
			fig.add_trace(go.Scatter(x = [data['Date'].min(), data['Date'].max()],
				y = [70, 70],showlegend=False,mode = "lines",line=dict(color='red', width=2)),row=2, col=1)
			fig.add_trace(go.Scatter(x = [data['Date'].min(), data['Date'].max()],
				y = [30, 30],showlegend=False,mode = "lines",line=dict(color='green', width=2)),row=2, col=1)

		if i == 'MA_50':
			ma_50 = MA(data['Close'], timeperiod=50, matype=0).fillna(0)
			fig.add_trace(go.Scatter(x=data['Date'], y=ma_50, name="Moving Average 50 days"),1,1)

		if i == 'MA_200':
			ma_200 = MA(data['Close'], timeperiod=200, matype=0).fillna(0)
			fig.add_trace(go.Scatter(x=data['Date'], y=ma_200, name="Moving Average 200 days"),1,1)

	fig.layout.update(title_text='Close stock price', width= 900, height=600, xaxis_rangeslider_visible=False)
	return fig
    


st.markdown("# Stock selection")
st.sidebar.markdown("# Main")



col1, col2, col3 = st.columns(3)

symbol_csv = pd.read_csv("S&P500-Symbols.csv")
symbol_list =  [s for s in symbol_csv['Symbol']]

indicator_list = ['RSI','MA_50', 'MA_200']

with col1:
	selected_stock = st.selectbox('Select dataset for prediction', symbol_list)
	

with col2:
	selected_start_date = st.date_input("Select a start date", value = date(2010, 1, 1), 
						  min_value = date(2010, 1, 1), max_value = date.today())
with col3:
	selected_end_date = st.date_input("Select an end date", value = date.today(), 
						  min_value = date(2010, 1, 1),max_value = date.today())


col_indicator, col_pred,_ = st.columns(3)
with col_indicator:
	selected_indicator = st.multiselect('Select a technical indicator', indicator_list, default=indicator_list[0])



data_load_state = st.text('Loading data...')
data = load_data(selected_stock, selected_start_date, selected_end_date)
data_load_state.text('Loading data... done!')


with col_pred:
	n_years = st.selectbox('Years of prediction:', ('0','1','2','3','4'))
	period = int(n_years) * 365

	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)


fig = plot_data(data,selected_indicator, forecast,period)
st.plotly_chart(fig)
