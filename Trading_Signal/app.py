import pandas as pd
import dash
from dash import html
from dash import dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from tqdm import tqdm

import time
import yfinance as yf
from talib import *
import pandas as pd
from datetime import date
import csv


def get_data(symbol_list, period = '1d', start_time='2000-1-1', end_time = date.today()):
    df = pd.DataFrame()
    for symbol in tqdm(symbol_list):
        df_temp = pd.DataFrame()

        tickerData = yf.Ticker(symbol)
        tickerData = tickerData.history(period = period, start = start_time, end = end_time)

        tickerData = tickerData.reset_index()
        for i in range(len(tickerData)):

            df_temp = df_temp.append({'date': tickerData['Date'][i],'stock': symbol,'value': tickerData['Close'][i]},ignore_index=True)
    
        #Signals
        rsi= RSI(df_temp["value"], timeperiod=14)
        rsi= rsi.fillna(0)
        df_temp['RSI'] = rsi
        
        df = pd.concat([df,df_temp],axis=0)
        #axis=0 is very important
    
    df.sort_values(by='date',inplace=True)
    df = df.reset_index()
    print(df)
    return df



# Load data
today = date.today()
#symbol_list = pd.read_csv('symbol_list.csv')
#symbol_list= [symbol for symbol in symbol_list['Symbol']]


symbol_list = ['AAPL','AMD']
df = get_data(symbol_list, start_time='2021-1-1', end_time = today)



# Initialise the app
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
app.config.suppress_callback_exceptions = True
# Define the app



def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list


app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('DASH - STOCK PRICES'),
                                 html.P('Visualising time series with Plotly - Dash.'),
                                 html.P('Pick one or more stocks from the dropdown below.'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='stockselector', options=get_options(df['stock'].unique()),
                                                      multi=True, value=[df['stock'].sort_values()[0]],
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='stockselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'})
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries',
                                     config={'displayModeBar': True},
                                     animate=True),
                                  dcc.Graph(id='signals',
                                     config={'displayModeBar': False},
                                     animate=True)
                             ])
                              ])
        ]

)


# Callback for timeseries price
@app.callback(Output('timeseries', 'figure'),
              [Input('stockselector', 'value')])
def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    # STEP 1
    trace = []
    df_sub = df
    # STEP 2
    # Draw and append traces for each stock
    for stock in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub[df_sub['stock'] == stock]['date'],
                                 y=df_sub[df_sub['stock'] == stock]['value'],
                                 mode='lines',
                                 opacity=0.9,
                                 name=stock,
                                 textposition='bottom center'))
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
                  
              ),
              }
    
              

    return figure




@app.callback(Output('signals', 'figure'),
              [Input('stockselector', 'value')])
def update_signals(selected_dropdown_value):
    ''' Draw traces of the feature 'change' based one the currently selected stocks '''
    trace = []
    df_sub = df
    # Draw and append traces for each stock
    for stock in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub[df_sub['stock'] == stock]['date'],
                                 y=df_sub[df_sub['stock'] == stock]['RSI'],
                                 mode='lines',
                                 opacity=0.7,
                                 name=stock,
                                 textposition='bottom center'))
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'t': 50},
                  height=250,
                  hovermode='x',
                  autosize=True,
                  title={'text': '14 Period RSI', 'font': {'color': 'white'}, 'x': 0.5},
                  yaxis={'showticklabels': False, 'range': [0, 100]},
              ),
              }

    return figure





if __name__ == '__main__':
    app.run_server(debug=True)
Footer
