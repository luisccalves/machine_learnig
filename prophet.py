# -*- coding: utf-8 -*-

# instalar o yfinance
# pip install yfinance

import warnings
from datetime import datetime, timedelta

#import bibliotecas
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly

warnings.filterwarnings('ignore')
pd.options.display.float_format = '${:,.2f}'.format

hj = datetime.today().strftime('%Y-%m-%d')
data_ini = '2016-01-01'
df_eth = yf.download('ETH-USD', data_ini, hj)
df_eth.tail()

df_eth.reset_index(inplace=True)

print(df_eth)

df = df_eth[["Date", "Adj Close"]]
df.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

print(df)

# Grafico Preço de fechamento
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y']))

model = Prophet(seasonality_mode='multiplicative')
model.fit(df)

# criar df com datas no futuro
df_futuro = model.make_future_dataframe(periods=60)
df_futuro.tail(60)

# previsao
previsao = model.predict(df_futuro)
print(previsao)

previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(60)

# grafico
plot_plotly(model, previsao)

plot_components_plotly(model, previsao)
