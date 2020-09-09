"""
This module gets the newest data from the exchange, transforms the data to volume bars, applies fag, trains a
nn-cc model on it and evaluates it.
"""

from get_data import get_data_from_exchange
from parameters import api_key
import pandas as pd
from fag import create_volume_bars, apply_triple_bar_method
from mlfinlab.features.fracdiff import frac_diff_ffd, plot_min_ffd
import plotly.graph_objects as go


def add_trace_to_fig(fig, df):
    """
    :param df: Dataframe with open, close, high, low, top_bar and low_bar
    :param fig: Initiated ploty-fig
    :return: 
    """
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=df.index,
            y=df['top_bar'] * df['close'] + df['top_bar'] * 1.2,
            marker=dict(
                color='green',
                size=2,
                line=dict(
                    color='green',
                    width=3
                )
            ),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=df['date_time'],
            y=df['bot_bar'] * df['close'] - df['bot_bar'] * 1.2,
            marker=dict(
                color='green',
                size=2,
                line=dict(
                    color='green',
                    width=3
                )
            ),
            showlegend=False
        )
    )
    return fig


exchange = 'bitmex'
exchange_parameters = api_key
symbol = 'BTC/USD'
symbol_path = symbol.replace('/', '-')
tf = '1m'
file_name = 'resources/Bitmex_{}_{}.parquet'.format(symbol_path, tf)
limit = 1000
starting_from = '2017-01-10 00:00:00Z'
header = ['date_time', 'open', 'high', 'low', 'close', 'volume']

# Gets the newest data and saves it into a parquet.
get_data_from_exchange(symbol, symbol_path, starting_from, tf, file_name, limit, header, exchange,
                       exchange_parameters)

df = pd.read_parquet(file_name)
"""
PREPROCESSING
"""
df.index = pd.to_datetime(df.index)
df['date_time'] = df.index

# Check for duplicate candles from bitmex.
duplicate_values = df[df.duplicated(keep=False)]
if not duplicate_values.empty:
    print('Duplicate rows: ', duplicate_values)

print("Number of minute bars: {}".format(df.shape[0]))
df_volume_bars = create_volume_bars(df)
print("Number of volume bars: {}".format(df_volume_bars.shape[0]))
df_volume_bars_labeled = apply_triple_bar_method(df_volume_bars, atr_mult=2.8, nr_of_bars_for_vertical=13,
                                                 period_atr=7)
# Create plot to see the labels
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_volume_bars_labeled.index,
                         y=df_volume_bars_labeled['close'],
                         mode='lines',
                         name='lines'))
fig = add_trace_to_fig(fig, df_volume_bars_labeled)
fig.write_html("plotly/df_volume_bars_labeled.html")
#
# Save the labels and only keep the values which have to be differentiated
labels = df_volume_bars_labeled[['top_bar', 'bot_bar']]
X = df_volume_bars_labeled[['open', 'close', 'high', 'low', 'sum_of_candles']]

# Plotting the graph to find the minimum value to use for the fractional differentiation (Where the line crosses the
# 0.95 mark).
fig = plot_min_ffd(X)
fig.axhline(y=0.95, color='r', linestyle='--')
fig.figure.savefig("plots/plot_min_ffd.png", bbox_inches='tight')

# Deriving the fractionally differentiated features
df_differentiated = frac_diff_ffd(X, 0.3)

fig = go.Figure(data=[go.Candlestick(x=df_differentiated.index,
                                     open=df_differentiated['open'],
                                     high=df_differentiated['high'],
                                     low=df_differentiated['low'],
                                     close=df_differentiated['close'])])

fig.write_html("plots/df_differentiated.html")
