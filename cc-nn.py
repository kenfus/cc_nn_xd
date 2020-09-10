"""
This module gets the newest data from the exchange, transforms the data to volume bars, applies fag, trains a
nn-cc model on it and evaluates it.
"""

from get_data import get_data_from_exchange
from parameters import api_key
import pandas as pd
import numpy as np
from fag import create_volume_bars, apply_triple_bar_method
from mlfinlab.features.fracdiff import frac_diff_ffd, plot_min_ffd
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, plot_roc_curve


def add_trace_to_fig(fig, _df, max_bars=5000):
    """
    :param max_bars:
    :param df: Dataframe with open, close, high, low, top_bar and low_bar
    :param fig: Initiated ploty-fig
    :return:
    """
    df = _df.copy()
    df = df.tail(max_bars)
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=df.index,
            y=np.round(df['top_bar']) * df['close'] + np.round(df['top_bar']) * 1.2,
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
            x=df.index,
            y=np.round(df['bot_bar']) * df['close'] - np.round(df['bot_bar']) * 1.2,
            marker=dict(
                color='red',
                size=2,
                line=dict(
                    color='red',
                    width=3
                )
            ),
            showlegend=False
        )
    )
    return fig


def _frac_diff_ffd(_df, d):
    """
    :param df: Dataset to differentiate with open, close, high, low, top_bar, low_bar
    :param d: differentiate value
    :return: differentiated price-values with the top_bar and low_bar readded
    """
    df = _df.copy()
    labels = df[['top_bar', 'bot_bar']]
    df = df[['open', 'high', 'low', 'close']]
    df = frac_diff_ffd(df, d)
    return df.join(labels)


def add_past_rows(df_, period=7):
    """
    :param df: Dataframe with open, high, low, close, top_bar, bot_bar
    :param period: How much of the past should be added to a row
    :return: Dataframe with memory.
    """
    df = df_.copy()
    labels = df[['top_bar', 'bot_bar']]
    df = df[['open', 'high', 'low', 'close']]
    df_unshifted = df.copy()
    for i in range(1, period):
        df = df.join(df_unshifted.shift(periods=i), how='inner', rsuffix='_' + str(i))
    return df.join(labels).dropna()


def calc_nr_neuros(df_differentiated_memory, output_neurons, alpha=2):
    """
    This function calculates with pi * thumb how many neuros there should be in a hidden layer.
    https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    :param df_differentiated_memory: df which will be given to the nn-cc
    :param alpha: parameter to change the number of neurons in the hidden layer
    :return: int
    """
    sample_size, input_dim = df_differentiated_memory.shape
    return int(sample_size / (alpha * (input_dim + output_neurons)))


exchange = 'bitmex'
exchange_parameters = api_key
symbol = 'BTC/USD'
symbol_path = symbol.replace('/', '-')
tf = '1m'
file_name = 'resources/{}_{}_{}.parquet'.format(exchange, symbol_path, tf)
limit = 1000
starting_from = '2018-01-10 00:00:00Z'
header = ['date_time', 'open', 'high', 'low', 'close', 'volume']

# Gets the newest data and saves it into a parquet.
get_data_from_exchange(symbol, symbol_path, starting_from, tf, file_name, limit, header, exchange,
                       exchange_parameters)

df = pd.read_parquet(file_name)
df = pd.read_parquet('resources/ETH-USDT.parquet')
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
print("1 df_volume_bars.columns: ", df_volume_bars.columns)
print("Number of volume bars: {}".format(df_volume_bars.shape[0]))
df_volume_bars_labeled = apply_triple_bar_method(df_volume_bars, atr_mult=2.8, nr_of_bars_for_vertical=13,
                                                 period_atr=7)
# Create plot to see the labels
fig = go.Figure()
df_volume_bars_labeled_plot = df_volume_bars_labeled.tail(5000)
fig.add_trace(go.Scatter(x=df_volume_bars_labeled_plot.index,
                         y=df_volume_bars_labeled_plot['close'],
                         mode='lines',
                         name='lines'))
fig = add_trace_to_fig(fig, df_volume_bars_labeled_plot)
fig.write_html("plots/df_volume_bars_labeled.html")
#

# Plotting the graph to find the minimum value to use for the fractional differentiation (Where the line crosses the
# 0.95 mark).
fig = plot_min_ffd(df_volume_bars_labeled[['open', 'high', 'low', 'close']])
fig.axhline(y=0.95, color='r', linestyle='--')
fig.figure.savefig("plots/plot_min_ffd.png", bbox_inches='tight')

# Deriving the fractionally differentiated features. Creates a plot to check out the data
df_differentiated = _frac_diff_ffd(df_volume_bars_labeled, 0.30)
df_differentiated_plot = df_differentiated.tail(5000)

fig = go.Figure(data=[go.Candlestick(x=df_differentiated_plot.index,
                                     open=df_differentiated_plot['open'],
                                     high=df_differentiated_plot['high'],
                                     low=df_differentiated_plot['low'],
                                     close=df_differentiated_plot['close'])])
fig.write_html("plots/df_differentiated.html")

df_differentiated_memory = add_past_rows(df_differentiated, 13)

# We create an array to predict here for the softmax fuction: [1, 0, 0] to go long, [0, 1, 0] for short, [0,0,1] for no trade.
y = df_differentiated_memory[['top_bar', 'bot_bar']]
X = df_differentiated_memory.drop(columns=['top_bar', 'bot_bar'])
# We also need, because we are using softmax, a label to tell it not to trade
y['no_trade'] = np.where(y['top_bar'] + y['bot_bar'] == 0, 1, 0)

# Marco Lopez del Prado said that we should have a time gap between test and train. This is what we do here.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
print(X_valid.tail())
X_train.to_csv('X_train.csv')
y_train.to_csv('y_train.csv')

# Parameter for the model
input_dim = X.shape[1]
output_dim = 3
nr_of_neuros = calc_nr_neuros(df_differentiated_memory, output_dim, 2)

# Keras Model
model = Sequential()
model.add(Dense(input_dim + 1, input_dim=input_dim, activation='relu'))
model.add(Dense(nr_of_neuros, activation='relu'))
model.add(Dense(nr_of_neuros, activation='relu'))
model.add(Dense(nr_of_neuros, activation='relu'))
model.add(Dense(nr_of_neuros, activation='relu'))
model.add(Dense(nr_of_neuros, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()
# Run keras Model
history = model.fit(
    X_train,
    y_train,
    batch_size=1024,
    epochs=150,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_valid, y_valid),
)

# Evaluate Model
pred_ = pd.DataFrame(model.predict(X_valid), columns=['top_bar', 'bot_bar', 'no_trade'])
print("Prediction description: ", pred_.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
print(pred_.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))).tail(10))

len_ = np.minimum(pred_.shape[0], 5000)
pred_ = pred_.tail(len_)
pred_.index = df_volume_bars.tail(len_).index
print("df_volume_bars.columns: ", df_volume_bars.columns)
df_predicted = df_volume_bars.join(pred_, how='inner')
print("df_predicted: ", df_predicted.tail(10))
fig = go.Figure()
df_predicted_plot = df_predicted.tail(len_)
fig.add_trace(go.Scatter(x=df_predicted_plot.index,
                         y=df_predicted_plot['close'],
                         mode='lines',
                         name='lines'))

fig = add_trace_to_fig(fig, df_predicted, len_)
fig.write_html("plots/df_predicted.html")
