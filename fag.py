"""
This module stands for "features automatically generated". It generates features automatically.
"""

import numpy as np


def get_daily_volume(_df, days_for_average=14, ratio=1 / 50):
    """
    :param df: candles with open, close, high, low and volume
    :param days_for_average: amount of days to get the daily averange volume
    :param ratio: ratio of daily volume to create one candle. Bro marco lopez de prado says a nice ratio is 1/50
    :returns: the optimal threshold of volume to create a volume bar
    """
    df = _df.copy()
    df = df.resample('D').sum()
    volume_threshold = df.tail(days_for_average)['volume'].mean() * ratio
    print("{} of daily {}-days average volume equals to {}.".format(ratio, days_for_average, volume_threshold))
    return volume_threshold


def create_volume_bars(_df, volume_for_new_candle=None):
    """
    :param volume_for_new_candle:
    :param df: candles with index (as date_time), open, close, high, low and volume
    :param volume_for_new_candle: volume_threshold to create a new bar. Usually calculated with a formula by our
    bro marco lopez de prado
    :returns: volume bars (without the unfinished volume bar)
    """
    df = _df.copy()
    if not volume_for_new_candle:
        volume_for_new_candle = get_daily_volume(df)
    df['cum_volume'] = df.volume.cumsum()
    df['nr_of_candle'] = np.floor_divide(df['cum_volume'], volume_for_new_candle)
    df['volume_in_candle'] = df['cum_volume'] - df['nr_of_candle'] * volume_for_new_candle
    df['sum_of_candles'] = df.groupby(['nr_of_candle'])['nr_of_candle'].transform('count')
    df_grouped = df.groupby('nr_of_candle').agg({'open': 'first',
                                                 'close': 'last',
                                                 'high': 'max',
                                                 'low': 'min',
                                                 'date_time': 'last',
                                                 'sum_of_candles': 'last',
                                                 'volume_in_candle': 'last'})
    # Drop last row because the last row is not complete do to the threshold not being reached:
    df_grouped = df_grouped.iloc[:-1]
    # Add the lost index back.
    df_grouped.index = df_grouped['date_time']
    return df_grouped


def reach_bar(_df, atr_mult, nr_of_bars_for_vertical, which='top'):
    """
    :param df: volume_bars with index (as date_time), open, close, high, low, atr and volume
    :param atr_mult: multiplier for the atr
    :param nr_of_bars_for_vertical: after how many bars the vertical bar will be produced.
    :param which: which barrier to create
    :return: dataframe with a 1 if a close-price will reach the top/bot bar in the next <nr_of_bars_for_vertical> bars.
    """
    df = _df.copy()
    if which == 'top':
        return np.where(df['close'].iloc[::-1].rolling(nr_of_bars_for_vertical, min_periods=0).max().iloc[::-1] >
                        df['close'] + df['ATR'] * atr_mult, 1, 0)
    elif which == 'bot':
        return np.where(df['close'].iloc[::-1].rolling(nr_of_bars_for_vertical, min_periods=0).min().iloc[::-1] <
                        df['close'] - df['ATR'] * atr_mult, 1, 0)
    else:
        print("Please use only 'top' or 'bot' as parameter")


def will_reach_top_bar(_df, atr_mult, nr_of_bars_for_vertical):
    """
    :param df: volume_bars with index (as date_time), open, close, high, low, atr and volume
    :param atr_mult: multiplier for the atr
    :param nr_of_bars_for_vertical: after how many bars the vertical bar will be produced.
    :return: series with information if the bar will reach the top bar in less than <nr_of_bars_for_vertical> bars
    """
    df = _df.copy()
    top_bars_reached = reach_bar(df, atr_mult, nr_of_bars_for_vertical, 'top')
    print("Price will pass a top bar {} times in this data.".format(np.sum(top_bars_reached)))
    return top_bars_reached


def will_reach_bot_bar(_df, atr_mult, nr_of_bars_for_vertical):
    """
    :param df: volume_bars with index (as date_time), open, close, high, low, atr and volume
    :param atr_mult: multiplier for the atr
    :param nr_of_bars_for_vertical: after how many bars the vertical bar will be produced.
    :return: series with information if the bar will reach the bot bar in less than <nr_of_bars_for_vertical> bars
    """
    df = _df.copy()
    bot_bars_reached = reach_bar(df, atr_mult, nr_of_bars_for_vertical, 'bot')
    print("Price will pass a bot bar {} times in this data.".format(np.sum(bot_bars_reached)))
    return bot_bars_reached


def apply_triple_bar_method(_df, atr_mult=2.8, nr_of_bars_for_vertical=13,
                            period_atr=7):
    """
    :param df: dataframe with volume bars with open, close, high and low.
    :param atr_mult:
    :param nr_of_bars_for_vertical:
    :param period_atr:
    :return: dataframe with volumebars, open high low close, atr and top bar and bot bar
    """
    df = _df.copy()
    df['TR'] = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    df['TR'] = np.maximum(df['TR'], np.abs(df['low'] - df['close'].shift(1)))
    df['ATR'] = df['TR'].ewm(span=period_atr).mean()
    df['top_bar'] = will_reach_top_bar(df, atr_mult, nr_of_bars_for_vertical)
    df['bot_bar'] = will_reach_bot_bar(df, atr_mult, nr_of_bars_for_vertical)
    # Sometimes both get reached in the next n candles. Thus we need to see which one get reached first
    # Is it doable without a for loop? Is this the correct approach?
    first = None
    for index, row in df.iterrows():
        if not first:
            if row['bot_bar']:
                first = 'bot'
            if row['top_bar']:
                first = 'top'
        if not row['top_bar'] and not row['bot_bar']:
            first = None
        if row['top_bar'] and row['bot_bar']:
            if first == 'top':
                df.loc[index, 'bot_bar'] = 0
            else:
                df.loc[index, 'top_bar'] = 0
    return df

