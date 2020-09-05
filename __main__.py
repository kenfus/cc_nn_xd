from data_stream import DataFeed


if __name__ == '__main__':
    DataFeed = DataFeed(exchange='bitmex', bar_type='volume', timeframe='1m', type_threshold='calculated', min_bars=30)
    DataFeed.preload_data()

    while True:
        if DataFeed.has_new_candle():
            new_candle = DataFeed.get_new_candle()
        else:
            continue
        signal = nn_cc(new_candle)
        if signal long :
            bla blub