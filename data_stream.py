class DataFeed(exchange='bitmex', bar_type='volume', timeframe='1m', type_threshold='calculated'):
    """
    This class makes the connection with the defined ccxt-exchange, gets the new bars and transforms them.
    """
    def __init__(self, exchange, bar_type, timeframe, type_threshold, min_bars):
        self.exchange = exchange
        self.timeframe = timeframe
        self.type_threshold= type_threshold
        self.current_threshold = 0.0
        self.min_bars = min_bars
        self.bar_type = bar_type
        self.candles = pd.DataFrame()
        self.type_bars = None
        self.last_candle = None

    def preload_data(self):
        """
        This function gets the newest data from <exchange>, creates type-bars from it and loads it into memory.
        Also calculates the threshold if it's set to calculated. This works by first reading the minute bars
        from the SSD, adding the newest candles to it, generate type_bars from it and keep only the incomplete part of
        the last type_bar. In the memory. Not 100% sure if that makes a difference. However, if you start with minute_bars
        at 14:00 (where the model was trained) instead at 15:00, the generated volume_bars will be different -> impact?
        """
        self.candles = get data from bitmex
        self.type_threshold = if not self.type_threshold.isnumeric() CAlculate threshold else self.type_threshold
        # get and transform candles from bitmex
        self.current_threshold = threshold of unfinished typebar
        self.candles = unfinished type bar

    def exchange_has_new_bar(self):
        """
        :returns True if the exchange has a new bar available.
        """
    def enough_for_new_bar(self):
        """
        :returns Checks if the new threshold is high enough to generate a new <type> bar.
        """
        self.current_threshold += threshold_of_newest_candle
        self.candles += newest candle
        return True if self.current_threshold > self.type_threshold else False

    def get_new_candle(self):
        """
        :returns The new <type> bar if applicable, else returns False
        """
        self.bar_types = transform(self.candles)
        self.candles = unsued candles
        self.threshold -= threshold of used candles
        return self.bar_types

    def has_new_candle(self):
        if self.exchange_has_new_bar():
            return True if self.enough_for_new_bar() else False
        else:
            return False

