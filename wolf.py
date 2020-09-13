"""
This python-module backtests a strategy. The input needs to have open, close, high, low, atr, top_bar, bot_bar and no trade.
It accepts a threshold, which decides what the propability by ML has to be to take a trade
"""

import numpy a np
import pandas as pd

class Wolf(df, threshold, starting_cash, atr_sl, atr_tp, slippage):
    """
    Wolf backtests a strategy. If the model said not to trade, it will adapt the sl.
    """
    def __init__():
        self.df = df[['open', 'close', 'high', 'low', 'atr', 'top_bar', 'bot_bar', 'no_trade']]
        self.df['top_bar'] = np.rint(self.df['top_bar'] - ( threshold - 0.5))
        self.df['bot_bar'] = np.rint(self.df['bot_bar'] - ( threshold - 0.5))
        self.df['no_trade'] = np.where((not self.df['top_bar']) and (not self.df['bot_bar']), 1, 0)
        self.current_cash = starting_cash
        self.slippage = slippage
        self.entry_price = None
        self.atr_tp = None
        self.atr_sl = None
        self.tp_level = None
        self.sl_level = None
        self.position_size = None
        self.trade = None
        self.trades = pd.DataFrame()

    def back_test():
        for (i, open, close, high, low, atr, top_bar, bot_bar, no_trade) in self.df.itertuples():
            if no_trade:
                if self.position_size > 0:
                    self._update_long()
                    if high > self.tp_level:
                        self.current_cash += (high - self.entry_price ) * self.position_size
                    if low < self.sl_level:
                        self.current_cash += (low - self.entry_price) * self.position_size

                if self.position_size < 0:
                    self._update_short()
                    if low < self.tp_level:
                        self.current_cash += ( low - self.entry_price ) * self.position_size
                    if high > self.sl_level:
                        self.current_cash += ( high - self.entry_price ) * self.position_size
                    
            if top_bar:
                self._update_long()
                if not self.position_size:
                    self.entry_price = close
                    self.position_size = self.current_cash
            if bot_bar:
                self._update_short()
                if not self.position_size:
                    self.entry_price = close
                    self.position_size = -self.current_cash



        def _go_long(close):
            self.entry_price = close
            self._update_long()

        def _go_short(close):
            self.entry_price = close
            self._update_short()

        def _update_long(close):
            if self.tp_level < close + self.atr * self.atr_tp:
                self.tp_level = close + self.atr * self.atr_tp
                self.tp_level = close - self.atr * self.sl_level

        def _update_short(close):
            if self.tp_level > close - self.atr * self.atr_tp:
                self.tp_level = close - self.atr * self.atr_tp
                self.tp_level = close + self.atr * self.sl_level
