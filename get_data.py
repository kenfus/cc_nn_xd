"""
This module gets historical data from an exchange with the help of ccxt.
"""

import os
import sys
import time
import ccxt
import datetime
import pandas as pd
import numpy as np
from parameters import api_key


def save_data(data, header, file_name, exchange, symbol_path, tf):
    """
    Saves the current data we have to a parquet-file.
    If the file already exists, it appends the new entry to it.
    """
    if os.path.exists(file_name):
        '''
        Checking if file exist. If it does, we will reload the file and append the new data to it.
        In this if statement the new data gets directly saved. The rest of the if statement
        gets avoided by using a return True
        '''
        print('Updating data....')
        df_old = pd.read_parquet(file_name)
        df_new = pd.DataFrame(data, columns=header)
        df_new.index = df_new['date_time'].apply(exchange.iso8601)
        df = df_old.append(df_new)
        df.to_parquet(file_name.format(symbol_path, tf))
        print('Successfully updated data!')
        return True

    df = pd.DataFrame(data, columns=header)
    df.index = df['date_time'].apply(exchange.iso8601)
    print('Tail of the data Im saving..')
    print(df.tail())
    df.to_parquet(file_name.format(symbol_path, tf))
    return True


def miliseconds_from(i):
    """
    :returns the timeframe in miliseconds.
    """
    switcher = {
        '1m': 60 * 1000,
        '5m': 60 * 1000 * 5,
        '15m': 60 * 1000 * 15
    }
    return switcher.get(i, "Invalid timeframe")


def get_data_from_exchange(symbol, symbol_path, starting_from, tf, file_name, limit, header, exchange, exchange_parameters):
    exchange = getattr(ccxt, exchange)({**exchange_parameters})
    if False:  # "'test' in exchange.urls:
        '''
        This checks if the exchange also has a test-net. If it does, we use that. Commented out for live-data.
        '''
        exchange.urls['api'] = exchange.urls['test']  # ←----- switch the base URL to testnet
    # set timeframe in msecs
    tf_multi = miliseconds_from(tf)
    hold = 30
    sleep = 1

    print(exchange.fetch_balance())
    exchange.load_markets()

    from_timestamp = exchange.parse8601(starting_from)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z")
    end = exchange.parse8601(now)

    if os.path.exists(file_name):
        '''
        Checking if file exist. If it does, we will reload the file, get the last status and continue from there.
        '''
        df = pd.read_parquet(file_name)
        newest_entry = np.max(df.index)
        from_timestamp = exchange.parse8601(newest_entry) + tf_multi

    print("Getting data from {} to {}".format(exchange.iso8601(from_timestamp), exchange.iso8601(end)))

    # make list to hold data
    data = []

    if exchange.has['fetchOHLCV'] == 'emulated':
        print(exchange.id, " cannot fetch old historical OHLCVs, because it has['fetchOHLCV'] =",
              exchange.has['fetchOHLCV'])
        sys.exit()

    candle_no = (int(end) - int(from_timestamp)) / tf_multi + 1
    print('downloading...')
    while from_timestamp < end:
        try:
            try:
                ohlcvs = exchange.fetch_ohlcv(symbol, tf, from_timestamp, limit)
                # --------------------------------------------------------------------
                # ADDED:
                # check if returned ohlcvs are actually
                # within the from_timestamp > ohlcvs > end range
                if (ohlcvs[0][0] > end) or (ohlcvs[-1][0] > end):
                    print(exchange.id, " got a candle out of range! has['fetchOHLCV'] = ", exchange.has['fetchOHLCV'])
                    save_data(data, header, file_name, exchange, symbol_path, tf)
                    return True

                from_timestamp += len(ohlcvs) * tf_multi
                data += ohlcvs
                print(str(len(data)) + ' of ' + str(int(candle_no)) + ' candles loaded...')
                # save_data(data, header, file_name, exchange, symbol_path, tf)
                time.sleep(sleep)

            except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable,
                    ccxt.RequestTimeout) as error:
                print('Got an error ', type(error).__name__, error.args, ', retrying in ', hold, ' seconds...')
                save_data(data, header, file_name, exchange, symbol_path, tf)
                time.sleep(hold)

        except KeyboardInterrupt as e:
            print('KeyboardInterrupt! Saving data...')
            save_data(data, header, file_name, exchange, symbol_path, tf)
            return True

    save_data(data, header, file_name, exchange, symbol_path, tf)


if __name__ == '__main__':
    # params:
    exchange = 'bitmex'
    exchange_parameters = api_key
    symbol = 'ETH/USD'
    symbol_path = symbol.replace('/', '-')
    tf = '1m'
    starting_from = '2018-08-02 00:00:00Z'
    file_name = 'resources/{}_{}_{}_{}.parquet'.format(exchange, starting_from, symbol_path, tf)
    limit = 1000
    header = ['date_time', 'open', 'high', 'low', 'close', 'volume']

    get_data_from_exchange(symbol, symbol_path, starting_from, tf, file_name, limit, header, exchange,
                           exchange_parameters)
