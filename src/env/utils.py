import pandas as pd
import os

def read_bitcoin_history(path):
    df = pd.read_csv(path)
    df_trim = df[['date', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT']]
    inverted_df_trim = df_trim.sort_index(ascending=False)
    trading_record_matrix = inverted_df_trim[['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT']].to_numpy()

    return trading_record_matrix

if __name__ == '__main__':
    path = os.path.join('/'.join(os.getcwd().split('/')[:-2]), 'data/Binance_BTCUSDT_minute.csv')
    trading_record_matrix = read_bitcoin_history(path)
    print(trading_record_matrix.shape)