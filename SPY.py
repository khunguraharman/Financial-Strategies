import bs4 as bs
import requests
import pickle
import datetime as dt
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:  # first table row is just headers for each column, not actually needed
        ticker = row.findAll('td')[0].text.strip()  # the ticker is the data in the first column of each row
        removehyphens = str.maketrans(".", "-")  # want to replace . with -
        ticker = ticker.translate(removehyphens)  # perform translation
        tickers.append(ticker)  # add ticker to list of tickers
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)
    return tickers


def get_data_from_yahoo(reload_sp500):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2020, 6, 30)
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            company = yf.Ticker(ticker)
            df = company.history(interval='1d', start=start, end=end)
            df['Adj Close'] = df.loc[:, ['Stock Splits', 'Close']].apply(get_adj_close, axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}' .format(ticker))


def get_adj_close(x):
    stock_split = x[0]
    closing_price = x[1]
    if stock_split == 0:
        stock_multiple = 1
    else:
        stock_multiple = stock_split
    adj_close = closing_price/stock_multiple
    return adj_close


def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': ticker}, inplace=True)  # rename Adj Close as ticker
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], 1, inplace=True)  # Drop all other columns
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')  # Added other ticker as a column
    main_df.to_csv('sp500_joined_Adj_Closes.csv')


def visualize_data():
    df = pd.read_csv('sp500_joined_Adj_Closes.csv')
    # df['AAPL'].plot()
    # plt.show()
    df_corr = df.corr()
    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)  # correlation is bounded by +/- 1
    plt.tight_layout()
    plt.show()


#get_data_from_yahoo(True)
#compile_data()
style.use('ggplot')
visualize_data()



