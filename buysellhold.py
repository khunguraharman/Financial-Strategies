from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split


def process_data_for_labels(ticker):
    hm_days = 7  # forecasting 7 days into the future
    df = pd.read_csv('sp500_joined_Adj_Closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):  # 1-7
        df['{}_{}'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(days):  # *args allows one to pass any number of parameters
    cols = [c for c in days]  # creating a list from the input arguments
    requirement = 0.01
    for col in cols:
        if col > requirement:
            return 1
        elif col < -requirement:
            return -1
        else:
            return 0


def extract_featuresets(ticker, hm_days):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = df.loc[:, ['{}_{}'.format(ticker, i) for i in range(1, hm_days+1)]].apply(
        buy_sell_hold, axis=1)
    vals = df['{}_target'.format(ticker)].to_numpy().tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))  # counts how man 1's, 0's, and -1's
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)  # replace any infinities with nan
    df.dropna(inplace=True)
    df_vals = df[[ticker for ticker in tickers]].pct_change()  # computes the pct_change that occurs
    # from row n-1 to row n
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    x = df_vals.to_numpy()
    y = df['{}_target'.format(ticker)].to_numpy()

    return x, y, df


def do_ml(ticker, hm_days):
    x, y, df = extract_featuresets(ticker, hm_days)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)
    #clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.LinearSVC()), ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(train_x, train_y)
    confidence = clf.score(test_x, test_y)
    print('Accuracy:', confidence)
    predictions = clf.predict(test_x)
    print('Predicted Spread:',  Counter(predictions))
    return confidence


results = do_ml('AAPL', 7)




# extract_featuresets('ZION')
# features, labels, data = extract_featuresets('ZION', 7)


