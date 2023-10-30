import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd 
import numpy as np
import ta 

import warnings
warnings.filterwarnings('ignore')

from binance import Client
from dotenv import load_dotenv
import os 

# Chargez les variables d'environnement à partir du fichier .env
load_dotenv()

# Accédez à vos clés d'API à l'aide de variables d'environnement
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, SECRET_KEY) 

def BackTest(serie, annualized_scalar=252):
    global serie_print
    
    # FUNCTION TO BUT THE RIGHT DATE FORMAT
    def date_format(df):
        df.index.name = "date_time"
        df = df.reset_index(drop=False)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['date_time'] = df['date_time'].dt.date
        df = df.set_index('date_time')
        return df
    
    def drawdown_function(serie):
        # We compute Cumsum of the returns
        cum = serie.dropna().cumsum() + 1

        # We compute max of the cumsum on the period (accumulate max)
        running_max = np.maximum.accumulate(cum)

        # We compute drawdown
        drawdown = cum/running_max - 1
        return drawdown

    # Import the benchmark
    sp500 = yf.download("^GSPC")["Adj Close"].pct_change(1)

    # Change the name
    sp500.name = "SP500"

    try:
        # Concat the returns and the sp500
        val = pd.concat((serie, sp500), axis=1).dropna()
    except:
        # Put the right date format
        sp500 = date_format(sp500)
        serie = date_format(serie)

        # Concat the returns and the sp500
        val = pd.concat((serie, sp500), axis=1).dropna()

    # Compute the drawdown
    drawdown = drawdown_function(serie) * 100

    # Compute max drawdown
    max_drawdown = -np.min(drawdown)

    # Create a figure with subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],
                        subplot_titles=("Cumulative Return", "Drawdown"))

    # Add traces for cumulative return
    fig.add_trace(go.Scatter(x=val.index, y=val["return"].cumsum() * 100,
                             mode='lines', name='Portfolio', line=dict(color="#39B3C7")),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=val.index, y=val["SP500"].cumsum() * 100,
                             mode='lines', name='SP500', line=dict(color="#B85A0F")),
                  row=1, col=1)

    # Add traces for drawdown
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values.flatten(),
                             fill='tozeroy', name='Drawdown', line=dict(color="#C73954")),
                  row=1, col=2)

    # Update axis labels and titles
    fig.update_xaxes(title_text="Date", showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return %", showgrid=True, row=1, col=1)
    fig.update_xaxes(title_text="Date", showgrid=True, row=1, col=2)
    fig.update_yaxes(title_text="Drawdown %", showgrid=True, row=1, col=2)

    # Set the subplot titles
    fig.update_layout(title=f"Backtesting {name}/USDT", title_font_size=20, title_x=0.5)

    # Display the plot
    fig.show()

    serie_print = serie
    # Compute the sortino
    sortino = np.sqrt(annualized_scalar) * serie.mean() / serie.loc[serie < 0].std()

    # Compute the beta
    beta = np.cov(val[["return", "SP500"]].values, rowvar=False)[0][1] / np.var(val["SP500"].values)

    # Compute the alpha
    alpha = annualized_scalar * (serie.mean() - beta * serie.mean())

    # Print the statistics
    print(f"Sortino: {np.round(sortino, 3)}")
    print(f"Beta: {np.round(beta, 3)}")
    print(f"Alpha: {np.round(alpha * 100, 3)} %")
    try:
        print(f"MaxDrawdown: {np.round(max_drawdown[0], 3)} %")
    except:
        print(f"MaxDrawdown: {np.round(max_drawdown, 3)} %")


def XGBoost(name, lag=10):
    def feature_engineering(df, lag=10):
        """ Create new variables"""

        lag = 10

        # We copy the dataframe to avoid interferences in the data
        df_copy = df.dropna().copy()

        # Create the returns
        df_copy["returns"] = df_copy["close"].pct_change(1)
        df_copy["returns_lag"] = df_copy["close"].pct_change(lag)

        # Create the volatilities
        df_copy["MSD 10"] = df_copy[["returns"]].rolling(10).std().shift(lag)
        df_copy["MSD 30"] = df_copy[["returns"]].rolling(30).std().shift(lag)

        # Create the Ichimoku
        IC = ta.trend.IchimokuIndicator(df_copy["high"], df_copy["low"])
        df_copy["ichimoku_a"] = IC.ichimoku_a().shift(lag)
        df_copy["ichimoku_b"] = IC.ichimoku_b().shift(lag)
        df_copy["ichimoku_base"] = IC.ichimoku_base_line().shift(lag)
        df_copy["ichimoku_conversion"] = IC.ichimoku_conversion_line().shift(lag)

        return df_copy.dropna()

    def import_historical_data(name, interval, start_date, end_date):
        klines = client.get_historical_klines(name, interval, start_date, end_date)
        klines = [[x[0], float(x[1]), float(x[2]), float(x[3]),
                  float(x[4]), float(x[5])] for x in klines]
        klines = pd.DataFrame(klines, columns=['date', 'open',
                                               'high', 'low',
                                               'close', 'volume'])
        klines['date'] = pd.to_datetime(klines['date'], unit='ms')
        klines = klines.set_index('date')
        return klines

    # Import historical data
    klines = import_historical_data(name, interval, start_date, end_date)

    # Create new features
    dfc = feature_engineering(klines, lag)

    dfc['dummy'] = 0
    dfc.loc[dfc['returns_lag'] > np.quantile(dfc['returns_lag'], 0.70), 'dummy'] = 1

    # Percentage train set
    split = int(0.90 * len(dfc))

    # Train set creation
    X_train = dfc.iloc[:split, 7:-1]
    y_train = dfc[['dummy']].iloc[:split]

    # Test set creation
    X_test = dfc.iloc[split:, 7:-1]
    y_test = dfc[['dummy']].iloc[split:]

    # Standardisation
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # Import the class
    from xgboost import XGBClassifier

    # Initialize the class
    reg = XGBClassifier(max_depth=10, n_estimators=200, random_state=100)

    # Fit the model
    reg.fit(X_train_sc, y_train)

    # Create prediction for the whole dataset
    X = np.concatenate((X_train_sc, X_test_sc), axis=0)

    dfc['prediction'] = reg.predict(X)

    # Compute the position
    dfc['position'] = dfc['prediction']

    # Compute the returns
    dfc['strategy'] = np.array([dfc['returns'].shift(i) for i in range(lag)]).sum(axis=0) * (dfc['position'].shift(lag))
    dfc['return'] = dfc['strategy'].dropna()
    dfc = dfc['return'].iloc[split:] / lag
    return dfc


namelist = ["ETC", "ETH", "LTC", "BTC", "XRP", "MATIC"]
returns = pd.DataFrame()

interval = Client.KLINE_INTERVAL_1DAY  
start_date = '2019-01-01'  
end_date = '2023-09-20'

for name in namelist:
    ret = XGBoost(f"{name}USDT", lag=7)
    returns = pd.concat((returns, ret), axis=1)
    BackTest(ret, 52)