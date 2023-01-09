import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
plt.style.use("seaborn")

class FeatureManager():

    def __init__(self,window = 50, lags =5) -> None:
        self.window = window
        self.lags = lags
        self.cols = []
        self.df = None

    def build_feature(self,data):

        sd = data.copy()

        open = sd["Open"]
        close = sd["Close"]
        high = sd["High"]
        low = sd["Low"]
        volume = sd["Volumn"]

        sd["returns"] = np.log(close/close.shift())
        sd["dir"] = np.where(sd["returns"] > 0,1,0)
        sd["sma"] = close.rolling(self.window).mean() - close.rolling(150).mean()
        sd["boll"] = (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()
        sd["boll7"] = (close - close.rolling(7).mean()) / close.rolling(7).std()
        sd["boll14"] = (close - close.rolling(14).mean()) / close.rolling(14).std()
        sd["boll21"] = (close - close.rolling(21).mean()) / close.rolling(21).std()
        sd["min"] = close.rolling(self.window).min() / close - 1
        sd["min7"] = close.rolling(7).min() / close - 1
        sd["min14"] = close.rolling(14).min() / close - 1
        sd["min21"] = close.rolling(21).min() / close - 1
        sd["max"] = close.rolling(self.window).max() / close - 1
        sd["max7"] = close.rolling(7).max() / close - 1
        sd["max14"] = close.rolling(14).max() / close - 1
        sd["max21"] = close.rolling(21).max() / close - 1
        sd["mom"] = ta.MOM(close, timeperiod=10)
        sd["mom7"] = ta.MOM(close, timeperiod=7)
        sd["mom14"] = ta.MOM(close, timeperiod=14)
        sd["mom21"] = ta.MOM(close, timeperiod=21)
        sd["vol"] = sd["returns"].rolling(self.window).std()
        sd["vol7"] = sd["returns"].rolling(7).std()
        sd["vol14"] = sd["returns"].rolling(14).std()
        sd["vol21"] = sd["returns"].rolling(21).std()
        sd["obv"] = ta.OBV(close,volume)
        sd["mfi7"] = ta.MFI(high, low, close, volume, timeperiod=7)
        sd["mfi14"] = ta.MFI(high, low, close, volume, timeperiod=14)
        sd["mfi21"] = ta.MFI(high, low, close, volume, timeperiod=21)
        sd["rsi7"] = ta.RSI(close, timeperiod=7)
        sd["rsi14"] = ta.RSI(close, timeperiod=14)
        sd["rsi21"] = ta.RSI(close, timeperiod=21)
        sd["adx7"] = ta.ADX(high, low, close, timeperiod=7)
        sd["adx14"] = ta.ADX(high, low, close, timeperiod=14)
        sd["adx21"] = ta.ADX(high, low, close, timeperiod=21)
        sd["roc"] = ta.ROC(close, timeperiod=10)
        sd["roc7"] = ta.ROC(close, timeperiod=7)
        sd["roc14"] = ta.ROC(close, timeperiod=14)
        sd["roc21"] = ta.ROC(close, timeperiod=21)
        sd["atr7"] = ta.ATR(high, low, close, timeperiod=7)
        sd["atr14"] = ta.ATR(high, low, close, timeperiod=14)
        sd["atr21"] = ta.ATR(high, low, close, timeperiod=21)
        sd["bop"] = ta.BOP(open, high, low, close)
        sd["ad"] =ta.AD(high, low, close, volume)
        sd["adosc"] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        sd["trange"] = ta.TRANGE(high, low, close)
        sd["ado"] = ta.APO(close, fastperiod=12, slowperiod=26, matype=0)
        sd["willr7"] = ta.WILLR(high, low, close, timeperiod=7)
        sd["willr14"] = ta.WILLR(high, low, close, timeperiod=14)
        sd["willr21"] = ta.WILLR(high, low, close, timeperiod=21)
        sd["dx7"] = ta.DX(high, low, close, timeperiod=7)
        sd["dx14"] = ta.DX(high, low, close, timeperiod=14)
        sd["dx21"] = ta.DX(high, low, close, timeperiod=21)
        sd["trix"] = ta.TRIX(close, timeperiod=30)
        sd["ultosc"] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        sd.dropna(inplace=True)

        #Add lags feature
        features = [
            "dir",      
            "sma",      
            "boll",     
            # "boll7",
            # "boll14",
            # "boll21",
            "min",      
            # "min7",      
            # "min14",
            # "min21",
            "max",      
            # "max7",      
            # "max14",
            # "max21",
            "mom",
            # "mom7",      
            # "mom14",
            # "mom21",
            "vol",      
            # "vol7",      
            # "vol14",
            # "vol21",
            "obv",      
            # "mfi7",     
            "mfi14",
            # "mfi21",
            # "rsi7",      
            "rsi14",
            # "rsi21",
            # "adx7",      
            "adx14",
            # "adx21",
            "roc",      
            # "roc7",      
            # "roc14",
            # "roc21",
            # "atr7",      
            "atr14",
            # "atr21",
            "bop",      
            "ad",       
            "adosc",     
            "trange",    
            "ado",       
            # "willr7",     
            "willr14",
            # "willr21",
            # "dx7",     
            "dx14",
            # "dx21",
            "trix",     # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
            "ultosc"    # Ultimate Oscillator
            ]
        
        for f in features:
            for lag in range(1,self.lags + 1):
                col = "{}_lag_{}".format(f,lag)
                sd[col] = sd[f].shift(lag)
                self.cols.append(col)

        sd.dropna(inplace=True)

        #Normalize
        scaler = StandardScaler(with_mean=True,with_std=True,copy=False)
        for col in self.cols:
            scaler.fit(sd[col].to_frame())
            sd[col] = scaler.transform(sd[col].to_frame())

        self.df = sd.copy()

    def show_heatmap(self):

        if self.df is not None:
            data = self.df[self.cols]
            plt.figure(figsize=(20,20))
            plt.matshow(data.corr(),fignum=1)
            plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
            plt.gca().xaxis.tick_bottom()
            plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=12)
            plt.title("Feature Correlation Heatmap", fontsize=14)
            plt.show()
        else:
            print("No data to show")

    def show_raw_visualization(self):

        size = len(self.cols)
        ncol = 2

        fig, axes = plt.subplots(
            nrows=int(size/self.lags/ncol), ncols=ncol, figsize=(16,32), dpi=360, facecolor="w", edgecolor="k")

        colors = [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        date = pd.to_datetime(self.df.index, unit="ms", utc=True)

        for i in range(int(size/5)):
            key  = self.cols[i*5]
            c = colors[i % (len(colors))]
            data = self.df[key]
            data.index = date
            data.plot(
                ax=axes[i // ncol, i % ncol],
                color=c,
                title="{}".format(key),
                rot=25,
            )
        plt.tight_layout()
            