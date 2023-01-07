import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.style.use("seaborn")

class FeatureManager():

    def __init__(self,window = 50, lags =5) -> None:
        self.window = window
        self.lags = lags
        self.cols = []
        self.df = None

    def build_feature(self,data):

        self.df = data.copy()
        self.df["returns"] = np.log(data["Close"]/data["Close"].shift())
        self.df["dir"] = np.where(self.df["returns"] > 0,1,0)
        self.df["sma"] = self.df["Close"].rolling(self.window).mean() - self.df["Close"].rolling(150).mean()
        self.df["boll"] = (self.df["Close"] - self.df["Close"].rolling(self.window).mean()) / self.df["Close"].rolling(self.window).std()
        self.df["min"] = self.df["Close"].rolling(self.window).min() / self.df["Close"] - 1
        self.df["max"] = self.df["Close"].rolling(self.window).max() / self.df["Close"] - 1
        self.df["mom"] = self.df["returns"].rolling(3).mean()
        self.df["vol"] = self.df["returns"].rolling(self.window).std()
        
        self.df.dropna(inplace=True)

        #Add lags feature
        features = ["dir","sma", "boll", "min", "max", "mom", "vol"]
        for f in features:
            for lag in range(1,self.lags + 1):
                col = "{}_lag_{}".format(f,lag)
                self.df[col] = self.df[f].shift(lag)
                self.cols.append(col)

        self.df.dropna(inplace=True)

        #Normalize
        scaler = StandardScaler(with_mean=True,with_std=True,copy=False)
        for col in self.cols:
            scaler.fit(self.df[col].to_frame())
            self.df[col] = scaler.transform(self.df[col].to_frame())

        