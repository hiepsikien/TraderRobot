import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
plt.style.use("seaborn")
from random import randint
import tr_utils as tu

class FeatureManager():

    def __init__(self,window = 50, lags =5) -> None:
        self.window = window
        self.cols = []
        self.df = None

    def build_feature(self,data,lags):
        '''
        Build the normalized features to feed to classifier
    
        Params:
            - data: the dataframe consist original data
            - lags: the lagging timeframe to be included
        Output: None. The self.df stored the processed output. self.df stores the feature columns.        
        '''
        sd = data.copy()

        open = sd["Open"]
        close = sd["Close"]
        high = sd["High"]
        low = sd["Low"]
        volume = sd["Volumn"]

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

        sd["returns"] = np.log(close/close.shift())
        sd["dir"] = np.where(sd["returns"] > 0,1,0)
        
        if (key:="sma") in features:
            sd[key] = close.rolling(self.window).mean() - close.rolling(150).mean()
        if (key:="boll") in features:
            sd[key] = (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()
        if (key:="boll7") in features:
            sd[key] = (close - close.rolling(7).mean()) / close.rolling(7).std()
        if (key:="boll14") in features:
            sd[key] = (close - close.rolling(14).mean()) / close.rolling(14).std()
        if (key:="boll21") in features:
            sd[key] = (close - close.rolling(21).mean()) / close.rolling(21).std()
        if (key:="min") in features:
            sd[key] = close.rolling(self.window).min() / close - 1
        if (key:="min7") in features:
            sd[key] = close.rolling(7).min() / close - 1
        if (key:="min14") in features:
            sd[key] = close.rolling(14).min() / close - 1
        if (key:="min21") in features:
            sd[key] = close.rolling(21).min() / close - 1
        if (key:="max") in features:
            sd[key] = close.rolling(self.window).max() / close - 1
        if (key:="max7") in features:
            sd[key] = close.rolling(7).max() / close - 1
        if (key:="max14") in features:
            sd[key] = close.rolling(14).max() / close - 1
        if (key:="max21") in features:
            sd[key] = close.rolling(21).max() / close - 1
        if (key:="mom") in features:
            sd[key] = ta.MOM(close, timeperiod=10)
        if (key:="mom7") in features:
            sd[key] = ta.MOM(close, timeperiod=7)
        if (key:="mom14") in features:
            sd[key] = ta.MOM(close, timeperiod=14)
        if (key:="mom21") in features:
            sd[key] = ta.MOM(close, timeperiod=21)
        if (key:="vol") in features:
            sd[key] = sd["returns"].rolling(self.window).std()
        if (key:="vol7") in features:
            sd[key] = sd["returns"].rolling(7).std()
        if (key:="vol14") in features:
            sd[key] = sd["returns"].rolling(14).std()
        if (key:="vol21") in features:
            sd[key] = sd["returns"].rolling(21).std()
        if (key:="obv") in features:
            sd[key] = ta.OBV(close,volume)
        if (key:="mfi7") in features:
            sd[key] = ta.MFI(high, low, close, volume, timeperiod=7)
        if (key:="mfi14") in features:
            sd[key] = ta.MFI(high, low, close, volume, timeperiod=14)
        if (key:="mfi21") in features:
            sd[key] = ta.MFI(high, low, close, volume, timeperiod=21)
        if (key:="rsi7") in features:
            sd[key] = ta.RSI(close, timeperiod=7)
        if (key:="rsi14") in features:
            sd[key] = ta.RSI(close, timeperiod=14)
        if (key:="rsi21") in features:
            sd[key] = ta.RSI(close, timeperiod=21)
        if (key:="adx7") in features:
            sd[key] = ta.ADX(high, low, close, timeperiod=7)
        if (key:="adx14") in features:
            sd[key] = ta.ADX(high, low, close, timeperiod=14)
        if (key:="adx21") in features:
            sd[key] = ta.ADX(high, low, close, timeperiod=21)
        if (key:="roc") in features:
            sd[key] = ta.ROC(close, timeperiod=10)
        if (key:="roc7") in features:
            sd[key] = ta.ROC(close, timeperiod=7)
        if (key:="roc14") in features:
            sd[key] = ta.ROC(close, timeperiod=14)
        if (key:="roc21") in features:
            sd[key] = ta.ROC(close, timeperiod=21)
        if (key:="atr7") in features:
            sd[key] = ta.ATR(high, low, close, timeperiod=7)
        if (key:="atr14") in features:
            sd[key] = ta.ATR(high, low, close, timeperiod=14)
        if (key:="atr21") in features:
            sd[key] = ta.ATR(high, low, close, timeperiod=21)
        if (key:="bop") in features:
            sd[key] = ta.BOP(open, high, low, close)
        if (key:="ad") in features:
            sd[key] =ta.AD(high, low, close, volume)
        if (key:="adosc") in features:
            sd[key] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        if (key:="trange") in features:
            sd[key] = ta.TRANGE(high, low, close)
        if (key:="ado") in features:
            sd[key] = ta.APO(close, fastperiod=12, slowperiod=26, matype=0)
        if (key:="willr7") in features:
            sd[key] = ta.WILLR(high, low, close, timeperiod=7)
        if (key:="willr14") in features:
            sd[key] = ta.WILLR(high, low, close, timeperiod=14)
        if (key:="willr21") in features:
            sd[key] = ta.WILLR(high, low, close, timeperiod=21)
        if (key:="dx7") in features:
            sd[key] = ta.DX(high, low, close, timeperiod=7)
        if (key:="dx14") in features:
            sd[key] = ta.DX(high, low, close, timeperiod=14)
        if (key:="dx21") in features:
            sd[key] = ta.DX(high, low, close, timeperiod=21)
        if (key:="trix") in features:
            sd[key] = ta.TRIX(close, timeperiod=30)
        if (key:="ultosc") in features:
            sd[key] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        sd.dropna(inplace=True)
        
        for f in features:
            for lag in range(1,lags + 1):
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

    def calculate_tp_or_sl(self,row,i,is_long,granular_data,timeframe_in_ms):    
        '''
        Calculate if a take profit, stop loss event happen
        '''
        
        if is_long:
            open_cond = row["long_decision_forward_{}".format(i-1)]==0
            tp_cond = row["High_forward_{}".format(i)] >= row["long_take_profit"]
            sl_cond = row["Low_forward_{}".format(i)] <= row["long_stop_loss"]
            
        else:
            open_cond = row["short_decision_forward_{}".format(i-1)]==0
            tp_cond = row["Low_forward_{}".format(i)] <= row["short_take_profit"]
            sl_cond = row["High_forward_{}".format(i)] >= row["short_stop_loss"]
          
        if open_cond:
            if tp_cond:                                         #  If take-profit hitted
                if sl_cond:
                    if is_long:
                        first_tp_index = tu.first_time_go_above_price(
                            start_time=int(row.name) + i * timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* timeframe_in_ms,
                            target_price = row["long_take_profit"],
                            granular_data=granular_data
                        )
                        first_sl_index = tu.first_time_go_below_price(
                            start_time=int(row.name) + i * timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* timeframe_in_ms,
                            target_price = row["long_stop_loss"],
                            granular_data=granular_data
                        )
                    else:
                        first_tp_index = tu.first_time_go_below_price(
                            start_time=int(row.name) + i * timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* timeframe_in_ms,
                            target_price = row["short_take_profit"],
                            granular_data=granular_data
                        )
                        first_sl_index = tu.first_time_go_above_price(
                            start_time=int(row.name) + i * timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* timeframe_in_ms,
                            target_price = row["short_stop_loss"],
                            granular_data=granular_data
                        )                                     #  stop-loss also hitted    
                    if first_tp_index < first_sl_index:         #  Granular says it is take-profit
                        return [True, False]
                    elif first_tp_index > first_sl_index:       #   Granular says it is take-profit
                        return [False, True]
                    else:                                       #   Granular can not tell, use random
                        ran_bool = (randint(0,99) < 50)
                        return [ran_bool, ~ran_bool]
                else:   # stop-loss not hitted
                    return [True, False]
            else:
                if sl_cond: #if only stop-loss hitted
                    return [False, True]
                else:           #if both not hitted
                    return [False, False]
        else:
            return [False, False]

    def prepare_trade_forward_data(self, data, granular_data,timeframe_in_ms, take_profit_rate = 0.05, stop_loss_rate = 0.025, max_duration = 7):
        '''   
        Add trade signal as long, short, no trade that can make a take profit close.
        
        Params:
        - data: the input data as dataframe, will be modified.
        - granular_data: the shorter timeframe data to be used for decide takeprofit or stoploss happen first
        - timeframe_in_ms: the timeframe used
        - take_profit_rate: take-profit as percentage
        - stop_loss_rate: stop-loss as percentage
        - max_duration: the longest duration for a position 

        Returns: No returns, input data is modified. Important added columns as belows:.
        - trade_signal: 0 as no trade, 1 as long, 2 as short
        - trade_resturn: equal to take_profit rate or 0
        - long_decision_forward_{i}: for long position, at future timeframe i, 1: take profit, -1: stop loss, 0: keep open, 2: closed 
        - short_decision_forward_{i}: similar as long_ but for short position
        '''
        data["long_take_profit"] = data["Close"]*(1+take_profit_rate)
        data["long_stop_loss"] = data["Close"]*(1-stop_loss_rate)
        data["short_take_profit"] = data["Close"]*(1-take_profit_rate)
        data["short_stop_loss"] = data["Close"]*(1+stop_loss_rate)
        data["long_decision_forward_0"] = 0
        data["short_decision_forward_0"] = 0
        trade_signal_str = "trade_signal"
        trade_return_str = "trade_return"
        data[trade_signal_str] = 0
        data[trade_return_str] = 0
        
        for i in range(1,max_duration+1):
            print("Processing {}/{}".format(i,max_duration))

            data["High_forward_{}".format(i)] = data["High"].shift(-i)
            data["Low_forward_{}".format(i)] = data["Low"].shift(-i)
    
            long_str = "long_decision_forward_{}".format(i)
            short_str = "short_decision_forward_{}".format(i)

            #Temporarily set all open as closed
            data[long_str] = 2
            data[short_str] = 2

            open_long_cond = data["long_decision_forward_{}".format(i-1)]==0
            open_short_condition = data["short_decision_forward_{}".format(i-1)]==0

            # Compute decision to be take profit or stop loss
            data[["ol_tp","ol_sl"]]=data.apply(lambda row: self.calculate_tp_or_sl(
                row=row,
                i=i,
                is_long=True,
                granular_data=granular_data,
                timeframe_in_ms=timeframe_in_ms
                ),
                result_type = "expand",
                axis=1
            )

            ol_tp_cond = data["ol_tp"]
            ol_sl_cond = data["ol_sl"]
            

            # Open long positions not hit either take-profit or stop-loss, leave it open
            data.loc[open_long_cond & ~ol_tp_cond & ~ol_sl_cond, long_str] = 0
            
            # Open long positions hit stop-loss
            data.loc[open_long_cond & ol_sl_cond & ~ol_tp_cond,long_str] = -1

            # Open long positions hit take-profit
            data.loc[open_long_cond & ol_tp_cond & ~ol_sl_cond,long_str] = 1
            data.loc[open_long_cond & ol_tp_cond & ~ol_sl_cond,trade_signal_str] = 1
            data.loc[open_long_cond & ol_tp_cond & ~ol_sl_cond,trade_return_str] = np.log(1 + take_profit_rate)      
                
            # Compute future outcome for open short positions
            data[["os_tp","os_sl"]] = data.apply(lambda row: self.calculate_tp_or_sl(
                row=row,
                i=i,
                is_long=False,
                granular_data=granular_data,
                timeframe_in_ms=timeframe_in_ms
                ),
                result_type = "expand",
                axis=1
            )
            os_tp_cond = data["os_tp"]
            os_sl_cond = data["os_sl"]
            
            # Open short positions not hit either take-profit or stop-loss, leave it open
            data.loc[open_short_condition & ~os_tp_cond & ~os_sl_cond, short_str] = 0

            # Open short positions hit stop-loss
            data.loc[open_short_condition & os_sl_cond & ~os_tp_cond,short_str] = -1

            # Open short positions hit take-profit
            data.loc[open_short_condition & os_tp_cond & ~os_sl_cond, short_str] = 1
            data.loc[open_short_condition & os_tp_cond & ~os_sl_cond,trade_signal_str] = 2
            data.loc[open_short_condition & os_tp_cond & ~os_sl_cond,trade_return_str] = np.log(1 + take_profit_rate)


    def show_heatmap(self):
        ''' Show the correlation between pairs of features
        '''
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
        '''Show the visualiazation of features value along time
        '''
        size = len(self.cols)
        ncol = 2

        fig, axes = plt.subplots(
            nrows=int(size/self.lags/ncol), ncols=ncol, figsize=(16,32), dpi=180, facecolor="w", edgecolor="k")

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
            