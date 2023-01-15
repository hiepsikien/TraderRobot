import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
plt.style.use("seaborn")
from random import randint

class FeatureManager():

    def __init__(self,window = 50, lags =5) -> None:
        self.window = window
        self.cols = []
        self.df = None

    def build_feature(self,data,lags):

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

    def first_time_go_above_price(self,start_time,end_time,target_price,granular_data):
        ''' 
        Find the time in ms that take profit and stop loss condition satisified in the granular data for short trade
        Return time in unix utc milisecond.
        '''
        lookup_data = granular_data.iloc[
            (granular_data.index>=start_time) & 
            (granular_data.index<end_time)].copy()

        first_index = lookup_data.loc[lookup_data["High"]>=target_price].index.min()
        return first_index

    def first_time_go_below_price(self,start_time,end_time,target_price,granular_data):
        ''' 
        Find the time in ms that take profit and stop loss condition satisified in the granular data for short trade
        Return time in unix utc milisecond.
        '''
        lookup_data = granular_data.iloc[
            (granular_data.index>=start_time) & 
            (granular_data.index<end_time)].copy()

        first_index = lookup_data.loc[lookup_data["Low"]<target_price].index.min()
        return first_index

    def calculate_tp_or_sl(self,row,i,is_long,granular_data,timeframe_in_ms):    
        '''Calculate if a take profit, stop loss event happen
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
                        first_tp_index = self.first_time_go_above_price(
                            start_time=int(row.name) + i * timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* timeframe_in_ms,
                            target_price = row["long_take_profit"],
                            granular_data=granular_data
                        )
                        first_sl_index = self.first_time_go_below_price(
                            start_time=int(row.name) + i * timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* timeframe_in_ms,
                            target_price = row["long_stop_loss"],
                            granular_data=granular_data
                        )
                    else:
                        first_tp_index = self.first_time_go_below_price(
                            start_time=int(row.name) + i * timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* timeframe_in_ms,
                            target_price = row["short_take_profit"],
                            granular_data=granular_data
                        )
                        first_sl_index = self.first_time_go_above_price(
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
        # Compute future will happen outcome and and trade signal
        # Will update long_decision_forward to 1: take profit, -1: stop loss, 0: keep open, 2: closed 
        # Will update trade_signal to 1: long, 2: short, 0: no trade
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
            