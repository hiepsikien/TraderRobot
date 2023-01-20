import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
plt.style.use("seaborn")
from random import randint
import utils as tu

TIMEFRAMES_IN_MS = {"15m":15*60*1000,"1h":60*60*1000,"1d":24*60*60*1000}
class FeatureManager():

    def __init__(self,target_col:str, window:int = 50) -> None:
        '''
        Initialize the instance

        Params:
        - window: a window to calculate
        - cols: list of features column
        - target_col: label column for classifier
        - df: dataframe hold both features and label data

        Return: none
        '''
        self.window = window
        self.cols = []
        self.target_col = target_col
        self.params = dict()

    def import_data(self,symbol: str, trade_timeframe: str, granular_timeframe: str):
        
        data_path = "../data/{}-{}.csv".format(symbol,trade_timeframe)
        self.df = pd.read_csv(
            data_path, 
            parse_dates=["Open Time"],
            index_col = "Open Time"
        )
        print("Imported {} with {} rows".format(data_path,len(self.df)))
        granular_data_path = "../nocommit/{}-{}.csv".format(symbol,granular_timeframe)
        self.granular_df = pd.read_csv(
            granular_data_path, 
            parse_dates=["Open Time"], 
            index_col = "Open Time"
        )
        print("Imported {} with {} rows".format(granular_data_path,len(self.granular_df)))
        self.trade_timeframe = trade_timeframe
        self.granular_timeframe = granular_timeframe

    def print_parameters(self):
        print("Features manager parameters:")
        print(self.params)

    def build_features(self,lags:int):
        '''
        Build the normalized features to feed to classifier
    
        Params:
            - lags: the lagging timeframe to be included
        Output: None. The self.df stored the processed output. self.df stores the feature columns.        
        '''

        self.params["lags"] = lags
        data = self.df

        open = data["Open"]
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volumn"]

        #Add lags feature
        features = [
            "returns",
            "dir",      
            "sma",      
            "boll",     
            "boll7",
            "boll14",
            "boll21",
            "min",      
            "min7",      
            "min14",
            "min21",
            "max",      
            "max7",      
            "max14",
            "max21",
            "mom",
            "mom7",      
            "mom14",
            "mom21",
            "vol",      
            "vol7",      
            "vol14",
            "vol21",
            "obv",      
            "mfi7",     
            "mfi14",
            "mfi21",
            "rsi7",      
            "rsi14",
            "rsi21",
            "adx7",      
            "adx14",
            "adx21",
            "roc",      
            "roc7",      
            "roc14",
            "roc21",
            "atr7",      
            "atr14",
            "atr21",
            "bop",      
            "ad",       
            "adosc",     
            "trange",    
            "ado",       
            "willr7",     
            "willr14",
            "willr21",
            "dx7",     
            "dx14",
            "dx21",
            "trix",     # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
            "ultosc",   # Ultimate Oscillator
            "high",
            "low",
            ]
        
        print("Calculating features...")
        data["returns"] = np.log(close/close.shift())
        data["dir"] = np.where(data["returns"] > 0,1,0)
        
        if (key:="sma") in features:
            data[key] = close.rolling(self.window).mean() - close.rolling(150).mean()
        if (key:="boll") in features:
            data[key] = (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()
        if (key:="boll7") in features:
            data[key] = (close - close.rolling(7).mean()) / close.rolling(7).std()
        if (key:="boll14") in features:
            data[key] = (close - close.rolling(14).mean()) / close.rolling(14).std()
        if (key:="boll21") in features:
            data[key] = (close - close.rolling(21).mean()) / close.rolling(21).std()
        if (key:="min") in features:
            
            data[key] = close.rolling(self.window).min() / close - 1
        if (key:="min7") in features:
            data[key] = close.rolling(7).min() / close - 1
        if (key:="min14") in features:
            data[key] = close.rolling(14).min() / close - 1
        if (key:="min21") in features:
            data[key] = close.rolling(21).min() / close - 1
        if (key:="max") in features:
            data[key] = close.rolling(self.window).max() / close - 1
        if (key:="max7") in features:
            data[key] = close.rolling(7).max() / close - 1
        if (key:="max14") in features:
            data[key] = close.rolling(14).max() / close - 1
        if (key:="max21") in features:
            data[key] = close.rolling(21).max() / close - 1
        if (key:="mom") in features:
            data[key] = ta.MOM(close, timeperiod=10)                    # type: ignore
        if (key:="mom7") in features:
            data[key] = ta.MOM(close, timeperiod=7)                     # type: ignore
        if (key:="mom14") in features:
            data[key] = ta.MOM(close, timeperiod=14)                    # type: ignore
        if (key:="mom21") in features:
            data[key] = ta.MOM(close, timeperiod=21)                    # type: ignore
        if (key:="vol") in features:
            data[key] = data["returns"].rolling(self.window).std()
        if (key:="vol7") in features:
            data[key] = data["returns"].rolling(7).std()
        if (key:="vol14") in features:
            data[key] = data["returns"].rolling(14).std()
        if (key:="vol21") in features:
            data[key] = data["returns"].rolling(21).std()
        if (key:="obv") in features:
            data[key] = ta.OBV(close,volume)                            # type: ignore
        if (key:="mfi7") in features:
            data[key] = ta.MFI(high, low, close, volume, timeperiod=7)  # type: ignore
        if (key:="mfi14") in features:
            data[key] = ta.MFI(high, low, close, volume, timeperiod=14) # type: ignore
        if (key:="mfi21") in features:
            data[key] = ta.MFI(high, low, close, volume, timeperiod=21) # type: ignore
        if (key:="rsi7") in features:
            data[key] = ta.RSI(close, timeperiod=7)                     # type: ignore
        if (key:="rsi14") in features:
            data[key] = ta.RSI(close, timeperiod=14)                    # type: ignore
        if (key:="rsi21") in features:
            data[key] = ta.RSI(close, timeperiod=21)                    # type: ignore
        if (key:="adx7") in features:
            data[key] = ta.ADX(high, low, close, timeperiod=7)          # type: ignore
        if (key:="adx14") in features:
            data[key] = ta.ADX(high, low, close, timeperiod=14)         # type: ignore
        if (key:="adx21") in features:
            data[key] = ta.ADX(high, low, close, timeperiod=21)         # type: ignore
        if (key:="roc") in features:
            data[key] = ta.ROC(close, timeperiod=10)                    # type: ignore
        if (key:="roc7") in features:
            data[key] = ta.ROC(close, timeperiod=7)                     # type: ignore
        if (key:="roc14") in features:
            data[key] = ta.ROC(close, timeperiod=14)                    # type: ignore
        if (key:="roc21") in features:
            data[key] = ta.ROC(close, timeperiod=21)                    # type: ignore
        if (key:="atr7") in features:
            data[key] = ta.ATR(high, low, close, timeperiod=7)          # type: ignore
        if (key:="atr14") in features:
            data[key] = ta.ATR(high, low, close, timeperiod=14)         # type: ignore
        if (key:="atr21") in features:
            data[key] = ta.ATR(high, low, close, timeperiod=21)         # type: ignore
        if (key:="bop") in features:
            data[key] = ta.BOP(open, high, low, close)                  # type: ignore
        if (key:="ad") in features:
            data[key] =ta.AD(high, low, close, volume)                  # type: ignore
        if (key:="adosc") in features:
            data[key] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10) # type: ignore
        if (key:="trange") in features:
            data[key] = ta.TRANGE(high, low, close)                                     # type: ignore
        if (key:="ado") in features:
            data[key] = ta.APO(close, fastperiod=12,slowperiod=26, matype=0)            # type: ignore
        if (key:="willr7") in features:
            data[key] = ta.WILLR(high, low, close, timeperiod=7)                        # type: ignore
        if (key:="willr14") in features:
            data[key] = ta.WILLR(high, low, close, timeperiod=14)                       # type: ignore
        if (key:="willr21") in features:
            data[key] = ta.WILLR(high, low, close, timeperiod=21)                       # type: ignore
        if (key:="dx7") in features:
            data[key] = ta.DX(high, low, close, timeperiod=7)                           # type: ignore
        if (key:="dx14") in features:
            data[key] = ta.DX(high, low, close, timeperiod=14)                          # type: ignore
        if (key:="dx21") in features:
            data[key] = ta.DX(high, low, close, timeperiod=21)                          # type: ignore
        if (key:="trix") in features:
            data[key] = ta.TRIX(close, timeperiod=30)                                   # type: ignore
        if (key:="ultosc") in features:
            data[key] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)  # type: ignore
        if (key:="high") in features:
            data[key] = high/close -1
        if (key:="low") in features:
            data[key] = low/close -1

        data.dropna(inplace=True)
        
        print("\nAdding feature:", end=" ")
        self.cols = []
        for f in features:
            print("{},".format(f), end=" ")
            for lag in range(1,lags + 1):
                col = "{}_lag_{}".format(f,lag)
                data[col] = data[f].shift(lag)
                self.cols.append(col)
        print("")
        data.dropna(inplace=True)

        #Normalize
        scaler = StandardScaler(with_mean=True,with_std=True,copy=False)
        print("\nNormalizing feature:", end=" ")
        for col in self.cols:
            print("{},".format(col),end=" ")
            scaler.fit(data[col].to_frame())
            data[col] = scaler.transform(data[col].to_frame())

        print("\nTotal {} features added.".format(len(self.cols)))

    def calculate_tp_or_sl(self,row,i:int,is_long:bool):    
        '''
        Calculate if a take profit or stop loss event happen first

        Params:
        - row: a Dataframe row
        - is_long: is that a long trade or not

        Return: a tupple reprent take profit and stop loss as boolean,
        example, (True, False) means it is a take profit, (False, False) mean not take profit
        not stop loss either.

        '''

        trade_timeframe_in_ms = TIMEFRAMES_IN_MS[self.trade_timeframe]
        
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
                            start_time=int(row.name) + i * trade_timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* trade_timeframe_in_ms,
                            target_price = row["long_take_profit"],
                            granular_data=self.granular_df
                        )
                        first_sl_index = tu.first_time_go_below_price(
                            start_time=int(row.name) + i * trade_timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* trade_timeframe_in_ms,
                            target_price = row["long_stop_loss"],
                            granular_data=self.granular_df
                        )
                    else:
                        first_tp_index = tu.first_time_go_below_price(
                            start_time=int(row.name) + i * trade_timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* trade_timeframe_in_ms,
                            target_price = row["short_take_profit"],
                            granular_data=self.granular_df
                        )
                        first_sl_index = tu.first_time_go_above_price(
                            start_time=int(row.name) + i * trade_timeframe_in_ms,
                            end_time=int(row.name) + (i+1)* trade_timeframe_in_ms,
                            target_price = row["short_stop_loss"],
                            granular_data=self.granular_df
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

    def prepare_trade_forward_data(self,take_profit_rate:float = 0.05, stop_loss_rate:float = 0.025, max_duration:int = 7):
        '''   
        Add trade signal as long, short, no trade that can make a take profit close.
        
        Params:
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

        self.params["trade_timeframe"] = self.trade_timeframe
        self.params["take_profit_rate"] = take_profit_rate
        self.params["stop_loss_rate"] = stop_loss_rate
        self.params["max_duration"] = max_duration

        df = self.df.copy()
        df["long_take_profit"] = df["Close"]*(1+take_profit_rate)
        df["long_stop_loss"] = df["Close"]*(1-stop_loss_rate)
        df["short_take_profit"] = df["Close"]*(1-take_profit_rate)
        df["short_stop_loss"] = df["Close"]*(1+stop_loss_rate)
        df["long_decision_forward_0"] = 0
        df["short_decision_forward_0"] = 0
        trade_signal_str = "trade_signal"
        trade_return_str = "trade_return"
        df[trade_signal_str] = 0
        df[trade_return_str] = 0
        
        print("Scanning {} future timeframes to build trade signal: ".format(max_duration))
        for i in range(1,max_duration+1):
            print("{},".format(i), end=" ")

            df["High_forward_{}".format(i)] = df["High"].shift(-i)
            df["Low_forward_{}".format(i)] = df["Low"].shift(-i)
    
            long_str = "long_decision_forward_{}".format(i)
            short_str = "short_decision_forward_{}".format(i)

            #Temporarily set all open as closed
            df[long_str] = 2
            df[short_str] = 2

            open_long_cond = df["long_decision_forward_{}".format(i-1)]==0
            open_short_condition = df["short_decision_forward_{}".format(i-1)]==0

            # Compute decision to be take profit or stop loss
            df[["ol_tp","ol_sl"]]=df.apply(lambda row: self.calculate_tp_or_sl(
                row=row,
                i=i,
                is_long=True,
                ),
                result_type = "expand",
                axis=1
            )

            ol_tp_cond = df["ol_tp"]
            ol_sl_cond = df["ol_sl"]
            

            # Open long positions not hit either take-profit or stop-loss, leave it open
            df.loc[open_long_cond & ~ol_tp_cond & ~ol_sl_cond, long_str] = 0
            
            # Open long positions hit stop-loss
            df.loc[open_long_cond & ol_sl_cond & ~ol_tp_cond,long_str] = -1

            # Open long positions hit take-profit
            df.loc[open_long_cond & ol_tp_cond & ~ol_sl_cond,long_str] = 1
            df.loc[open_long_cond & ol_tp_cond & ~ol_sl_cond,trade_signal_str] = 1
            df.loc[open_long_cond & ol_tp_cond & ~ol_sl_cond,trade_return_str] = np.log(1 + take_profit_rate)      
                
            # Compute future outcome for open short positions
            df[["os_tp","os_sl"]] = df.apply(lambda row: self.calculate_tp_or_sl(
                row=row,
                i=i,
                is_long=False
                ),
                result_type = "expand",
                axis=1
            )
            os_tp_cond = df["os_tp"]
            os_sl_cond = df["os_sl"]
            
            # Open short positions not hit either take-profit or stop-loss, leave it open
            df.loc[open_short_condition & ~os_tp_cond & ~os_sl_cond, short_str] = 0

            # Open short positions hit stop-loss
            df.loc[open_short_condition & os_sl_cond & ~os_tp_cond,short_str] = -1

            # Open short positions hit take-profit
            df.loc[open_short_condition & os_tp_cond & ~os_sl_cond, short_str] = 1
            df.loc[open_short_condition & os_tp_cond & ~os_sl_cond,trade_signal_str] = 2
            df.loc[open_short_condition & os_tp_cond & ~os_sl_cond,trade_return_str] = np.log(1 + take_profit_rate)

            self.df[trade_signal_str] = df[trade_signal_str]
            self.df[trade_return_str] = df[trade_return_str]
            self.df.dropna(inplace=True)

        print("\nLabel producing completed. \n Value counts:")
        print(self.df[self.target_col].value_counts())
            

    def plot_features_correlation(self):
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

    def plot_features(self,lags:int):
        '''Show the visualiazation of features value along time
        
        Params:
        - lags: the lagging value. We dont want to show lags

        Returns: None
        '''
        size = len(self.cols)
        ncol = 2

        fig, axes = plt.subplots(
            nrows=int(size/lags/ncol), ncols=ncol, figsize=(16,32), dpi=180, facecolor="w", edgecolor="k")

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
            