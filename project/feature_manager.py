from pyexpat import features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from talib import abstract
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
plt.style.use("seaborn")
from random import randint
import tr_utils as tu
import config as cf


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

    def import_trading_data(self,symbol: str, trade_timeframe: str):
        '''Import trading data
        '''
        data_path = "../data/{}-{}.csv".format(symbol,trade_timeframe)
        self.df = pd.read_csv(
            data_path, 
            parse_dates=["Open Time"],
            index_col = "Open Time"
        )
        print("Imported trading data from {} with {} rows".format(data_path,len(self.df)))
        self.trade_timeframe = trade_timeframe
    
    def import_macro_data(self,symbol: str, macro_timeframe: str):
        '''
        Import macro data, one level higher than trading data.
        Use it to calculate technical analysis indicator
        '''
        macro_data_path = "../data/{}-{}.csv".format(symbol,macro_timeframe)
        self.macro_df = pd.read_csv(
            macro_data_path, 
            parse_dates=["Open Time"], 
            index_col = "Open Time"
        )
        print("Imported macro data from {} with {} rows".format(macro_data_path,len(self.macro_df)))
        self.macro_timeframe = macro_timeframe

    def import_super_data(self,symbol: str, super_timeframe: str):
        '''
        Import super macro data, one level higher than trading data.
        Use it to calculate technical analysis indicator
        '''
        data_path = "../data/{}-{}.csv".format(symbol,super_timeframe)
        self.super_df = pd.read_csv(
            data_path, 
            parse_dates=["Open Time"], 
            index_col = "Open Time"
        )
        print("Imported super macro data from {} with {} rows".format(data_path,len(self.super_df)))
        self.super_timeframe = super_timeframe

    def import_granular_data(self,symbol: str, granular_timeframe: str):
        '''Import most detail data, used to check stoploss and take profit hit
        '''
        granular_data_path = "../nocommit/{}-{}.csv".format(symbol,granular_timeframe)
        self.granular_df = pd.read_csv(
            granular_data_path, 
            parse_dates=["Open Time"], 
            index_col = "Open Time"
        )
        print("Imported granular data from {} with {} rows".format(granular_data_path,len(self.granular_df)))
        self.granular_timeframe = granular_timeframe

    def calculate_external_features(self,features:list[str]):
        print("Calculating external features ...")
        if (key:="hashrate") in features:
            hashrate_path = "../data/bitcoin_hashrate.csv"
            self.import_new_feature(
                filename=hashrate_path,
                index_col = "Date",
                import_label="hashrate",
                new_label=key
            )
        
        if (key:="fed_rate") in features:
            fed_rate_path = "../data/fed_rate.csv"
            self.import_new_feature( 
                filename=fed_rate_path,
                index_col = "timestamp", 
                import_label="value",
                new_label=key
        )
        if (key:="google_trend") in features:
            bitcoin_google_trend_path = "../data/bitcoin_google_trend.csv"
            self.import_new_feature( 
                filename=bitcoin_google_trend_path,
                index_col = "timestamp", 
                import_label="Bitcoin",
                new_label=key
            )

        if (key:="gold") in features:
            gold_path = "../data/gold.csv"
            self.import_new_feature(
                filename=gold_path,
                index_col = "timestamp",
                import_label="Close/Last",
                new_label=key
            )
        
        if (key:="nasdaq") in features:
            nasdaq_path = "../data/nasdaq.csv"
            self.import_new_feature(
                filename=nasdaq_path,
                index_col = "timestamp",
                import_label="Close/Last",
                new_label=key
            )

        if (key:="sp500") in features:
            sp500_path = "../data/sp500.csv"
            self.import_new_feature(
                filename=sp500_path,
                index_col = "timestamp",
                import_label="Close",
                new_label=key
            )

    def import_new_feature(self,filename:str, index_col: str, import_label: str, new_label: str):
        df = pd.read_csv(filename, index_col=index_col).sort_index(ascending = True)
        self.df[new_label] = df[import_label]
        self.df[new_label].fillna(method="ffill", inplace = True)
        
    def print_parameters(self):
        print("Features manager parameters:")
        print(self.params)

    def build_features(self,lags:int,macro_lags:int,super_lags:int, 
        features:list[str], macro_features:list[str], super_features:list, scaler:str = "MaxAbs"):
        '''
        Build the normalized features to feed to classifier
    
        Params:
            - lags: the lagging timeframe to be included
        Output: None. The self.df stored the processed output. self.df stores the feature columns.        
        '''
        self.params["lags"] = lags
        self.params["scaler"] = scaler
            
        # Import external features
        self.calculate_external_features(features=features)

        # Calculating TA features for trading timeframe
        self.calculate_technical_analysis_features(features=features,level="trade")
        self.calculate_technical_analysis_features(features=macro_features,level="macro")
        self.calculate_technical_analysis_features(features=super_features,level="super")

        # Calculate candle stick features for trading timeframe
        self.calculate_candle_sticks(features=features,level="trade")
        self.calculate_candle_sticks(features=macro_features,level="macro")
        self.calculate_candle_sticks(features=super_features,level="super")

        # Adding lags
        self.add_features_with_lags(features=features,lags=lags,level="trade")
        self.add_features_with_lags(features=macro_features,lags=macro_lags,level="macro")
        self.add_features_with_lags(features=super_features,lags=super_lags,level="super")
 
        # macro features with trading features
        self.merge_features_all_level()
       
        # Drop na before normalize
        self.df.dropna(inplace=True)

        # Normalize
        self.normalize_features(scaler = scaler)

        print("\nTotal {} features added.".format(len(self.cols)))

    def merge_features_all_level(self):
        merged_pd = pd.concat([self.df,self.macro_df[self.macro_cols],self.super_df[self.super_cols]],axis=1)
        
        for col in self.macro_cols:
            merged_pd[col].fillna(method="ffill",inplace=True)

        for col in self.super_cols:
            merged_pd[col].fillna(method="ffill",inplace=True)

        self.df = merged_pd
        self.cols = self.cols + self.macro_cols + self.super_cols


    def add_features_with_lags(self,features:list[str],lags:int,level:str):
        
        print("\nAdding features for {} timeframe with lags {}:".format(level,lags), end=" ")

        match level:
            case "super":
                data = self.super_df.copy()
            case "macro":
                data = self.macro_df.copy()
            case "trade":
                data = self.df.copy()
            case other:
                raise ValueError("level must be one of: trading, macro, super")
        
        cols = []
        for f in features:
            print("{},".format(f), end=" ")
            for lag in range(1,lags + 1):
                col = "{}_{}_lag_{}".format(f,level,lag)
                data[col] = data[f].shift(lag)
                cols.append(col)
        print("")

        match level:
            case "super":
                self.super_df = data.copy()
                self.super_cols = cols
            case "macro":
                self.macro_df = data.copy()
                self.macro_cols = cols
            case "trade":
                self.df = data.copy()
                self.cols = cols
 
    def normalize_features(self,scaler=None):
        ''' Normalize features

            Params:
            -scaler: 'Standard' , 'MinMax', 'MaxAbs'

            Returns: None
        '''
        used_scaler = None
        match scaler:
            case "Standard":
                used_scaler = StandardScaler(with_mean=True,with_std=True,copy=False)
            case "MinMax":
                used_scaler = MinMaxScaler(copy = False)
            case "MaxAbs":
                used_scaler = MaxAbsScaler(copy = False)
            case other:
                raise ValueError("'scaler' must be one of 'Standard','MinMax','MaxAbs'")
            
        if used_scaler:
            print("\nNormalizing features with {}:".format(scaler), end=" ")
            for col in self.cols:
                print("{},".format(col),end=" ")
                used_scaler.fit(self.df[col].to_frame())
                self.df[col] = used_scaler.transform(self.df[col].to_frame())

    def calculate_candle_sticks(self,features:list[str], level:str):
        ''' 
        Calculate the candle stick patterns indicators

        Params:
        - features: list of features
        - level: one of 'trade', 'macro', 'super'

        Returns: None
        '''
        print("Calculating candlestick of {} timeframe".format(level))

        data = None
        match level:
            case "super":
                data = self.super_df.copy()
            case "macro":
                data = self.macro_df.copy()
            case "trade":
                data = self.df.copy()
            case other:
                raise ValueError("level must be one of: trading, macro, super")

        open = data["Open"]
        high = data["High"]
        close = data["Close"]
        low = data["Low"]

        all_indicators = [method for method in dir(abstract) if method.startswith('CDL')]
        
        for indicator in all_indicators:
            if indicator in features:
                data[str(indicator)] = getattr(abstract, indicator)(open,high,low,close)

        match level:
            case "super":
                self.super_df = data
            case "macro":
                self.macro_df = data
            case "trade":
                self.df = data

    def calculate_technical_analysis_features(self,features: list[str],level:str):
        ''' 
        Calculate the technical analysis not including candle stick
        
        Params:
        - features: list of features
        - level: one of 'trade', 'macro', 'super'

        Returns: None
        '''
        print("Calculating TA indicators of {} timeframe".format(level))

        data = None
        match level:
            case "super":
                data = self.super_df.copy()
            case "macro":
                data = self.macro_df.copy()
            case "trade":
                data = self.df.copy()
            case other:
                raise ValueError("Argument 'level' must be one of: 'trade', 'macro', 'super'")

        open = data["Open"]
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volumn"]

        data["returns"] = np.log(close/close.shift())
        data["dir"] = np.where(data["returns"] > 0,1,0)
         
        if (key:="sma_3_10") in features:
            data[key] = close.rolling(3).mean() - close.rolling(7).mean()
        
        if (key:="sma_7_30") in features:
            data[key] = close.rolling(7).mean() - close.rolling(14).mean()

        if (key:="sma_14_50") in features:
            data[key] = close.rolling(14).mean() - close.rolling(90).mean()

        if (key:="sma_28_90") in features:
            data[key] = close.rolling(28).mean() - close.rolling(90).mean()

        if (key:="cci7") in features:
            data[key] = ta.CCI(high,low,close,timeperiod = 7)       #type: ignore
        
        if (key:="cci14") in features:
            data[key] = ta.CCI(high,low,close,timeperiod = 14)      #type: ignore

        if (key:="cci30") in features:
            data[key] = ta.CCI(high,low,close,timeperiod = 30)      #type: ignore
        
        up_bb, mid_bb, low_bb = ta.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0) #type:ignore

        if (key:="up_bb") in features:
            data["up_bb"] = up_bb
        
        if (key:="low_bb") in features:
            data["low_bb"] = low_bb
            
        if (key:="min") in features:
            data[key] = close.rolling(self.window).min() / close - 1
        if (key:="min7") in features:
            data[key] = close.rolling(7).min() / close - 1
        if (key:="min14") in features:
            data[key] = close.rolling(14).min() / close - 1
        if (key:="min30") in features:
            data[key] = close.rolling(30).min() / close - 1
        if (key:="max") in features:
            data[key] = close.rolling(self.window).max() / close - 1
        if (key:="max7") in features:
            data[key] = close.rolling(7).max() / close - 1
        if (key:="max14") in features:
            data[key] = close.rolling(14).max() / close - 1
        if (key:="max30") in features:
            data[key] = close.rolling(30).max() / close - 1
        if (key:="mom") in features:
            data[key] = ta.MOM(close, timeperiod=10)                                                # type: ignore
        if (key:="mom7") in features:
            data[key] = ta.MOM(close, timeperiod=7)                                                 # type: ignore
        if (key:="mom14") in features:
            data[key] = ta.MOM(close, timeperiod=14)                                                # type: ignore
        if (key:="mom30") in features:
            data[key] = ta.MOM(close, timeperiod=30)                                                # type: ignore
        if (key:="vol") in features:
            data[key] = data["returns"].rolling(self.window).std()
        if (key:="vol7") in features:
            data[key] = data["returns"].rolling(7).std()
        if (key:="vol14") in features:
            data[key] = data["returns"].rolling(14).std()
        if (key:="vol30") in features:
            data[key] = data["returns"].rolling(30).std()
        if (key:="obv") in features:
            data[key] = ta.OBV(close,volume)                                                        # type: ignore
        if (key:="mfi7") in features:
            data[key] = ta.MFI(high, low, close, volume, timeperiod=7)                              # type: ignore
        if (key:="mfi14") in features:
            data[key] = ta.MFI(high, low, close, volume, timeperiod=14)                             # type: ignore
        if (key:="mfi30") in features:
            data[key] = ta.MFI(high, low, close, volume, timeperiod=30)                             # type: ignore
        if (key:="rsi7") in features:
            data[key] = ta.RSI(close, timeperiod=7)                                                 # type: ignore
        if (key:="rsi14") in features:
            data[key] = ta.RSI(close, timeperiod=14)                                                # type: ignore
        if (key:="rsi30") in features:
            data[key] = ta.RSI(close, timeperiod=30)                                                # type: ignore
        if (key:="rsi60") in features:
            data[key] = ta.RSI(close, timeperiod=60)                                                # type: ignore
        if (key:="rsi90") in features:
            data[key] = ta.RSI(close, timeperiod=90)                                                # type: ignore
        if (key:="rsi180") in features:
            data[key] = ta.RSI(close, timeperiod=180)                                                # type: ignore
        if (key:="adx7") in features:
            data[key] = ta.ADX(high, low, close, timeperiod=7)                                      # type: ignore
        if (key:="adx14") in features:
            data[key] = ta.ADX(high, low, close, timeperiod=14)                                     # type: ignore
        if (key:="adx30") in features:
            data[key] = ta.ADX(high, low, close, timeperiod=30)                                     # type: ignore
        if (key:="roc") in features:
            data[key] = ta.ROC(close, timeperiod=10)                                                # type: ignore
        if (key:="roc7") in features:
            data[key] = ta.ROC(close, timeperiod=7)                                                 # type: ignore
        if (key:="roc14") in features:
            data[key] = ta.ROC(close, timeperiod=14)                                                # type: ignore
        if (key:="roc30") in features:
            data[key] = ta.ROC(close, timeperiod=30)                                                # type: ignore
        if (key:="atr7") in features:
            data[key] = ta.ATR(high, low, close, timeperiod=7)                                      # type: ignore
        if (key:="atr14") in features:
            data[key] = ta.ATR(high, low, close, timeperiod=14)                                     # type: ignore
        if (key:="atr30") in features:
            data[key] = ta.ATR(high, low, close, timeperiod=30)                                     # type: ignore
        if (key:="bop") in features:
            data[key] = ta.BOP(open, high, low, close)                                              # type: ignore
        if (key:="ad") in features:
            data[key] =ta.AD(high, low, close, volume)                                              # type: ignore
        if (key:="adosc") in features:
            data[key] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)             # type: ignore
        if (key:="trange") in features:
            data[key] = ta.TRANGE(high, low, close)                                                 # type: ignore
        if (key:="ado") in features:
            data[key] = ta.APO(close, fastperiod=12,slowperiod=26, matype=0)                        # type: ignore
        if (key:="willr7") in features:
            data[key] = ta.WILLR(high, low, close, timeperiod=7)                                    # type: ignore
        if (key:="willr14") in features:
            data[key] = ta.WILLR(high, low, close, timeperiod=14)                                   # type: ignore
        if (key:="willr30") in features:
            data[key] = ta.WILLR(high, low, close, timeperiod=30)                                   # type: ignore
        if (key:="dx7") in features:
            data[key] = ta.DX(high, low, close, timeperiod=7)                                       # type: ignore
        if (key:="dx14") in features:
            data[key] = ta.DX(high, low, close, timeperiod=14)                                      # type: ignore
        if (key:="dx30") in features:
            data[key] = ta.DX(high, low, close, timeperiod=30)                                      # type: ignore
        if (key:="trix") in features:
            data[key] = ta.TRIX(close, timeperiod=30)                                               # type: ignore
        if (key:="ultosc") in features:
            data[key] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)  # type: ignore
        if (key:="high") in features:
            data[key] = high/close -1
        if (key:="low") in features:
            data[key] = low/close -1
        
        match level:
            case "super":
                self.super_df = data
            case "macro":
                self.macro_df = data
            case "trade":
                self.df = data
    '''
    def get_macro_open_time(self,current_time:int):
        maybe_in = None
        last_index = None
        for i in self.macro_df.index:
            if current_time >= int(i):
                maybe_in = True
                last_index = i 
            else:
                if maybe_in and (current_time < int(i)):
                    return last_index
        return None
    '''

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

        trade_timeframe_in_ms = cf.TIMEFRAMES_IN_MS[self.trade_timeframe]
        
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
            print("{}".format(i), end=" ")

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
        print(self.df[self.target_col].value_counts().sort_index())            

    def plot_trade_signal(self,dpi:int=240,save_to_file:bool = False):
        long = self.df.loc[self.df["trade_signal"] == 1]
        short = self.df.loc[self.df["trade_signal"] == 2]
        plt.figure(figsize = (20,10), dpi = dpi)
        plt.plot(self.df.index,self.df["Close"])
        plt.scatter(long.index,long["Close"],color="g",marker="^")      #type: ignore
        plt.scatter(short.index,short["Close"],color="r",marker="v")    #type: ignore
        if save_to_file:
            plt.savefig("../out/trade_signal.png")
        plt.show()

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
            