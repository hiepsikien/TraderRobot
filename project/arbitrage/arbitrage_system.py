import sys
sys.path.append("../")
import pandas as pd
from pandas import DataFrame
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
import talib
from typing import Dict, List
from numpy import NaN
from random import randint

TIMEFRAME_IN_MIN = {
    "1m":1,
    "5m":5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 4 * 60,
    "12h": 12 * 60,
    "1d": 24 * 60
}

AGGREGATED_DATA_PATH = "/media/andy/UbuntuDisk/CryptoData/traderrobot/Aggregated/"
TRADE_TIMEFRAME = "4h"
TRADE_TIMEFRAME_IN_MS = 4 * 60 * 60 * 1000
GRANULAR_TIMEFRAME = "5m"
MACD_FAST = 26
MACD_SLOW= 12
MACD_SIGNAL= 9
SHORT_MULTIPLES = 1
MEDIUM_MULTIPLES = 3
LONG_MULTIPLES = 8
HOLDING_PERIOD = 14
MARKET_CYCLE_SPEED = ["FAST","MEDIUM","SLOW"]
BASE_PAIR = "BTCUSDT"

def import_trading_data(timeframe:str, tickers:List[Dict]):
    dataframe_dict:Dict[str,DataFrame] = dict()
    print("...loading trading data...")
    for ticker in tickers:
        pair = ticker["symbol"]
        if str(pair).endswith("USDT"):
            print(pair, end=" ")
            filename = f"{AGGREGATED_DATA_PATH}futures/{timeframe}/{pair}-{timeframe}.h5"
            df = pd.read_hdf(filename)
            dataframe_dict[pair] = df
    return dataframe_dict

def import_granular_data(timeframe:str,tickers:List):
    granular_dataframe_dict: Dict[str,DataFrame] = dict()
    print("...loading granular data...")
    for ticker in tickers:
        pair = ticker["symbol"]
        if str(pair).endswith("USDT"):
            print(pair, end=" ")
            filename = f"{AGGREGATED_DATA_PATH}futures/{timeframe}/{pair}-{timeframe}.h5"
            gdf = pd.read_hdf(filename)
            granular_dataframe_dict[pair] = gdf
    return granular_dataframe_dict

def examine_data(timeframe:str,dfs:Dict[str,DataFrame]):
    len_list = []
    coverage_list = []
    last_list = []
    for pair in dfs.keys():
        len_list.append(len(dfs[pair]))
        must_length = (dfs[pair].index.max()-dfs[pair].index.min())/TIMEFRAME_IN_MIN[timeframe]*60*1000+1
        real_length = len(dfs[pair])
        coverage_list.append(real_length/must_length)
        last_list.append(dfs[pair].index.max())
    df = pd.DataFrame({"symbol": dfs.keys(), "length": len_list,\
        "coverage":coverage_list, "last":last_list})
    return df

def calculate_macd(pairs:List[str],dfs:Dict[str,DataFrame]):
    selected_pairs = pairs+[BASE_PAIR]
    for pair in selected_pairs:
        df = dfs[pair]
        df[f"{pair}_LOG_RETURN"] = np.log(df["Close"].pct_change()+1)
        
        _,_,df[f"{pair}_FAST_MACD"] = talib.MACD(df["Close"],fastperiod=MACD_FAST, \
            slowperiod=MACD_SLOW,signalperiod=MACD_SIGNAL)
        df[f"{pair}_FAST_MACD"] = df[f"{pair}_FAST_MACD"]/df["Close"]
        
        _,_,df[f"{pair}_MEDIUM_MACD"] = talib.MACD(df["Close"],fastperiod=MACD_FAST*MEDIUM_MULTIPLES,\
            slowperiod=MACD_SLOW*MEDIUM_MULTIPLES,signalperiod=MACD_SIGNAL*MEDIUM_MULTIPLES)
        df[f"{pair}_MEDIUM_MACD"] = df[f"{pair}_MEDIUM_MACD"]/df["Close"]
        
        _,_,df[f"{pair}_SLOW_MACD"] = talib.MACD(df["Close"],fastperiod=MACD_FAST*LONG_MULTIPLES,\
            slowperiod=MACD_SLOW*LONG_MULTIPLES,signalperiod=MACD_SIGNAL*LONG_MULTIPLES)
        df[f"{pair}_SLOW_MACD"] = df[f"{pair}_SLOW_MACD"]/df["Close"]
    
    # We need BTC price to draw graph
    BITCOIN_PRICE = dfs["BTCUSDT"]["Close"]
    BITCOIN_PRICE.name = "BITCOIN_PRICE"

    merged_df = pd.concat([dfs[symbol][[f"{symbol}_LOG_RETURN",\
        f"{symbol}_FAST_MACD",f"{symbol}_MEDIUM_MACD",f"{symbol}_SLOW_MACD"]]\
            for symbol in selected_pairs] + [BITCOIN_PRICE],axis=1)
    
    merged_df.sort_index(ascending=True,inplace=True)
    # merged_df.fillna(method="ffill", inplace = True)
    # merged_df.dropna(inplace=True)
    return merged_df, pairs

def calculate_macd_diff(dataframe:DataFrame,pairs:list[str]):
    df = dataframe.copy()
    
    #Calculating diff
    for pair in pairs:
        for cycle in MARKET_CYCLE_SPEED:
            diff = df[f"{pair}_{cycle}_MACD"] - df[f"BTCUSDT_{cycle}_MACD"]
            diff.name = f"{pair}_DIFF_{cycle}"
            
            diff_bear = df[df["BTCUSDT_SLOW_MACD"]<0][f"{pair}_{cycle}_MACD"] \
                - df[df["BTCUSDT_SLOW_MACD"]<0][f"BTCUSDT_{cycle}_MACD"]
            diff_bear.name = f"{pair}_DIFF_{cycle}_BEAR"
            
            diff_bull = df[df["BTCUSDT_SLOW_MACD"]>0][f"{pair}_{cycle}_MACD"] \
            - df[df["BTCUSDT_SLOW_MACD"]>0][f"BTCUSDT_{cycle}_MACD"]
            diff_bull.name = f"{pair}_DIFF_{cycle}_BULL"
            
            diff_bear_mean = diff_bear.expanding().mean()
            diff_bear_mean.name = f"{pair}_DIFF_{cycle}_BEAR_MEAN"
            
            diff_bull_mean = diff_bull.expanding().mean()
            diff_bull_mean.name = f"{pair}_DIFF_{cycle}_BULL_MEAN"
            
            df = pd.concat([df,diff,diff_bear,diff_bull,diff_bear_mean,diff_bull_mean],axis=1)
            
    return df

def make_market_trend(dataframe:DataFrame,key:str):
    
    #Make market trend data
    shifted = dataframe[dataframe[key] * dataframe[key].shift()<0].copy()
    shifted = pd.concat([shifted,dataframe.head(1)],axis=0)
    shifted.sort_index(inplace=True,ascending=True)
    shifted["start"] = shifted.index
    shifted["end"] = shifted["start"].shift(-1)
    shifted["value"] = shifted[key]
    return shifted[["end","value"]]

def get_aggregate_report(trade_df:DataFrame,
                         holding_period:int,
                         equity:float):
    
    profit = trade_df.groupby("open")["profit"].sum()
    profit.name = "profit"
    aggreegate_df = pd.DataFrame(
        index = [i for i in range(
            int(trade_df["open"].min()),
            int(trade_df["open"].max()),
            TRADE_TIMEFRAME_IN_MS)
         ],
        data = {"profit":profit}
        )
    aggreegate_df["real_profit"] =aggreegate_df["profit"].shift(holding_period)
    aggreegate_df["equity"] = equity
    aggreegate_df.fillna(0,inplace=True)
    aggreegate_df["cumsum_profit"] = aggreegate_df["real_profit"].cumsum()
    aggreegate_df["balance"] = aggreegate_df["equity"] + aggreegate_df["cumsum_profit"]
    aggreegate_df["return"] = aggreegate_df["balance"].pct_change()
    return aggreegate_df

def get_sharpe(aggreegate_df,risk_free_rate:float = 0.05):
    return (aggreegate_df["return"].mean()- risk_free_rate/(365*6))/aggreegate_df["return"].std() * np.sqrt(365*6)

def arbitrate_both_sides(row:pd.Series,
                         market_cycle:str,
                         altcoin_cycle:str,
                         n_pairs:int,
                         pairs:List[str]):
    
    is_bull = False
    long_pairs = short_pairs = []
    
    if row[f"BTCUSDT_{market_cycle}_MACD"] > 0:
        is_bull = True
    
    long_dict = dict()
    short_dict = dict()
    inv_long_dict = dict()
    inv_short_dict = dict()
    
    for pair in pairs:
        if (is_bull):
            diff = float(row[f"{pair}_DIFF_{altcoin_cycle}_BULL"])
            diff_mean = float(row[f"{pair}_DIFF_{altcoin_cycle}_BULL_MEAN"])
        else:
            diff = float(row[f"{pair}_DIFF_{altcoin_cycle}_BEAR"])
            diff_mean = float(row[f"{pair}_DIFF_{altcoin_cycle}_BEAR_MEAN"])
        
        if (diff_mean > 0) & (diff > 0):
            long_dict[pair] = diff_mean
            inv_long_dict[diff_mean] = pair
        elif (diff_mean < 0) & (diff < 0):
            short_dict[pair] = diff_mean
            inv_short_dict[diff_mean] = pair
    
    sorted_inv_long_dict = sorted(inv_long_dict,reverse=True)
    long_pairs = [inv_long_dict[sorted_inv_long_dict[i]] for i in range(min(len(long_dict),n_pairs))]
    
    sorted_inv_short_dict = sorted(inv_short_dict,reverse=False)
    short_pairs = [inv_short_dict[sorted_inv_short_dict[i]] for i in range(min(len(short_dict),n_pairs))]
    
    long_dict[BASE_PAIR] = 0
    short_dict[BASE_PAIR] = 0
    
    if long_pairs and not short_pairs:
        short_pairs += [BASE_PAIR]
    elif short_pairs and not long_pairs:
        long_pairs += [BASE_PAIR]
    
    return long_pairs, short_pairs, long_dict, short_dict

def analyze_trade(trade_df:DataFrame):
    
    n_trade = len(trade_df)
    n_win = len(trade_df[trade_df["profit"]>0])
    n_lose = len(trade_df[trade_df["profit"]<0])
    average_win_size = trade_df[trade_df["profit"]>0]["profit"].mean()
    average_lose_size = trade_df[trade_df["profit"]<0]["profit"].mean()
    
    print(f"Trades: {n_trade}")
    print(f"Win: {n_win/n_trade}, Average: {average_win_size}")
    print(f"Lose: {n_lose/n_trade}, Average: {average_lose_size}")

    trade_type = trade_df["type"].value_counts()
    
    n_long = trade_type["long"] if "long" in trade_type.keys() else 0
    n_short = trade_type["short"] if "short" in trade_type.keys() else 0
    
    print(f"Long: {n_long}({n_long/(n_trade)}), Short: {n_short}({n_short/(n_trade)})")
    
    close_type = trade_df["close_type"].value_counts()
    
    n_tp = close_type[1] if 1 in close_type.keys() else 0
    n_sl = close_type[-1] if -1 in close_type.keys() else 0
    n_exp = close_type[0] if 0 in close_type.keys() else 0
    print(f"TP: {n_tp}({n_tp/(n_trade)}), SL: {n_sl}({n_sl/(n_trade)}), Exp:{n_exp}({n_exp/(n_trade)})")
    print("\nClose type for LONG:")
    print(trade_df[trade_df["type"]=="long"]["close_type"].value_counts())
    print("\nClose type for SHORT:")
    print(trade_df[trade_df["type"]=="short"]["close_type"].value_counts())
    
def check_stoploss(pair:str,
                  enter_time:int,
                  end_time:int,
                  take_profit:float,
                  stop_loss:float,
                  direction:str,
                  trading_df_dict:Dict[str,DataFrame],
                  granular_df_dict:Dict[str,DataFrame]
                  ):

    df = granular_df_dict[pair].loc[enter_time:end_time]
    
    try:
        enter_price = trading_df_dict[pair].loc[enter_time,:]["Close"]
        if (enter_price is NaN):
            assert ValueError("enter price is not a number")
    except Exception as e:
        print(e)
        assert ValueError(f"Error at {pair} {enter_time}")

    out = 0
    
    if direction == "LONG":
        take_profit_price = enter_price * (1+take_profit)           #type: ignore
        stop_loss_price = enter_price * (1-stop_loss)               #type: ignore
    
        first_index_high = df.loc[df["High"]>=take_profit_price].index.min()
        first_index_low = df.loc[df["Low"]<=stop_loss_price].index.min()
        
    elif direction == "SHORT":
        take_profit_price = enter_price * (1-take_profit)           #type: ignore
        stop_loss_price = enter_price * (1+stop_loss)               #type: ignore
        
        first_index_low = df.loc[df["Low"]<=take_profit_price].index.min()
        first_index_high = df.loc[df["High"]>=stop_loss_price].index.min()
     
    else:
        assert ValueError("type must be LONG or Short")
    
    if first_index_high is NaN:                 #type: ignore
        if first_index_low is NaN: # both stop loss and take profit not reached #ignore: type   #type: ignore
            out = 0
        else: #first_index_low is not NaN, mean only take profit reached
            out = 1
    else:
        if first_index_low is NaN: # Only first_index_high is not NaN   #type: ignore
            out = -1
        else: # Both first_index_low and first_index_high is not NaN
            if first_index_low < first_index_high: # take profit reached first  #type: ignore
                out = 1
            elif first_index_low == first_index_high: # both reached in the same timeframe  #type: ignore
                if (randint(0,99) < 50):
                    out = -1
                else:
                    out = 1
            else: # first_index_low > first_index_high, mean stop loss reached first
                out = -1
  
    return out if direction == "SHORT" else -out