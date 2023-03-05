import pandas as pd
import talib

def to_dataframe(results:list,columns:list[str]):
    """ Export analysis results to dataframe, augumented with more analysis
    Args:
        results (list): _description_

    Returns:
        Dataframe: 
    """
    df = pd.DataFrame(
        data=results,
        columns=columns)    
    
    df["n"] = df["tp"]+df["fp"]+df["tn"]+df["fn"]       
    df["acc"] = (df["tp"]+df["tn"])/(df["n"])
    df["precise_p"] = df["tp"]/(df["tp"]+df["fp"])
    df["precise_n"] = df["tn"]/(df["tn"]+df["fn"])
    df["sharpe_p"] = df["mean_p"]/df["std_p"]
    df["sharpe_n"] = df["mean_n"]/df["std_n"]
    
    return df

def calculate_analysis(df:pd.DataFrame,pos_df:pd.DataFrame,neg_df:pd.DataFrame):
    """ Calculate tp, fp, tn, fn, mean and std of positive and negative

    Args:
        d (pd.DataFrame): Input dataframe
        pos_df (pd.DataFrame): dataframe of positive
        neg_df (pd.DataFrame): dataframe of negative

    Returns:
        tp (int): true positive, aka correct buy
        fp (int): false positive, aka incorrect buy
        tn (int): true negative, aka correc sell
        fn (int): false negative, aka incorrect sell
        mean_p (float): mean profit if buy
        std_p (float): standard deviation of profit if buy
        mean_n (float): mean profit if sell
        std_n (float): standard deviation of profit if sell
    """
    tp = len(pos_df[pos_df["Close"]<pos_df["shift"]])
    fp = len(pos_df[pos_df["Close"]>pos_df["shift"]])
    tn = len(neg_df[neg_df["Close"]>neg_df["shift"]])
    fn = len(neg_df[neg_df["Close"]<neg_df["shift"]])
    
    mean_p = pos_df["price_change"].mean()
    std_p = pos_df["price_change"].std()
    mean_s = -neg_df["price_change"].mean()
    std_s =  neg_df["price_change"].std()
    
    number_p = len(df["Close"]<df["shift"])
    number_n = len(df["Close"]>df["shift"])
    recall_p = tp/number_p
    recall_n = tn/number_n

    return tp,fp,tn,fn,mean_p,std_p,mean_s,std_s,recall_p,recall_n

def macd_backtest(data:pd.DataFrame,fast:int, slow:int, n_last:int,close_col:str="Close",return_df:bool=True):
    """ Backtest MACD strategy

    Args:
        data (pd.DataFrame): input data
        fast (int): fast line duration
        slow (int): slow line duration
        n_last (int): number of last days to analyze
        close_col (str, optional): _description_. Defaults to "Close".
        return_df (bool, optional): return dataframe if true, else list
    """
    
    name=f"macd_{fast}_{slow}"
    macd = talib.EMA(data[close_col],fast) - talib.EMA(data[close_col],slow)    #type: ignore
    close = data["Close"].values
    df = pd.DataFrame({"Close":close,name:macd})
    results = []
    print(".",end=" ")
    for i in range(1,fast):
        d = df.copy()
        d["shift"] = d["Close"].shift(-i)
        d["price_change"] = (d["shift"] - d["Close"])/d["Close"]
        d.dropna(inplace=True)
        
        d = d.tail(n_last).copy()
        
        pos_df = d[d[name]>0]
        neg_df = d[d[name]<0]
        
        tp,fp,tn,fn,mean_p,std_p,mean_s,std_s,\
            recall_p,recall_n = calculate_analysis(d,pos_df,neg_df)
        
        results.append((slow,fast,
                        tp,fp,tn,fn,i,
                        mean_p,std_p,mean_s,std_s,
                        recall_p,recall_n))
    
    columns = ["slow","fast","tp","fp","tn","fn","t",
               "mean_p","std_p","mean_n","std_n","recall_p","recall_n"]   
     
    if return_df:
        return to_dataframe(results,columns)
    else:
        return results, columns
    
def backtest_macd_combination(df:pd.DataFrame, n_last:int,close_col:str="Close",fast_min:int=3, fast_max:int=14, 
                  slow_min:int=7, slow_max:int=30):
    """ Calculate accuracy of MACD indicator
        macd = ema(fast) - ema(slow)
        macd > 0, buy
        macd < 0, sell

    Args:
        df (pd.DataFrame): input dataframe with column "Close"
        n_last (int): last duration length to test
        fast_min (int, optional): min period of fast line. Defaults to 3.
        fast_max (int, optional): max period of fast line. Defaults to 14.
        slow_min (int, optional): min period of slow line. Defaults to 7.
        slow_max (int, optional): max period of slow line. Defaults to 30.
        
    Returns:
        'slow' (int, optional): period of slow line. 
        'fast' (int, optional): period of fast line. 
        'tp', true positive, aka correct buy
        'fp', false positive, aka incorrect buy
        'tn', true negative, aka correc sell
        'fn', false negative, aka incorrect sell
        'duration', number of future timeframe to compare price
        'mean_p', mean profit if buy
        'std_p', standard deviation of profit if buy
        'mean_n', mean profit if sell
        'std_n', standard deviation of profit if sell
        'recall_p', ratio of true positive to all positive
        'recall_n', ratio of true negative to all negative
        'n',    sum of tp, tn, fp, fn
        'acc', (tp+tn)/n
        'precise_p', tp/(tp+fp)
        'precise_n', tn/(tn+fn)
        'sharpe_p', mean_p/std_p
        'sharpe_n' mean_n/std_n
    """
    
    data = df.tail(n_last + slow_max + fast_max)
    results = []
    print("Calculating MACD", end="")
    for fast in range(fast_min,fast_max+1):
        for slow in range(max(slow_min,fast+1),slow_max+1):
            print(".",end=" ")
            result, columns = macd_backtest(
                data=data,
                fast=fast,
                slow=slow,
                n_last=n_last,
                close_col=close_col,
                return_df=False
                )
            results += result   #type: ignore
   
    return to_dataframe(results, columns)   #type: ignore

def rsi_backtest(data:pd.DataFrame,n_last:int, period:int,close_col:str="Close",return_df:bool=True):
    """ Backtest RSI

    Args:
        data (pd.DataFrame): data
        n_last (int): number of last days
        period (int): rsi period
        close_col (str, optional): _description_. Defaults to "Close".
        return_df (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: RSI analysis of accuracy, precision, recall, sharpe ...
    """
    name=f"rsi_{period}"
    close = data[close_col].values
    values = talib.RSI(close,period)        #type: ignore
    df = pd.DataFrame({"Close":close,name:values})
    results = []
    for duration in range(1,period):
        print(".", end=" ")
        d = df.copy()
        d["shift"] = d["Close"].shift(-duration)
        d["price_change"] = (d["shift"] - d["Close"])/d["Close"]
        d.dropna(inplace=True)
        d = d.tail(n_last).copy()
        pos_df = d[d[name]<30]
        neg_df = d[d[name]>70]
        
        tp,fp,tn,fn,mean_p,std_p,mean_s,std_s,\
            recall_p,recall_n = calculate_analysis(d,pos_df,neg_df)
        
        results.append((period,tp,fp,tn,fn,duration,mean_p,std_p,mean_s,std_s,recall_p,recall_n))
    
    columns = ["period","tp","fp","tn","fn","t","mean_p","std_p","mean_n","std_n","recall_p","recall_n"]   
    
    if return_df:
        return to_dataframe(results,columns)
    else:
        return results, columns
            
def backtest_rsi_combination(df:pd.DataFrame, n_last:int,close_col:str="Close", period_min:int=3, period_max:int=14):
    """ Calculate accuracy of RSI indicator
        RSI > 70, as overbought, to buy
        RSI < 30, as oversell, to sell

    Args:
        df (pd.DataFrame): input dataframe with column "Close"
        n_last (int): last duration length to test
        period_min (int, optional): min test rsi. Defaults to 3.
        period_max (int, optional): max test rsi. Defaults to 14.

    Returns:
        Dataframe: a dataframe with following columns
            'period', RSI period
            'tp', true positive, aka correct buy
            'fp', false positive, aka incorrect buy
            'tn', true negative, aka correc sell
            'fn', false negative, aka incorrect sell
            'duration', number of future timeframe to compare price
            'mean_p', mean profit if buy
            'std_p', standard deviation of profit if buy
            'mean_n', mean profit if sell
            'std_n', standard deviation of profit if sell
            'recall_p', ratio of true positive to all positive
            'recall_n', ratio of true negative to all negative
            'n',    sum of tp, tn, fp, fn
            'acc', (tp+tn)/n
            'precise_p', tp/(tp+fp)
            'precise_n', tn/(tn+fn)
            'sharpe_p', mean_p/std_p
            'sharpe_n' mean_n/std_n
    """
    data = df.tail(n_last + 2 * period_max)
    results = []
    print("Calculating RSI", end="")
    for period in range(period_min,period_max+1):    
        result, columns = rsi_backtest(
            data = data,
            n_last=n_last,
            period=period,
            close_col=close_col,
            return_df=False)
        results += result   #type: ignore
        
    return to_dataframe(results,columns)    #type: ignore
    