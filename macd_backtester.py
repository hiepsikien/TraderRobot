## Back test with leverage for trader
## also include optimizer function
## for Tripple SMA algorithm

from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
plt.style.use("seaborn")
from my_logging import logger
from base_backtester import BaseBacktester

class MACDBacktester(BaseBacktester):
    
    #Initialize the instance
    def __init__(self,filepath,symbol, start, end, tc):

        logger.debug("IN")
        BaseBacktester.__init__(self=self,filepath=filepath,symbol=symbol,start=start,end=end,tc=tc)
        logger.debug("OUT")

    #Give the class instance a name
    def __repr__(self) -> str:
        return "MACDBacktester(symbol={}, start = {}, end = {})".format(self.symbol, self.start,self.end)


    #Prepare data, do the back test and report performance
    def test_strategy(self,macd):
        logger.debug("IN")
        self.MA_SLOW = macd[0]
        self.MA_FAST = macd[1]
        self.MA_SIGNAL = macd[2]

        #Call the prepare data
        self.prepare_data(macd)

        #Run back test
        self.run_backtest()

        # Calculate the compound results
        data = self.results.copy(deep=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        #Call the showing of performance
        self.print_performance()
        logger.debug("OUT")
    
    # Prepare the data for back test
    def prepare_data(self, macd):
    
        ########################## Strategy-Specific #############################
        
        # logger.debug("IN")
        data = self.data.copy()
        close = data[["Close"]].copy().to_numpy().flatten()

        (ma_slow,ma_fast,ma_signal) = macd

        macd, macd_signal, macd_hist = ta.MACD(close,fastperiod=ma_fast,slowperiod=ma_slow,signalperiod=ma_signal)
    

        data["MACD"] = macd
        data["MACD_SIGNAL"] = macd_signal
        data["MACD_HIST"] = macd_hist
        
        data.dropna(inplace = True)
                
        cond1 = data.MACD_HIST > 0
        cond2 = data.MACD_HIST < 0
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        self.results = data

        # logger.debug("OUT")

        ##########################################################################


    # Try all parameter values and find the best.
    def optimize_strategy(self, MA_SLOW, MA_FAST, MA_SIGNAL, metric):
        # logger.debug("IN")

        self.metric = metric
        MA_SLOW = range(1,MA_SLOW)
        MA_FAST = range(1,MA_FAST)
        MA_SIGNAL = range(1,MA_SIGNAL)

        combinations = list(product(MA_SLOW,MA_FAST,MA_SIGNAL))

        multiples = []
        sharpes = []

        for comb in combinations:
            (ma_slow,ma_fast,ma_signal) = comb
            multiple_ratio = -1
            sharpe_ratio = -1
            if ((ma_slow <ma_fast) & (ma_signal < ma_slow)):
                self.prepare_data(macd = comb)
                self.run_backtest()
                multiple_ratio = self.calculate_multiple(self.results.strategy)
                sharpe_ratio = self.calculate_sharpe(self.results.strategy)
            
            multiples.append(multiple_ratio)
            sharpes.append(sharpe_ratio)

            logger.debug("Combination ({},{},{}) | Multiple =  {} | Sharpe = {}".format(ma_slow,ma_fast,ma_signal,round(multiple_ratio,5), round(sharpe_ratio,5)))

        self.results_overview = pd.DataFrame(data = np.array(combinations), columns = ["MA_SLOW","MA_FAST","MA_SIGNAL"])
        self.results_overview["Multiple"] = multiples
        self.results_overview["Sharpe"] = sharpes
        self.find_best_strategy(metric)
        # logger.debug("OUT")

    #Find the optimal strategy
    def find_best_strategy(self,metric):

        # logger.debug("IN")

        best = self.results_overview.nlargest(1,metric)
        
        MA_SLOW = best.MA_SLOW.iloc[0]
        MA_FAST = best.MA_FAST.iloc[0]
        MA_SIGNAL = best.MA_SIGNAL.iloc[0]
        multiple = best.Multiple.iloc[0]
        sharpe = best.Sharpe.iloc[0]

        print("MA_SLOW: {} | SMA_FAST: {} | SMA_SIGNAL: {} | Multiple: {} | Sharpe: {}".format(MA_SLOW, MA_FAST, MA_SIGNAL, round(multiple,5), round(sharpe, 5))) 
        
        self.test_strategy(macd = (MA_SLOW, MA_FAST, MA_SIGNAL))

        # logger.debug("OUT")

    # Calculate and print various performance mertrics
    def print_performance(self):
        # logger.debug("IN")
        print("MACD STRATEGY | INSTRUMENT = {} | MAs = {}".format(self.symbol,[self.MA_SLOW, self.MA_FAST, self.MA_SIGNAL]))
        BaseBacktester.print_performance(self)
        # logger.debug("OUT")

    