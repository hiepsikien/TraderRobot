## Back test with leverage for trader
## also include optimizer function
## for Tripple SMA algorithm

from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from my_logging import logger
from base_backtester import BaseBacktester

class TrippleSMABacktester(BaseBacktester):
    
    #Initialize the instance
    def __init__(self,filepath,symbol, start, end, tc):

        logger.debug("IN")
        BaseBacktester.__init__(self=self,filepath=filepath,symbol=symbol,start=start,end=end,tc=tc)
        logger.debug("OUT")

    #Give the class instance a name
    def __repr__(self) -> str:
        return "TrippleSMABacktester(symbol={}, start = {}, end = {})".format(self.symbol, self.start,self.end)


    #Prepare data, do the back test and report performance
    def test_strategy(self,smas):
        logger.debug("IN")
        self.SMA_S = smas[0]
        self.SMA_M = smas[1]
        self.SMA_L = smas[2]

        #Call the prepare data
        self.prepare_data(smas)

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
    def prepare_data(self, smas):
    
        ########################## Strategy-Specific #############################
        
        logger.debug("IN")
        data = self.data[["Close", "returns"]].copy()
        data["SMA_S"] = data.Close.rolling(window = smas[0]).mean()
        data["SMA_M"] = data.Close.rolling(window = smas[1]).mean()
        data["SMA_L"] = data.Close.rolling(window = smas[2]).mean()
        
        data.dropna(inplace = True)
                
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        self.results = data

        logger.debug("OUT")

        ##########################################################################


    # Try all parameter values and find the best.
    def optimize_strategy(self, SMA_S_range, SMA_M_range, SMA_L_range, metric):
        logger.debug("IN")
        self.metric = metric

        SMA_S_range = range(1,SMA_S_range)
        SMA_M_range = range(1,SMA_M_range)
        SMA_L_range = range(1,SMA_L_range)

        combinations = list(product(SMA_S_range,SMA_M_range,SMA_L_range))

        multiples = []
        sharpes = []

        for comb in combinations:
            (sma_s,sma_m,sma_l) = comb
            con1 = sma_s < sma_m
            con2 = sma_m < sma_l
            multiple_ratio = -1
            sharpe_ratio = 9999
            if (con1 & con2):
                self.prepare_data(smas = comb)
                self.run_backtest()
                multiple_ratio = self.calculate_multiple(self.results.strategy)
                sharpe_ratio = self.calculate_sharpe(self.results.strategy)
            
            multiples.append(multiple_ratio)
            sharpes.append(sharpe_ratio)

            logger.debug("Combination ({},{},{}) | Multiple =  {} | Sharpe = {}".format(sma_s,sma_m,sma_l,round(multiple_ratio,5), round(sharpe_ratio,5)))

        self.results_overview = pd.DataFrame(data = np.array(combinations), columns = ["SMA_S","SMA_M","SMA_L"])
        self.results_overview["Multiple"] = multiples
        self.results_overview["Sharpe"] = sharpes
        self.find_best_strategy(metric)
        logger.debug("OUT")

    #Find the optimal strategy
    def find_best_strategy(self,metric):

        logger.debug("IN")

        best = self.results_overview.nlargest(1,metric)
        
        SMA_S = best.SMA_S.iloc[0]
        SMA_M = best.SMA_M.iloc[0]
        SMA_L = best.SMA_L.iloc[0]
        multiple = best.Multiple.iloc[0]
        sharpe = best.Sharpe.iloc[0]

        print("SMA_S: {} | SMA_M: {} | SMA_L: {} | Multiple: {} | Sharpe: {}".format(SMA_S, 
        SMA_M, SMA_L, round(multiple,5), round(sharpe, 5))) 
        
        self.test_strategy(smas = (SMA_S,SMA_M, SMA_L))

        logger.debug("OUT")

    # Calculate and print various performance mertrics
    def print_performance(self):
        logger.debug("IN")
        print("TRIPLE SMA STRATEGY | INSTRUMENT = {} | SMAs = {}".format(self.symbol, [self.SMA_S, self.SMA_M, self.SMA_L]))
        BaseBacktester.print_performance(self)
        logger.debug("OUT")

    