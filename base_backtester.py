## Back test with leverage for trader
## also include optimizer function
## for Tripple SMA algorithm

from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from my_logging import logger

class BaseBacktester():
    
    #Initialize the instance
    def __init__(self,filepath,symbol, start, end, tc):

        logger.debug("IN")
        
        print("Initialized future back tester")

        self.filepath = filepath # File store historical data used for testing
        self.symbol = symbol # Target symbol to be tested
        self.start = start # start date for data import
        self.end = end # end date for data import
        self.tc = tc # trading cost
        self.results = None #variable to store the results
        self.get_data() # Import the data right after initialize the object
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        logger.debug("OUT")

    #Give the class instance a name
    def __repr__(self) -> str:
        return "BaseBacktester(symbol={}, start = {}, end = {})".format(self.symbol, self.start,self.end)


    #Import the data
    def get_data(self):
        logger.debug("IN")
        # Create a dataframe from CSV with date as index column
        raw = pd.read_csv(self.filepath, parse_dates=["Date"],index_col="Date") 

        # Trimming to data of start and end region
        raw = raw.loc[self.start:self.end].copy()

        #Create a returns column as log of return after each interval
        raw["returns"] = np.log(raw.Close/raw.Close.shift(1))
        
        #Provide that data to the back tester
        self.data = raw
        logger.debug("OUT")

    #Prepare data, do the back test and report performance
    def test_strategy(self,smas):
        pass
    
        ########################## Strategy-Specific #############################
        
        ##########################################################################

    
    # Prepare the data for back test
    def prepare_data(self, smas):
        pass
    
        ########################## Strategy-Specific #############################
        
        ##########################################################################


    # Run back test
    def run_backtest(self):
        
        logger.debug("IN")
        #Make a copy of the prepared data
        data = self.results.copy()

        #Calculate the return each time we trade based on the position change
        data["strategy"] = data["position"].shift(1) * data["returns"]
        
        #Calculate the number of trades
        data["trades"] = data.position.diff().fillna(0).abs()
        
        #Subtract the trading fee from return
        data.strategy = data.strategy + data.trades  * self.tc

        #Update the results
        self.results = data

        logger.debug("OUT")


    # Calculate and print various performance mertrics
    def print_performance(self):
        logger.debug("IN")
        data = self.results.copy()
        strategy_multiple = round(self.calculate_multiple(data.strategy),6)
        bh_multiple = round(self.calculate_multiple(data.returns),6)
        out_perform = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(data.strategy),6)
        ann_mean = round(self.calculate_annualized_mean(data.strategy),6)
        ann_std = round(self.calculate_annualized_std(data.strategy),6)
        sharpe = round(self.calculate_sharpe(data.strategy),6)

        print(100 * "=")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(out_perform))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))
        
        print(100 * "=")

        logger.debug("OUT")

    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1

    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)
    
    #Plot the cumulative performance of the trading strategy compared to buy and hold
    def plot_results(self):

        if self.results is None:
            print("Run test_strategy() first")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns","cstrategy"]].plot(title=title,figsize=(12,8))
    