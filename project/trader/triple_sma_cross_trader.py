
# from binance.client import Client
# from binance import ThreadedWebsocketManager
from trader.base_trader import BaseTrader
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import time
from my_logging import logger

class TrippleSMATrader(BaseTrader):  # Triple SMA Crossover
    
    def __init__(self,client, twm, symbol, bar_length, sma_s, sma_m, sma_l, units, position = 0, leverage = 5):
        
        # self.name = symbol+bar_length
        BaseTrader.__init__(self,client,twm,symbol,bar_length,leverage)
        self.units = units
        self.position = position

        #self.trades = 0 
        #self.trade_values = []
        
        #*****************add strategy-specific attributes here******************
        self.SMA_S = sma_s
        self.SMA_M = sma_m
        self.SMA_L = sma_l
        #************************************************************************
    
    def define_strategy(self):
        # logger.debug(self.name + ": define_strategy: IN") 
        data = self.data.copy()
        
        #******************** define your strategy here ************************
        data = data[["Close"]].copy()
        
        data["SMA_S"] = data.Close.rolling(window = self.SMA_S).mean()
        data["SMA_M"] = data.Close.rolling(window = self.SMA_M).mean()
        data["SMA_L"] = data.Close.rolling(window = self.SMA_L).mean()
        
        data.dropna(inplace = True)
                
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1
        #***********************************************************************
    
        self.prepared_data = data.copy()
        # logger.debug(self.name + ": define_strategy: OUT") 
    
    def execute_trades(self): # Adj! 
        logger.debug(self.name + ": execute_trades: IN") 
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = self.client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = self.client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = 2 * self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = self.client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL") 
            elif self.position == -1:
                order = self.client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
        elif self.prepared_data["position"].iloc[-1] == -1: # if position is short -> go/stay short
            if self.position == 0:
                order = self.client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING SHORT") 
            elif self.position == 1:
                order = self.client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = 2 * self.units)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        logger.debug(self.name + ": execute_trades: OUT") 

    def do_when_candle_closed(self):
        self.define_strategy()
        self.execute_trades()
        