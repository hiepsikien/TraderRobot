from datetime import datetime, timedelta
from binance import ThreadedWebsocketManager
import pandas as pd
import time
from my_logging import logger

class BaseTrader():

    available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

    def __init__(self,client,twm,symbol,bar_length,leverage) -> None:
        self.client = client
        self.twm = twm
        self.symbol = symbol
        self.bar_length = bar_length
        self.leverage = leverage
        self.cum_profits = 0


    def start_trading(self, historical_days):
        logger.debug(self.name + ": start_trading: IN")        
        self.client.futures_change_leverage(symbol = self.symbol, leverage = self.leverage) # NEW
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days = historical_days)
            self.twm.start_kline_futures_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length) # Adj: start_kline_futures_socket
        # "else" to be added later in the course 
        logger.debug(self.name + ": start_trading: OUT") 

    def set_name(self,name):
        self.name = name

    def stop_trading(self):
        logger.debug(self.name + ": stop_trading: IN")                
        self.twm.stop()
        logger.debug(self.name + ": stop_trading: OUT") 
    

    def get_most_recent(self, symbol, interval, days):
        logger.debug(self.name + ": get_most_recent: IN") 
        now = datetime.utcnow()
        past = str(now - timedelta(days = days))
    
        bars = self.client.futures_historical_klines(symbol = symbol, interval = interval,
                                            start_str = past, end_str = None, limit = 1000) # Adj: futures_historical_klines
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]
        
        self.data = df
        logger.debug(self.name + ": get_most_recent: OUT")  
    
    def stream_candles(self, msg):
        # logger.debug("stream_candles: IN") 
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        
        # print out
        print(".", end = "", flush = True) 
    
        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]
        
        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.do_when_candle_closed()

        # logger.debug("stream_candles: OUT") 
        

    def report_trade(self, order, going): # Adj!
        # logger.debug(self.name + ": "+  going) 
        time.sleep(0.1)
        order_time = order["updateTime"]
        trades = self.client.futures_account_trades(symbol = self.symbol, startTime = order_time)
        order_time = pd.to_datetime(order_time, unit = "ms")
        
        # extract data from trades object
        df = pd.DataFrame(trades)
        columns = ["qty", "quoteQty", "commission","realizedPnl"]
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        base_units = round(df.qty.sum(), 5)
        quote_units = round(df.quoteQty.sum(), 5)
        commission = -round(df.commission.sum(), 5)
        real_profit = round(df.realizedPnl.sum(), 5)
        price = round(quote_units / base_units, 5)
        
        # calculate cumulative trading profits
        self.cum_profits += round((commission + real_profit), 5)
        
        # print trade report
        # logger.info("\n")
        logger.info(2 * "\n" + 100* "-")
        logger.info("{} | {}".format(order_time, going)) 
        logger.info("Trader = {} | Symbol = {} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(self.name,self.symbol, order_time, base_units, quote_units, price))
        logger.info("{} | Profit = {} | CumProfits = {} ".format(order_time, real_profit, self.cum_profits))
        logger.info(100 * "-" + "\n")
        # logger.debug("report_trade: OUT") 

    def do_when_candle_closed(self):
        logger.critical(self.name + ": Action when candle completed: UNIMPLEMENTED")