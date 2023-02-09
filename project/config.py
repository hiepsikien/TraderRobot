# directory
from __future__ import annotations

TRAINED_MODEL_DIR = "../saved_models/"
TENSORBOARD_LOGDIR = "../logs/tensorboard_log/"
RESULTS_DIR = "../logs/results/"

TRADE_ENV_PARAMETER = {
    "buy_trading_fee" : 0.0002,
    "sell_trading_fee" : 0.0002,
    "take_profit_rate" : 0.02,
    "stop_loss_rate" : -0.01,
    "money_sleep_cost" : 0.0005,
    "reward_scaling": 100
}

CHECKPOINT_CALLBACK = {
    "frequency": 10_000,
    "save_dir": "../saved_models/checkpoint/"
}

CANDLESTICK_INDICATORS = [
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS"
]

BITCOIN_EXTERNAL_INDICATORS = [
    "hashrate",
    "fed_rate",
    "gold",
    "nasdaq",
    "sp500",
    "google_trend",     
]

TRADING_TA_INDICATORS = [ 
    "sma_3_10",
    "sma_7_30",
    "sma_14_50",
    "sma_28_90",         
    # "boll",     
    # "boll7",
    # "boll14",
    # "boll21",
    # "min",      
    # "min7",      
    # "min14",
    # "min21",
    # "max",      
    # "max7",      
    # "max14",
    # "max21",
    # "mom",
    # "mom7",      
    # "mom14",
    # "mom21",
    # "vol",      
    # "vol7",      
    # "vol14",
    # "vol21",
    # "obv",      
    # "mfi7",     
    # "mfi14",
    # "mfi21",
    "rsi7",      
    "rsi14",
    "rsi30",
    # "adx7",      
    # "adx14",
    # "adx21",
    # "roc",      
    # "roc7",      
    # "roc14",
    # "roc21",
    # "atr7",      
    # "atr14",
    # "atr21",
    # "bop",      
    # "ad",       
    # "adosc",     
    # "trange",    
    # "ado",       
    # "willr7",     
    # "willr14",
    # "willr21",
    "dx7",     
    "dx14",
    "dx21",
    # "trix",     # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    # "ultosc",   # Ultimate Oscillator
    # "high",
    # "low",
]

MACRO_TA_INDICATORS = [
    "sma_3_10",
    "sma_7_30",
    "sma_14_50",
    "sma_28_90",           
    # "boll",     
    # "boll7",
    # "boll14",
    # "boll21",
    # "min",      
    # "min7",      
    # "min14",
    # "min21",
    # "max",      
    # "max7",      
    # "max14",
    # "max21",
    # "mom",
    # "mom7",      
    # "mom14",
    # "mom21",
    # "vol",      
    # "vol7",      
    # "vol14",
    # "vol21",
    # "obv",      
    # "mfi7",     
    # "mfi14",
    # "mfi21",
    "rsi7",      
    "rsi14",
    "rsi30",
    # "rsi60",
    # "rsi90",
    # "adx7",      
    # "adx14",
    # "adx21",
    # "roc",      
    # "roc7",      
    # "roc14",
    # "roc21",
    # "atr7",      
    # "atr14",
    # "atr21",
    # "bop",      
    # "ad",       
    # "adosc",     
    # "trange",    
    # "ado",       
    # "willr7",     
    # "willr14",
    # "willr21",
    "dx7",     
    "dx14",
    "dx21",
    # "trix",     # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    # "ultosc",   # Ultimate Oscillator
    # "high",
    # "low",
]

SUPER_TA_INDICATORS = [    
    "sma_3_10",
    "sma_7_30",
    "sma_14_50",
    # "sma_28_90",           
    # "boll",     
    # "boll7",
    # "boll14",
    # "boll21",
    # "min",      
    # "min7",      
    # "min14",
    # "min21",
    # "max",      
    # "max7",      
    # "max14",
    # "max21",
    # "mom",
    # "mom7",      
    # "mom14",
    # "mom21",
    # "vol",      
    # "vol7",      
    # "vol14",
    # "vol21",
    # "obv",      
    # "mfi7",     
    # "mfi14",
    # "mfi21",
    "rsi7",      
    "rsi14",
    "rsi30",
    # "rsi60",
    # "rsi90",
    # "adx7",      
    # "adx14",
    # "adx21",
    # "roc",      
    # "roc7",      
    # "roc14",
    # "roc21",
    # "atr7",      
    # "atr14",
    # "atr21",
    # "bop",      
    # "ad",       
    # "adosc",     
    # "trange",    
    # "ado",       
    # "willr7",     
    # "willr14",
    # "willr21",
    "dx7",     
    "dx14",
    "dx21",
    # "trix",     # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    # "ultosc",   # Ultimate Oscillator
    # "high",
    # "low",
]

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Model Parameters
A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.01, 
    "learning_rate": 0.0007
}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url