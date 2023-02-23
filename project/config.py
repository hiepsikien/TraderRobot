# directory
from __future__ import annotations

TRAINED_MODEL_DIR = "../saved_models/"
TENSORBOARD_LOGDIR = "../logs/tensorboard_log/"
RESULTS_DIR = "../logs/results/"

TRADE_ENV_PARAMETER = {
    "buy_trading_fee" : 0.0002,
    "sell_trading_fee" : 0.0002,
    "reward_scaling": 1,
}
REWARD = {
    "max_duration": 10,
    "take_profit_rate" : 0.2,
    "stop_loss_rate" : -0.2,
    "risk_free_annual_return": 0.0,
    "no_trade_penalty" : 5e-4,
    "amplified_rate": 1.1,
    "discount_rate": 0.9,
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

FOR_4H_INDICATORS = [
    ["sma_3_10","sma_7_30","rsi7","rsi14","rsi30","cci7","cci14","cci30","dx7","dx14","dx30"] + BITCOIN_EXTERNAL_INDICATORS,
    ["sma_3_10","sma_7_30","rsi7","rsi14","rsi30","cci7","cci14","cci30","dx7","dx14","dx30"],
    ["sma_3_10","sma_7_30","rsi7","rsi14","rsi30","cci7","cci14","cci30","dx7","dx14","dx30"],
    ["sma_3_10","rsi7","rsi14","cci7","cci14","dx7","dx14"]
]

FOR_1D_INDICATORS = [
    ["sma_3_10","sma_7_30","rsi7","rsi14","rsi30","cci7","cci14","cci30","dx7","dx14","dx30"] + BITCOIN_EXTERNAL_INDICATORS,
    ["sma_3_10","sma_7_30","rsi7","rsi14","rsi30","cci7","cci14","cci30","dx7","dx14","dx30"],
    ["sma_3_10","rsi7","rsi14","cci7","cci14","dx7","dx14"]
]

TA_INDICATORS_FULL = {
    "sma_3_10",
    "sma_7_30",
    "sma_14_50",
    "sma_28_90",    
    "cci7",
    "cci14",
    "cci30",     
    "up_bb",
    "low_bb"
    "min",      
    "min7",      
    "min14",
    "min30",
    "max",      
    "max7",      
    "max14",
    "max30",
    "mom",
    "mom7",      
    "mom14",
    "mom30",
    "vol",      
    "vol7",      
    "vol14",
    "vol30",
    "obv",      
    "mfi7",     
    "mfi14",
    "mfi30",
    "rsi7",      
    "rsi14",
    "rsi30",
    "adx7",      
    "adx14",
    "adx30",
    "roc",      
    "roc7",      
    "roc14",
    "roc30",
    "atr7",      
    "atr14",
    "atr30",
    "bop",      
    "ad",       
    "adosc",     
    "trange",    
    "ado",       
    "willr7",     
    "willr14",
    "willr30",
    "dx7",     
    "dx14",
    "dx30",
    "trix",     # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    "ultosc",   # Ultimate Oscillator
    "high",
    "low",
}

TIMEFRAMES_IN_MS = {
    "1m":60*1000,
    "5m":5*60*1000,
    "15m":15*60*1000,
    "1h":60*60*1000,
    "4h":4*60*60*1000,
    "1d":24*60*60*1000,
    "1w": 7*24*60*60*1000,
}



# Model Parameters
A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.01, 
    "learning_rate": 0.0007
}

DQN_PARAMS = {
    "learning_rate": 1e-4,
    "buffer_size": 1_000_000,  
    "learning_starts": 10_000,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 10000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}

C_PPO_PARAMS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_steps": 1024,
    "ent_coef": 0.03,
    "learning_rate": 0.0002,
    "batch_size": 64
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