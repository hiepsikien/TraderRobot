import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

class CryptoTradingEnv(gym.Env):
    ''' 
    A symbol crypto trading envirnment for OpenAI.
    The agent can only trade one crypto, take 3 different 
    actions do nothing, long or short for its whole asset each timeframe.
    The reward is the asset value change percentage 
    
    state[0]: current position, -1 if shorting, 0 if netural, 1 if long
    state[1]: current price
    state[2]: last price
    state[3:]: indicators

    '''

    def __init__(
        self,
        df: pd.DataFrame,        # Processed data
        state_space: int,      # Number of state
        reward_scaling: float,   # Initial cash balance
        indicators: list[str],   # indicator to be used in observation
        buy_trading_fee: float = 0.0002,     #buy cost
        sell_trading_fee: float = 0.0002,    #sell cost
        money_sleep_cost = 0.00002,
        day:int=0,                          #timeframe passed from start
        initial:bool = True,                # is that intial state or not

    ):
        self.day = day
        self.df = df
        self.buy_trading_fee = buy_trading_fee
        self.sell_trading_fee = sell_trading_fee
        self.money_sleep_cost = money_sleep_cost
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.indicators = indicators
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(self.state_space,))
        self.terminal = False
        self.initial = initial
        
        #initialize state
        self.state = self._initiate_state()

        #initialize reward
        self.position = 0
        self.reward = 0
        self.last_price = 0.0
        self.trades = 0
        self.cost = 0
        self.episode = 0
        self.action_memory = []
        self.rewards_memory = []
        self.state_memory = []
        self._seed()
        self.count = 0
    
    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        The main logic for each time step.
        If the action is -1, short all portfolio. 
        If the action is 0, do nothing
        If the action is +1 long all portfolio. 
        The agent will do intermediate go netral first 
        if from long to short and vice-verse.
        '''
        self.reward = 0.0
        #check if it last day
        
        self.terminal = (self.day >= len(self.df)-1)

        if self.terminal:
            print("It is done")
            return self.state, self.reward, self.terminal, {}

        else:
            data = self.df.iloc[self.day,:]
            current_price = data["Close"]
            enter_price = self.state[2]
            self.reward = 0
            match int(self.state[0]):
                case 2: # In short position
                    match int(action):
                        case 0:
                            # Close short position, increase trade count
                            self.reward = (enter_price-current_price)/enter_price - self.buy_trading_fee
                            self.trades+=1
                            enter_price = 0
                        case 1:
                            # Close short and then long
                            self.reward = (enter_price-current_price)/enter_price - self.buy_trading_fee
                            self.reward -= self.buy_trading_fee
                            self.trades+=2
                            enter_price = current_price
                        case 2:
                            # Stay short
                            self.reward = (enter_price-current_price)/enter_price
                            pass
                        case other:
                            raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
            
                case 0: # In neural position
                    match int(action):
                        case 0:
                            # Stay netral
                            self.reward -= self.money_sleep_cost
                        case 1:
                            # From netral to long
                            self.reward = -self.buy_trading_fee
                            enter_price = current_price
                            self.trades +=1
                        case 2:
                            # From neutral to short
                            self.reward = - self.sell_trading_fee
                            enter_price = current_price
                            self.trades +=1
                        case other:
                            raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
                    
                case 1: # In long position
                    match int(action):
                        case 0:
                            # Close long position
                            self.reward = (current_price-enter_price)/enter_price - self.sell_trading_fee
                            enter_price = 0
                            self.trades+=1
                        case 1:
                            # Stay short
                            self.reward = (current_price-enter_price)/enter_price
                            pass
                        case 2:
                            # Close long position and then short
                            self.reward = (current_price-enter_price)/enter_price - self.sell_trading_fee
                            self.reward -= self.sell_trading_fee
                            self.trades+=2
                            enter_price = current_price
                        case other:
                            raise(ValueError("Action must be 0,1,2. We got {}".format(action)))

                case other:
                    raise(ValueError("Position must be 0,1,2. We got {}".format(int(self.state[0]))))


            self.state = [action, current_price, enter_price] + data[self.indicators].tolist()
            self.reward = self.reward * self.reward_scaling
            
            
            self.action_memory.append(action)
            self.rewards_memory.append(self.reward)
            self.state_memory.append(self.state)
            
            self.day +=1

            return self.state, self.reward, self.terminal, {}
    
    def reset(self):
        ''' Reset the envirnoment'''
        self.state = self._initiate_state()

        self.day = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.action_memory = []
        self.episode += 1
        self.count = 0

        return self.state

    def _initiate_state(self):
        ''' Initialize the state '''

        data = self.df.iloc[self.day,:]
        return [0,data["Close"],0] + data[self.indicators].tolist()
        
    def render(self, mode="human", close=False):
        pos = {0:"NETRAUL",1:"LONG",2:"SHORT"}
        print("{}: {} | Reward : {}  {}".format(self.day,pos[self.position], self.reward,self.terminal))
        return self.state
        