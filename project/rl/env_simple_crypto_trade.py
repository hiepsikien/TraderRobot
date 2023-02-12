import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import rl.rewards as rwd


class CryptoTradingEnv(gym.Env):
    ''' 
    A symbol crypto trading envirnment for OpenAI.
    The agent can only trade one crypto, take 3 different 
    actions do nothing, long or short for its whole asset each timeframe.
    The reward is the asset value change percentage 
    
    state[0]: current position, 2 if SHORT, 0 if NEUTRAL, 1 if LONG
    state[1]: latest price
    state[2]: entered price if current position is SHORT or LONG, 0 if NEUTRAL
    state[3:]: indicators
    '''
    def __init__(
        self,
        df: pd.DataFrame,                   # Processed data
        state_space: int,                   # Number of observation state
        reward_scaling: float,              # Initial cash balance
        indicators: list[str],              # indicator to be used in observation
        buy_trading_fee: float = 0.0002,    # buy fee as percentage
        sell_trading_fee: float = 0.0002,   # sell fee as percentage
        take_profit_rate:float = 0.01,      # Take profit
        stop_loss_rate:float = -0.01,       # Stop loss
        day:int=0,                          # timeframe passed from start
        initial:bool = True,                # is that intial state or not

    ):
        self.day = day
        self.df = df
        self.buy_trading_fee = buy_trading_fee
        self.sell_trading_fee = sell_trading_fee
        self.stop_loss_rate = stop_loss_rate
        self.take_profit_rate = take_profit_rate
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
        self.trades = 0
        self.episode = 0
        self.action_memory = []
        self.trade_profit_memory = []
        self.reward_memory = []
        self.cost_memory = []
        self.state_memory = []
        self._seed()

    def get_reward_memory(self):
        return self.reward_memory
    
    def get_trade_profit_memory(self):
        return self.trade_profit_memory

    def get_action_memory(self):
        return self.action_memory
    
    def get_cost_memory(self):
        return self.cost_memory
    
    def get_trade_number(self):
        return self.trades

    def _seed(self, seed=None):
        ''' Set random seed
        '''
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
        #check if it last day
        self.terminal = (self.day >= len(self.df))
        last_day = self.day == (len(self.df)-1)

        if self.terminal:
            return self.state, self.reward, self.terminal, {}

        else:
            self.position = int(self.state[0])
            data = self.df.iloc[self.day,:]
            current_price = data["Close"]
            enter_price = self.state[2]
            last_price = self.state[1]
            
            trading_cost = 0.0
            asset_value_change = 0.0  
            trade_profit_before_cost = 0.0
            self.assumed_trade_profit = 0.0

            #Other update
            match int(self.position):
                case 2: # In short position
                    asset_value_change = -(current_price-last_price)/last_price
                    self.assumed_trade_profit = -(current_price-enter_price)/enter_price
                    match int(action):
                        case 0:
                            # Close short position, increase trade count
                            self.trades+=1
                            trading_cost = -self.buy_trading_fee
                            enter_price = 0
                            trade_profit_before_cost = self.assumed_trade_profit
                        case 1:
                            # Close short and then long
                            self.trades+=2
                            trading_cost = 2 * -self.buy_trading_fee
                            enter_price = current_price
                            trade_profit_before_cost = self.assumed_trade_profit
                        case 2:
                            # Stay short
                            pass
                        case other:
                            raise(ValueError("Action must be 0,1,2. We got {}".format(int(action))))
            
                case 0: # In neural position
                    match int(action):
                        case 0:
                            # Stay netral
                            pass
                        case 1:
                            # From netral to long
                            trading_cost = -self.buy_trading_fee
                            enter_price = current_price
                            self.trades +=1
                        case 2:
                            # From neutral to short
                            trading_cost = -self.sell_trading_fee
                            enter_price = current_price
                            self.trades +=1
                        case other:
                            raise(ValueError("Action must be 0,1,2. We got {}".format(int(action))))
                    
                case 1: # In long position
                    asset_value_change = (current_price-last_price)/last_price
                    self.assumed_trade_profit = (current_price-enter_price)/enter_price
                    match int(action):
                        case 0:
                            # Close long position
                            trading_cost = -self.sell_trading_fee
                            enter_price = 0
                            self.trades+=1
                            trade_profit_before_cost = self.assumed_trade_profit
                        case 1:
                            # Stay long
                            pass
                        case 2:
                            # Close long position and then short
                            self.trades+=2
                            trading_cost =  -2*self.sell_trading_fee
                            enter_price = current_price
                            trade_profit_before_cost = self.assumed_trade_profit
                        case other:
                            raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
                case other:
                    raise(ValueError("Position must be 0,1,2. We got {}".format(int(self.position))))

            if (last_day):
                # neglect minor different in possible incured trading cost
                self.trade_profit = self.assumed_trade_profit
                
            else:
                self.trade_profit = trade_profit_before_cost + trading_cost

            
            #Get reward
            self.reward, self.trade_reward, self.trade_penalty= rwd.get_reward_beta(
                state = self.state,
                action=action,
                new_price=current_price,
                env=self
            )    

            # Update state    
            self.previous_position = self.position
            self.position = int(action)
            self.reward = self.reward * self.reward_scaling
            self.state = [self.position, current_price, enter_price] + data[self.indicators].tolist()

            # Update memory
            self.action_memory.append(action)
            self.reward_memory.append(self.reward)
            self.trade_profit_memory.append(self.trade_profit)
            self.asset_value_change_memory.append(asset_value_change)
            self.cost_memory.append(trading_cost)
            self.state_memory.append(self.state)
            
            self.day +=1
            return self.state, self.reward, self.terminal, {}

    def get_accumulated_profit(self):
        return np.exp(np.log(np.array(self.trade_profit_memory)+1).sum()) - 1

    def reset(self):
        ''' Reset the envirnoment'''
        self.day = 0
        self.state = self._initiate_state()
        self.trades = 0
        self.terminal = False
        self.reward_memory = []
        self.action_memory = []
        self.trade_profit_memory = []
        self.asset_value_change_memory = []
        self.cost_memory = []
        self.episode += 1

        return self.state

    def _initiate_state(self):
        ''' Initialize the state '''
        data = self.df.iloc[self.day,:]
        return [0,data["Close"],0] + data[self.indicators].tolist()
        
    def render(self, mode="human", close=False):
        position_state = {0:"NEUTRAL",1:"LONG",2:"SHORT"}
        print("{}: Previous:{} | Action:{} | Reward:{} | TradeRwd:{} | TradePen:{}| Profit:{} | Assumed Profit:{} |{}".
            format(
                self.day, 
                position_state[self.previous_position], 
                position_state[self.position], 
                round(self.reward,5), 
                round(self.trade_reward,5), 
                round(self.trade_penalty,5), 
                round(self.trade_profit,5), 
                round(self.assumed_trade_profit,5), 
                self.terminal))
        
        return self.state