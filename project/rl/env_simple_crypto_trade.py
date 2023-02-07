import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

class CryptoTradingEnv(gym.Env):
    ''' 
    A symbol crypto trading envirnment for OpenAI.
    The agent can only trade one crypto, take 3 different 
    actions do nothing, long or short for its whole asset each timeframe.
    The reward is the asset value change percentage 
    
    state[0]: current position, 2 if SHORT, 0 if NEUTRAL, 1 if LONG
    state[1]: current price
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
        money_sleep_cost:float = 0.00002,   # penalty of keep money dont trade
        take_profit_rate:float = 0.01,      # Take profit
        stop_loss_rate:float = -0.01,       # Stop loss
        day:int=0,                          # timeframe passed from start
        initial:bool = True,                # is that intial state or not

    ):
        self.day = day
        self.df = df
        self.buy_trading_fee = buy_trading_fee
        self.sell_trading_fee = sell_trading_fee
        self.money_sleep_cost = money_sleep_cost
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
        self.rewards_memory = []
        self.profit_memory = []
        self.state_memory = []
        self.cost_memory = []
        self._seed()

    def _seed(self, seed=None):
        ''' Set random seed
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _cal_reward_with_sltp(self,return_rate:float):
        ''' 
        Calculate reward with regard to stop loss and take profit
        If it is over take profit and stop loss level, return full, 
        otherwise, return half of that
        '''
        if (return_rate > self.take_profit_rate) or (return_rate < self.stop_loss_rate):
            return return_rate
        else:
            return 0.0

    def _get_reward(self,position,action,enter_price,current_price):
        '''
        Reward function for ML model.

        Params:
        - position: current position 0:Neutral, 1:Long, 2: Short
        - action: predicted action 0:Neutral, 1: Long, 2: Short
        - enter_price: the entered price if it is in long or short, otherwise 0
        - current_price: the current price

        Return: reward 
        '''

        reward = 0.0
        return_rate = (enter_price-current_price)/enter_price if enter_price>0 else 0
        match int(position):
            case 2: # In short position
                match action:
                    case 0:
                        # Close short position 
                        reward = self._cal_reward_with_sltp(return_rate) - self.buy_trading_fee
                    case 1:
                        # Close short and then long     
                        reward = self._cal_reward_with_sltp(return_rate) - 2 * self.buy_trading_fee
                    case 2:
                        # Stay short
                        reward = self._cal_reward_with_sltp(return_rate)
                    case other:
                        raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
        
            case 0: # In neural position
                match action:
                    case 0:
                        # Stay netral
                        reward = -self.money_sleep_cost
                    case 1:
                        # From netral to long
                        reward = -self.buy_trading_fee
                    case 2:
                        # From neutral to short
                        reward = -self.sell_trading_fee
            
                    case other:
                        raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
                
            case 1: # In long position
                match action:
                    case 0:
                        # Close long position
                        reward = self._cal_reward_with_sltp(return_rate) - self.sell_trading_fee
                    case 1:
                        # Stay long
                        reward = self._cal_reward_with_sltp(return_rate)
                    case 2:
                        # Close long position and then short
                        reward = self._cal_reward_with_sltp(return_rate) - 2 *self.sell_trading_fee
                    case other:
                        raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
            case other:
                raise(ValueError("Position must be 0,1,2. We got {}".format(self.position)))
        return reward
    
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

        if self.terminal:
            return self.state, self.reward, self.terminal, {}

        else:
            self.position = int(self.state[0])
            data = self.df.iloc[self.day,:]
            current_price = data["Close"]
            enter_price = self.state[2]
            
            #Get reward
            self.reward= self._get_reward(
                position = self.position,
                action = int(action),
                enter_price = self.state[2],
                current_price = data["Close"]
            )
            
            trading_cost = 0.0
            trade_profit = (enter_price-current_price)/enter_price if enter_price>0 else 0  

            #Other update
            match int(self.position):
                case 2: # In short position
                    match int(action):
                        case 0:
                            # Close short position, increase trade count
                            self.trades+=1
                            trading_cost = -self.buy_trading_fee
                            enter_price = 0
                        case 1:
                            # Close short and then long
                            self.trades+=2
                            trading_cost = 2 * -self.buy_trading_fee
                            enter_price = current_price
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
                    match int(action):
                        case 0:
                            # Close long position
                            trading_cost = -self.sell_trading_fee
                            enter_price = 0
                            self.trades+=1
                        case 1:
                            # Stay short
                            pass
                        case 2:
                            # Close long position and then short
                            self.trades+=2
                            trading_cost =  -2*self.sell_trading_fee
                            enter_price = current_price
                        case other:
                            raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
                case other:
                    raise(ValueError("Position must be 0,1,2. We got {}".format(int(self.position))))

            self.profit = trade_profit + trading_cost
            self.previous_position = self.position
            self.position = int(action)
            self.state = [self.position, current_price, enter_price] + data[self.indicators].tolist()
            self.reward = self.reward * self.reward_scaling
            self.action_memory.append(action)
            self.rewards_memory.append(self.reward)
            self.profit_memory.append(self.profit) 
            self.cost_memory.append(trading_cost)
            self.state_memory.append(self.state)
            self.day +=1
            return self.state, self.reward, self.terminal, {}

    def reset(self):
        ''' Reset the envirnoment'''
        self.day = 0
        self.state = self._initiate_state()
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.action_memory = []
        self.profit_memory = []
        self.cost_memory = []
        self.episode += 1

        return self.state

    def _initiate_state(self):
        ''' Initialize the state '''
        data = self.df.iloc[self.day,:]
        return [0,data["Close"],0] + data[self.indicators].tolist()
        
    def render(self, mode="human", close=False):
        position_state = {0:"NEUTRAL",1:"LONG",2:"SHORT"}
        print("{}: Previous: {} | Action: {} | Reward: {} | Profit : {} {}".
            format(
                self.day, 
                position_state[self.previous_position], 
                position_state[self.position], 
                round(self.reward,5), 
                round(self.profit,5), 
                self.terminal))
        
        return self.state
    
    def plot_multiple(self,data:pd.DataFrame):
        only_first_change = data.loc[data["action"] != data["action"].shift()]
        long = only_first_change.loc[data["action"]==1]
        short = only_first_change.loc[data["action"]==2]
        neutral = only_first_change.loc[data["action"]==0]
        plt.figure(figsize=(12,6),dpi=720)
        plt.title("RobotTrader Performance")
        plt.plot(data.index,data["cumsum_profit"],color="tab:blue",label="profit_after_cost")
        plt.plot(data.index,data["cumsum_cost"],color="tab:red",label="trading_cost")
        plt.plot(data.index,data["relative_price"],color="tab:cyan",label="relative_price")
        plt.scatter(long.index,long["relative_price"],color="tab:green",marker="o")             #type: ignore
        plt.scatter(neutral.index,neutral["relative_price"],color="tab:orange",marker="o")       #type: ignore
        plt.scatter(short.index,short["relative_price"],color="tab:red",marker="o")           #type: ignore
        plt.ylabel("Multiples")
        plt.xlabel("Timeframe")
        plt.legend()
        plt.show()

    def make_result_data(self):
        result = pd.DataFrame({
            "reward":self.rewards_memory,
            "profit":self.profit_memory,
            "action":self.action_memory,
            "cost": self.cost_memory
            })
        result["log_profit"] = np.log(result["profit"]+1)
        result["cumsum_profit"] = np.exp(result["log_profit"].cumsum(axis=0))
        result["log_cost"] = np.log(1-result["cost"])
        result["cumsum_cost"] = np.exp(result["log_cost"].cumsum(axis=0)) -1
        result["cumsum_profit_before_cost"] = result["cumsum_profit"] + result["cumsum_cost"]
        result["price"] = self.df["Close"].values
        result["relative_price"] = result["price"]/result.iloc[0,:]["price"]
        print(result["action"].value_counts())
        return result.copy()