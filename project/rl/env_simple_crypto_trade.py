from math import gamma
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import rl.rewards as rwd
import wandb


class CryptoTradingEnv(gym.Env):
    """ 
    A symbol crypto trading envirnment for OpenAI.
    The agent can only trade one crypto, take 3 different 
    actions do nothing, long or short for its whole asset each timeframe.
    The reward is the asset value change percentage 
    
    state[0]: current position, 2 if SHORT, 0 if NEUTRAL, 1 if LONG
    state[1]: price change
    state[2]: assumed position profit
    state[3]: unallocated reward
    state[4]: reached take profit or not
    state[5]: reached stop loss or not
    state[6]: positions duration
    state[7]: enter price
    state[8]: percentage of position up day
    state[9]: percentage of position down day
    state[10:]: indicators
    """
    def __init__(
        self,
        trade_timeframe:str,                # Trading timeframe
        df: pd.DataFrame,                   # Processed data
        state_space: int,                   # Number of observation state
        reward_scaling: float,              # Initial cash balance
        indicators: list[str],              # indicator to be used in observation
        buy_trading_fee: float,             # buy fee as percentage
        sell_trading_fee: float,            # sell fee as percentage
        day:int=1,                          # start from 1 to calculate price_change of initial state
        initial:bool = True,                # is that intial state or not

    ):

        self.day = day
        self.trade_timeframe = trade_timeframe
        self.df = df
        self.buy_trading_fee = buy_trading_fee
        self.sell_trading_fee = sell_trading_fee
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.indicators = indicators
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(self.state_space,))
        self.terminal = False
        self.initial = initial
        
       
        #initialize reward
        self.previous_position = 0
        self.reward = 0
        self.trades = 0
        self.episode = 0
        self.action_memory = []
        self.trade_profit_memory = []
        self.reward_memory = []
        self.unallocated_reward_memory = []
        self.cost_memory = []
        self.state_memory = []
        self.asset_value_change_memory = []
        self.ep_reward_list = []
        self.ep_profit_list = []
        self.ep_discount_profit_list = [] 
        self._seed()

        #initialize state
        self.state = self._initiate_state()

    def get_trades_info(self):
        stat_dict = {}
        df = pd.DataFrame({"action":self.action_memory})
        for action in range(3):
            g = df['action'].ne(df['action'].shift()).cumsum()
            g = g[df['action'].eq(action)]
            g = g.groupby(g).count().sort_values()
            stat_dict[action] = g.describe()[["count","mean","std","min","25%","50%","75%","max"]]
        long_short_ratio = stat_dict[1]["count"]/stat_dict[2]["count"] if stat_dict[2]["count"]>0 else 0  
        long_avg_leng = stat_dict[1]["mean"]
        short_avg_len = stat_dict[2]["mean"]
        return long_short_ratio, long_avg_leng, short_avg_len
    
    
    
    def get_reward_memory(self):
        return self.reward_memory
    
    def get_asset_value_change_memory(self):
        return self.asset_value_change_memory
    
    def get_unallocated_reward_memory(self):
        return self.unallocated_reward_memory
    
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
        """
        The main logic for each time step.
        If the action is -1, short all portfolio. 
        If the action is 0, do nothing
        If the action is +1 long all portfolio. 
        The agent will do intermediate go netral first 
        if from long to short and vice-verse.

        Args:
            action (_type_): predicted action

        Returns:
            state (array): new state after taking action
            reward (float): the reward receive after taking action
            info: supplement info
            terminal (bool): is that the last state or not
        """
        
        '''
        
        '''
        #check if it last day
        self.terminal = (self.day >= len(self.df))

        if self.terminal:
            self.trade_profit = self.assumed_trade_profit
            self.reward = self.unallocated_reward
            self.unallocated_reward = 0
            self.state = [0 for i in range(self.state_space)]
            self.action_memory.append(int(action))
            self.reward_memory.append(self.reward)
            self.unallocated_reward_memory.append(self.unallocated_reward)
            self.trade_profit_memory.append(self.trade_profit)
            self.asset_value_change_memory.append(0)
            self.cost_memory.append(0)
            self.state_memory.append(self.state)
            return self.state, self.reward, self.terminal, {}

        else:
            self.previous_position = int(self.state[0])
            self.assumed_trade_profit = self.state[2]
            unallocated_reward = self.state[3]
            reached_take_profit = int(self.state[4])
            reached_stop_loss = int(self.state[5])
            position_duration = int(self.state[6])
            enter_price = int(self.state[7])
            up_day_num = int(self.state[8])
            down_day_num = int(self.state[9])
            action = int(action)
            data = self.df.iloc[self.day,:]         #type: ignore
            current_price = data["Close"]
            previous_price = self.df.iloc[self.day-1,:]["Close"]
            price_change = (current_price-previous_price)/previous_price
            position_price_change = (current_price-enter_price)/enter_price if enter_price>0 else 0
            market_trend = self.get_market_trend(data)
            
            asset_value_change = 0
            self.assumed_trade_profit = 0
            match self.previous_position:
                case 1:
                    asset_value_change = price_change
                    self.assumed_trade_profit = position_price_change
                case 2:
                    asset_value_change = -price_change
                    self.assumed_trade_profit = -position_price_change
    
            #Get reward
            self.reward, trade_profit_before_cost, \
                self.position, trade_cost, trade_num, \
                    self.unallocated_reward, \
                        reached_take_profit, reached_stop_loss \
                            = rwd.get_reward_as_immediate(
                                timestep=self.day,
                                position = self.previous_position,
                                action=action,
                                buy_trading_fee=self.buy_trading_fee,
                                sell_trading_fee=self.sell_trading_fee,
                                timeframe=self.trade_timeframe,
                                assumed_trade_profit=self.assumed_trade_profit,
                                asset_value_change=asset_value_change,
                                reached_take_profit=reached_take_profit,
                                reached_stop_loss=reached_stop_loss,
                                unallocated_reward=unallocated_reward,
                                market_trend=market_trend
            )
            self.before_action_assumed_trade_profit = self.assumed_trade_profit
            if (self.position == self.previous_position):
                position_duration+=1
                up_day_num += 1 if asset_value_change > 0 else 0
                down_day_num += 1 if asset_value_change < 0 else 0
            else:
                position_duration=0
                self.assumed_trade_profit = 0
                enter_price = current_price
                up_day_num = 0
                down_day_num = 0
            
            self.trade_profit = trade_profit_before_cost - trade_cost
            self.trades += trade_num
            
            up_day_ratio = up_day_num/position_duration if position_duration > 0 else 0
            down_day_ratio = down_day_num/position_duration if position_duration > 0 else 0
                    
            self.state = [self.position, price_change, self.assumed_trade_profit, \
                self.unallocated_reward, reached_take_profit, reached_stop_loss, \
                    position_duration, enter_price,up_day_ratio,down_day_ratio] \
                    + data[self.indicators].tolist()

            # Update memory
            self.action_memory.append(action)
            self.reward_memory.append(self.reward)
            self.unallocated_reward_memory.append(self.unallocated_reward)
            self.trade_profit_memory.append(self.trade_profit)
            self.asset_value_change_memory.append(asset_value_change)
            self.cost_memory.append(trade_cost)
            self.state_memory.append(self.state)
            
            self.day+=1
            
            return self.state, self.reward, self.terminal, {} 
    
    def get_market_trend(self,data):
        return data['sma_3_10_level1_lag_1'] > 0
    
    def get_accumulated_cost(self):
        """
        Get accumulated profit

        Returns:
            float: accumulated profit
        """
        return np.exp(np.log(np.array(self.cost_memory)+1).sum())-1
    
    def get_discount_sum_profit(self,gamma:float=0.99):
        """ Get cummulative profit with discount factor gamma

        Args:
            gamma (float, optional): _description_. Defaults to 0.99.

        Returns:
            _type_: _description_
        """
        return np.exp(np.dot(np.log(np.array(self.trade_profit_memory)+1),
                      [gamma ** i for i in range(len(self.trade_profit_memory))]))-1
    
    def get_win_rate(self):
        """Return win rate, long win rate, short win rate

        Returns:
            _type_: _description_
        """
        df = pd.DataFrame({"action":self.action_memory,"profit":self.trade_profit_memory})
        df["position"] = df["action"].shift(-1)
        open_position_df = df.loc[(df["position"]!=df["position"].shift()) & (df["position"]!=0) ]
        long_df = open_position_df.loc[df["position"]==1]
        short_df = open_position_df.loc[df["position"]==2]
        n_win_long = len(long_df[df["profit"]>0])
        n_win_short  = len(short_df[df["profit"]>0])
        n_position = len(long_df)+len(short_df)
        long_win_rate = n_win_long/len(long_df) if n_win_long > 0  else 0
        short_win_rate = n_win_short/len(short_df) if n_win_short > 0  else 0            
        win_rate = (n_win_long+n_win_short)/n_position if n_position > 0 else 0
        return long_win_rate, short_win_rate, win_rate
    
    def get_total_reward(self,gamma:float=0.99):
        """Get cummulative reward with discount factor gamma

        Args:
            gamma (float, optional): _description_. Defaults to 0.99.

        Returns:
            _type_: _description_
        """
        return np.dot(self.reward_memory,[gamma ** i for i in range(len(self.reward_memory))])
    
    def get_sum_profit(self):
        """
        Get accumulated profit

        Returns:
            float: accumulated profit
        """
        return np.exp(np.log(np.array(self.trade_profit_memory)+1).sum()) - 1

    def reset(self):
        ''' Reset the envirnoment'''
        
        # Loging
        long_short_ratio, long_avg_len, short_avg_len = self.get_trades_info()
        gamma = wandb.config["gamma"] if "gamma" in wandb.config.keys() else 1
        ep_profit = self.get_sum_profit()
        ep_discount_profit = self.get_discount_sum_profit(gamma)
        ep_reward = self.get_total_reward(gamma)
        self.ep_profit_list.append(ep_profit)
        self.ep_reward_list.append(ep_reward)
        self.ep_discount_profit_list.append(ep_discount_profit)
        long_win_rate, short_win_rate, win_rate = self.get_win_rate()
        
        df = pd.DataFrame({
            'profit':self.ep_profit_list,
            'discount_profit':self.ep_discount_profit_list,
            'reward':self.ep_reward_list
        })
        
        wandb.log({
            'trade_profit': ep_profit,
            'discount_trade_profit': ep_discount_profit,
            'total_cost' : self.get_accumulated_cost(),
            'n_trades': self.trades,
            'ep_reward': ep_reward,
            'win_rate': win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'long_short_ratio': long_short_ratio,
            'avg_long_duration': long_avg_len,
            'avg_short_duration': short_avg_len,
            'discount_profit_mean_10': df["discount_profit"].tail(10).mean(),
            'profit_mean_10': df["profit"].tail(10).mean(),
            'reward_mean_10': df["reward"].tail(10).mean()
            })
        
        #Reset
        self.day = 1
        self.state = self._initiate_state()
        self.trades = 0
        self.terminal = False
        self.reward_memory = []
        self.unallocated_reward_memory = []
        self.action_memory = []
        self.trade_profit_memory = []
        self.asset_value_change_memory = []
        self.cost_memory = []
        self.episode += 1

        return self.state

    def _initiate_state(self):
        """ Initialize the state

        Returns:
            _type_: Return the initial state
        """
        
        data = self.df.iloc[self.day-1,:]     #type: ignore
        self.state = [0 for i in range(10)] + data[self.indicators].tolist()
        # Update memory
        self.action_memory.append(0)
        self.reward_memory.append(0)
        self.unallocated_reward_memory.append(0)
        self.trade_profit_memory.append(0)
        self.asset_value_change_memory.append(0)
        self.cost_memory.append(0)
        self.state_memory.append(self.state)    
        return self.state
        
    def render(self, mode="human", close=False):
        """ Render the prediction sequences

        Args:
            mode (str, optional): _description_. Defaults to "human".
            close (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        position_state = {0:"NEUTRAL",1:"LONG",2:"SHORT"}
        print("{}: Previous:{} | Action:{} | Rwd:{} | UnRwd:{} | Profit:{} | B-ATP:{} | A-ATP:{} | {}"
            .format(
                self.day-1, 
                position_state[self.previous_position], 
                position_state[self.position], 
                round(self.reward,5),
                round(self.unallocated_reward,5),
                round(self.trade_profit,5),
                round(self.before_action_assumed_trade_profit,5),
                round(self.assumed_trade_profit,5), 
                self.terminal
            ))
        
        return self.state