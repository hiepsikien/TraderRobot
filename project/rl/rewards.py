from argparse import Action
import numpy as np
import config as cf

def get_money_cost(timeframe:str,risk_free_annual_return:float):
    """ Get money cost for a specific timeframe according to a
    annual risk free return

    Args:
        timeframe (str): the timeframe, such as '1d','4h','1h'
        risk_free_annual_return (float): yearly risk free return, such as bank interest rate

    Raises:
        ValueError: raise error if not defined timeframe argument

    Returns:
        money_cost (float)
    """
    if timeframe not in cf.TIMEFRAMES_IN_MS.keys():
        raise ValueError(f"No config info for timeframe {timeframe}")
    else:
        multiple = cf.TIMEFRAMES_IN_MS["1d"]/cf.TIMEFRAMES_IN_MS[timeframe]
        money_cost = np.exp(risk_free_annual_return/(multiple*365)) - 1
        
    return money_cost    

def get_linear_descending(number:int,max:int):
    return 0 if number>max else (max-number)/max

def get_linear_ascending(duration:int,max:int):
    return 1 if duration>max else duration/max


def calculate_reward_no_stop_loss(assumed_trade_profit:float, 
                                         position:int, 
                                         action:int,
                                         unallocated_reward: float, 
                                         asset_value_change:float,
                                         reached_take_profit:int,
                                         reached_stop_loss:int,
                                         market_trend:int):
    
    NO_TRADE_PENALTY = cf.REWARD["no_trade_penalty"]
        
    bonus = 0
    reward = 0
    trade_profit = 0
    if position == 0:
        if action == 0:
            reward = np.log(1 - NO_TRADE_PENALTY)
        else:
            reward = 0
    else:
        if action != position:
            reward = np.log(1+assumed_trade_profit)
            trade_profit = assumed_trade_profit
            
            # Encorage open short position
            if position == 2:
                if trade_profit > 0.1:
                    bonus = 0.1
                elif trade_profit > 0.05:
                    bonus = 0.05
                elif trade_profit > 0.01:
                    bonus = 0.01
            
        if action == position:
            reward = np.log(1+asset_value_change)
    
    reward += bonus
    
    new_position = action
    unallocated_reward = 0
    reached_take_profit = 0
    reached_stop_loss = 0
              
    return reward, trade_profit, new_position, unallocated_reward, \
        reached_take_profit, reached_stop_loss
    
def calculate_reward_with_stop_loss_beta(assumed_trade_profit:float, 
                                         position:int, 
                                         action:int,
                                         unallocated_reward: float, 
                                         asset_value_change:float,
                                         reached_take_profit:int,
                                         reached_stop_loss:int):
    
    """ Calculate reward regarding take profit and stop loss more sophisticately.
        When take profit or stop loss reached, it is given big reward or heavy penalty but
        is not forced to close.
        There is flag recored take profit and stop loss has been reached yet.
        If it is, the reward will be only based on, not full, new change in asset value. 
        When it close the position, it will receive full the unallocated. 
        The function assigned only part of total reward, keep some to assign later.
        The point is the total reward must respect to cumsum of log(1+trade_profit).
        Maximizing the reward will maximize the profit.

    Args:
        assumed_trade_profit (float): assumed trade profit
        position (int): current position
        action (int): action
        unallocated_reward (float): accumulated reward not allocated yet
        asset_value_change (float): value change of the asset comparing to previous timeframe
        reached_take_profit (int): 1 if has reached take profit since open or 0 if not
        reached_stop_loss (int): 1 if has reached stop loss since open or 0 if not

    Returns:
        reward (float): the reward
        trade_profit (float): trade profit
        new_position (int): new position
        unallocated_reward (float): updated unclaimed reward
        reached_take_profit (int): updated reached_take_profit
        reached_stop_loss (int): updated reached_stop_loss
    """
    
    DISCOUNT_RATE = cf.REWARD["discount_rate"]
    AMPLIFIED_RATE = cf.REWARD["amplified_rate"]
    TAKE_PROFIT_RATE = cf.REWARD["take_profit_rate"]
    STOP_LOSS_RATE = cf.REWARD["stop_loss_rate"]
    NO_TRADE_PENALTY = cf.REWARD["no_trade_penalty"]
    
    new_position = action
    new_reward = asset_value_change
    unallocated_reward+=new_reward
    reward = 0
    
    bonus_reward = 0
    
    if new_position!=position:
        # If close position, give all unallocated reward, reset variables to 0
        reached_take_profit = 0
        reached_stop_loss = 0
        reward = unallocated_reward
        unallocated_reward = 0
    else:
        if assumed_trade_profit > TAKE_PROFIT_RATE:
            # For case profit is over take profit threshold
            # If it is the first time, give reward as discounted of unallocated_reward, update reached_take_profit flag
            # Else, only give reward as a positive discounted of add_reward if add_reward greater than 0, else penalty it with heavier negative reward
            
            if (reached_take_profit == 0):
                reached_take_profit = 1
                bonus_reward = 0.03
                reward = DISCOUNT_RATE * unallocated_reward + bonus_reward
            else:
                reward = DISCOUNT_RATE * new_reward if new_reward > 0 \
                    else AMPLIFIED_RATE * new_reward
                
        elif assumed_trade_profit < STOP_LOSS_RATE:
            # For case loss is over stop loss threshold
            # If it is first time reached, give it an amiplified of unallocated_reward, which is negative
            # Else, give it a discounted reward if added_reward positive, or amplified if negative
            if (reached_stop_loss == 0):
                reached_stop_loss = 1
                reward = AMPLIFIED_RATE * unallocated_reward
            else:
                reward = DISCOUNT_RATE * new_reward if new_reward > 0  \
                    else AMPLIFIED_RATE * new_reward
        else:
            # For case either current position in between stop loss and take profit
            # If it already reached either stop loss or take profit before, give discounted positive of added reward if positive or amplified if negative
            # Else, give it discounted of extra added reward if added reward is positive or zero if negative
            if (reached_stop_loss) or (reached_take_profit):
                reward = DISCOUNT_RATE * new_reward if new_reward > 0 \
                    else AMPLIFIED_RATE * new_reward
            else:
                reward = DISCOUNT_RATE * new_reward if new_reward > 0 else 0
            
        if position == 0 and action == 0:
            reward = np.log(1 - NO_TRADE_PENALTY)
        
        unallocated_reward -= reward 
        
    trade_profit = assumed_trade_profit if new_position!=position else 0     
    
    return reward, trade_profit, new_position, unallocated_reward, \
        reached_take_profit, reached_stop_loss

def calculate_reward_with_stop_loss(assumed_trade_profit:float, 
                                    position:int, 
                                    action:int):
    """
    Calculate reward, trade_profit and new position with take_profit and stop_loss event.
    When take_profit or stop_loss event occured, suggested action is overided.

    Args:
        assumed_trade_profit (float): assumed trade profit not considerint take profit/stop loss event
        position (int): current position
        action (int): suggested action


    Returns:
        reward (float), trade_profit (float), new_position (int): reward, profit after stop_loss event, new_position after take profit/stop loss event.
    """
    
    new_position = position
    
    TAKE_PROFIT_RATE = cf.REWARD["take_profit_rate"]
    STOP_LOSS_RATE = cf.REWARD["stop_loss_rate"]
    NO_TRADE_PENALTY = cf.REWARD["no_trade_penalty"]
    
    if assumed_trade_profit > TAKE_PROFIT_RATE:
        reward = np.log(1+TAKE_PROFIT_RATE)
        new_position = 0
        trade_profit = TAKE_PROFIT_RATE
    elif assumed_trade_profit < STOP_LOSS_RATE:
        reward =np.log(1+STOP_LOSS_RATE)
        new_position = 0
        trade_profit = STOP_LOSS_RATE
    else:
        new_position = action
        reward = np.log(1+assumed_trade_profit) if new_position!= position else 0
        trade_profit = assumed_trade_profit if new_position!=position else 0     
    
    if position == 0 and action == 0:
        reward = np.log(1 - NO_TRADE_PENALTY)
    
    return reward, trade_profit, new_position

def get_trade_cost(position:int,
                   new_position:int,
                   buy_trading_fee:float,
                   sell_trading_fee:float):
    """
    Calculate trade cost

    Args:
        position (int): previous position
        new_position (int): new position
        buy_trading_fee (float): buy fee
        sell_trading_fee (float): sell fee

    Returns:
        trade_cost, trade_num (float, int): trade cost and number of trades 
    """
    c_new_position = -1 if new_position == 2 else new_position
    c_position = -1 if position == 2 else position
    
    diff_pos = c_new_position - c_position
    trade_cost = 0
    
    trade_cost = diff_pos*buy_trading_fee if (diff_pos) > 0 \
        else -diff_pos * sell_trading_fee 
    
    return trade_cost, abs(c_new_position-c_position)

def get_reward_as_immediate(timestep:int, 
                            position:int, 
                            action:int, 
                            assumed_trade_profit:float,
                            buy_trading_fee:float, 
                            sell_trading_fee:float,
                            timeframe:str, 
                            unallocated_reward:float,
                            asset_value_change:float,
                            reached_take_profit:int,
                            reached_stop_loss:int,
                            market_trend:int,
                            ):
    """
    Get reward and others 

    Args:
        timestep (int): elapsed timestep
        position (int): current position
        assumed_trade_profit (float): assumed trade profit based on entered position 
        unallocated_reward (float): the reward kept for later distribution
        asset_value_change (float): the change of asset value comparing to previous timeframe
        action (int): suggested action
        env (_type_): training envirnment
        asset_value_change (float): value change of the asset comparing to previous timeframe
        reached_take_profit (int): 1 if has reached take profit since open or 0 if not
        reached_stop_loss (int): 1 if has reached stop loss since open or 0 if not

    Returns:
        reward (float): reward
        trade_profit (float): trade profit, it is take_profit_rate or stop_loss_rate if reached
        new_position (int): the new position, it might be 0, different from suggested action, if take_profit or stop_loss reached
        trade_cost (float): trade cost
        enter_price (float): the new entered price
        trade_num (int): number of trade    
    """
    
    reward, trade_profit, \
        new_position, unallocated_reward, \
            reached_take_profit, reached_stop_loss = calculate_reward_no_stop_loss(
        action = action,
        position = position,
        assumed_trade_profit = assumed_trade_profit,
        unallocated_reward=unallocated_reward,
        asset_value_change=asset_value_change,
        reached_stop_loss=reached_stop_loss,
        reached_take_profit=reached_take_profit,
        market_trend = market_trend,
    )
    
    trade_cost, trade_num = get_trade_cost(
        position = position,
        new_position = new_position,
        buy_trading_fee=buy_trading_fee,
        sell_trading_fee=sell_trading_fee)        
    
    #Discount reward with money cost
    money_cost = get_money_cost(
        timeframe = timeframe,
        risk_free_annual_return = cf.REWARD["risk_free_annual_return"]
    )
    
    reward -= trade_cost + timestep * money_cost
    
    return reward, trade_profit, new_position, trade_cost, trade_num, unallocated_reward,\
        reached_take_profit, reached_stop_loss