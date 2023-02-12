from config import REWARD as rwd

def calculate_reward(trade_profit:float, position:int, action:int, 
                                  take_profit_rate:float, stop_loss_rate:float):
    ''' 
    Calculate reward with regard to stop loss and take profit
    It encorages agent to keep position when stop loss not reached
    '''

    if (trade_profit > take_profit_rate) or (trade_profit < stop_loss_rate):
        return trade_profit
    else:
        if (trade_profit>0) and (position == action):
            return trade_profit
        else:
            return 0.0           

def calculate_reward_beta(trade_profit:float, position:int, action:int, 
                                  take_profit_rate:float, stop_loss_rate:float):
    ''' 
    Calculate reward with regard to stop loss and take profit
    It encorages closing position when either take profit or stop loss reached 
    It encourages keeping position open if stop loss not reached
    '''

    if (trade_profit > take_profit_rate):
        return trade_profit if (action != position) else trade_profit * rwd["not_take_profit_penalty"]
    elif (trade_profit < stop_loss_rate):
        return trade_profit if (action != position) else trade_profit * rwd["not_stop_loss_penalty"]
    elif (trade_profit>0) and (position == action):
        return trade_profit 
    else:
        return 0
                
def get_reward_beta(state,new_price,action, env):
    '''
    Reward considering accumulated multiple of asset value, devided by number of elapsed day
  
    Params:
    - state: previous state
    - action: predicted action 0:Neutral, 1: Long, 2: Short
    - new_price: the new price

    Return: 
    - reward = (accumulated_profit + trade_reward - trade_cost - no_trade_penalty)/(day+1)
    '''
    
    position = int(state[0])
    enter_price = float(state[2])
    
    price_change_since_enter = (new_price-enter_price)/enter_price if enter_price>0 else 0
    
    trade_profit = price_change_since_enter if position == 1 else - price_change_since_enter

    no_trade_penalty = rwd["no_trade_penalty"] if ((action==0) and (position==0)) else 0

    trade_reward =  calculate_reward(
                trade_profit = trade_profit,
                position = position,
                action = action,
                take_profit_rate = env.take_profit_rate,
                stop_loss_rate = env.stop_loss_rate
    )        

    accumulated_profit = env.get_accumulated_profit()
    
    trade_cost = get_trade_cost(
        position,
        action,
        env.buy_trading_fee,
        env.sell_trading_fee
    )

    reward = (accumulated_profit + trade_reward - trade_cost - no_trade_penalty)/(
        1+rwd["money_cost"])**(1+env.day)
    
    return reward, trade_reward, trade_cost

def get_trade_cost(position,action,buy_trading_fee,sell_trading_fee):
    '''
    Trading penalty equal trading cost
    '''
    c_action = -1 if action == 2 else action
    c_position = -1 if position == 2 else position

    trading_cost = 0
    match c_action - c_position:
        case 0:
            pass
        case 1:
            trading_cost = buy_trading_fee
        case 2:
            trading_cost = 2 * buy_trading_fee
        case -1:
            trading_cost = sell_trading_fee
        case -2:
            trading_cost = 2 * sell_trading_fee

    return trading_cost

def get_reward(state, new_price:float, action:int, env):
                        
    '''
    Reward function for simple agent

    Params:
    - state: previous state
    - action: predicted action 0:Neutral, 1: Long, 2: Short
    - new_price: the new price

    Return: reward 
    '''

    position = int(state[0])
    enter_price = float(state[2])
    
    reward = 0.0

    price_change_since_enter = (new_price-enter_price)/enter_price if enter_price>0 else 0

    match int(position):
        case 2: # In short position
            reward = calculate_reward_beta(
                trade_profit = -price_change_since_enter,
                position = position,
                action = action,
                take_profit_rate = env.take_profit_rate,
                stop_loss_rate = env.stop_loss_rate
            )
            match action:
                case 0:
                    # Close short position 
                    reward -=  env.buy_trading_fee
                case 1:
                    # Close short and then long     
                    reward -= 2 * env.buy_trading_fee
                case 2:
                    # Stay short
                    pass
                case other:
                    raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
    
        case 0: # In neural position
            match action:
                case 0:
                    # Stay netral
                    reward = -env.money_sleep_cost
                case 1:
                    # From netral to long
                    reward = -env.buy_trading_fee
                case 2:
                    # From neutral to short
                    reward = -env.sell_trading_fee
        
                case other:
                    raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
            
        case 1: # In long position
            reward = calculate_reward_beta(
                trade_profit = price_change_since_enter,
                position = position,
                action = action,
                take_profit_rate = env.take_profit_rate,
                stop_loss_rate = env.stop_loss_rate
            )
            match action:
                case 0:
                    # Close long position
                    reward -= env.sell_trading_fee
                case 1:
                    # Stay long
                    pass
                case 2:
                    # Close long position and then short
                    reward -= 2 * env.sell_trading_fee
                case other:
                    raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
        case other:
            raise(ValueError("Position must be 0,1,2. We got {}".format(position)))
    
    return reward