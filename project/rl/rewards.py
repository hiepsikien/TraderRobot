from ast import match_case


def calculate_reward_with_stop_loss(trade_profit:float, position:int, action:int, 
                                  take_profit_rate:float, stop_loss_rate:float):
    ''' 
    Calculate reward with regard to stop loss and take profit
    It encorage the reward to keep position as long as stop loss not reached
    '''

    if (trade_profit > take_profit_rate) or (trade_profit < stop_loss_rate):
        return trade_profit
    else:
        if (trade_profit>0) and (position == action):
            return trade_profit
        else:
            return 0.0                      
                
            
def get_reward_simple_env(position:int,action:int,enter_price,current_price:float,
                        take_profit_rate:float, stop_loss_rate:float,buy_trading_fee:float,
                        sell_trading_fee:float, money_sleep_cost:float):
    '''
    Reward function for simple agent

    Params:
    - position: current position 0:Neutral, 1:Long, 2: Short
    - action: predicted action 0:Neutral, 1: Long, 2: Short
    - enter_price: the entered price if it is in long or short, otherwise 0
    - current_price: the current price

    Return: reward 
    '''

    reward = 0.0
    price_change = (current_price-enter_price)/enter_price if enter_price>0 else 0
    match int(position):
        case 2: # In short position
            reward = calculate_reward_with_stop_loss(
                trade_profit = -price_change,
                position = position,
                action = action,
                take_profit_rate = take_profit_rate,
                stop_loss_rate = stop_loss_rate
            )
            match action:
                case 0:
                    # Close short position 
                    reward -=  buy_trading_fee
                case 1:
                    # Close short and then long     
                    reward -= 2 * buy_trading_fee
                case 2:
                    # Stay short
                    pass
                case other:
                    raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
    
        case 0: # In neural position
            match action:
                case 0:
                    # Stay netral
                    reward = -money_sleep_cost
                case 1:
                    # From netral to long
                    reward = -buy_trading_fee
                case 2:
                    # From neutral to short
                    reward = -sell_trading_fee
        
                case other:
                    raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
            
        case 1: # In long position
            reward = calculate_reward_with_stop_loss(
                trade_profit = price_change,
                position = position,
                action = action,
                take_profit_rate = take_profit_rate,
                stop_loss_rate = stop_loss_rate
            )
            match action:
                case 0:
                    # Close long position
                    reward -= sell_trading_fee
                case 1:
                    # Stay long
                    pass
                case 2:
                    # Close long position and then short
                    reward -= 2 * sell_trading_fee
                case other:
                    raise(ValueError("Action must be 0,1,2. We got {}".format(action)))
        case other:
            raise(ValueError("Position must be 0,1,2. We got {}".format(position)))
    
    return reward