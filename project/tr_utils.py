from random import randint
import pandas as pd

def under_rebalance(data,target_col):
    '''
    Under sampling. All categories have data numbers equal to the smallest one
    
    Params:
    - data: input dataframe
    - target_col: name of the target col

    Return: balanced dataframe
    '''
    val_counts = data[target_col].value_counts().sort_index()
    cat_length = val_counts.min()
    return fix_rebalance(cat_length=cat_length,data=data,target_col=target_col)

def over_rebalance(data,target_col):
    '''
    Over-sampling. All categories have data numbers equal to the largest one

    Params:
    - data: input dataframe
    - target_col: name of the target col
    
    Return: balanced dataframe
    '''
    val_counts = data[target_col].value_counts().sort_index()
    cat_length = val_counts.max()
    return fix_rebalance(cat_length=cat_length,data=data,target_col=target_col)


def fix_rebalance(cat_length,data,target_col):
    ''' 
    All categories have a fix number of data
    
    Params:
    - cat_length: wanted number of data for each categories
    - data: input dataframe
    - target_col: name of the target col
    Return: balanced dataframe
    '''
    val_counts = data[target_col].value_counts().sort_index()
    
    data_list = []
    for i in val_counts.index:
        data_cat = data[data[target_col]==i]
        times = int(cat_length/val_counts[i])
        remainer = cat_length - times * val_counts[i] 
        for j in range(0,times):
            data_list.append(data_cat) 
        data_remainder =  data_cat.sample(
            remainer,
            axis = 0,
            random_state=randint(0,100))
        data_list.append(data_remainder)
    balanced_data = pd.concat(data_list).sample(
        frac=1,
        random_state=randint(0,100)
    )
    return balanced_data

def first_time_go_above_price(start_time,end_time,target_price,granular_data):
    ''' 
    Find the time in ms that take profit and stop loss condition satisified in the granular data for short trade
    Return time in unix utc milisecond.
    '''
    lookup_data = granular_data.iloc[
        (granular_data.index>=start_time) & 
        (granular_data.index<end_time)].copy()

    first_index = lookup_data.loc[lookup_data["High"]>=target_price].index.min()
    return first_index

def first_time_go_below_price(start_time,end_time,target_price,granular_data):
    ''' 
    Find the time in ms that take profit and stop loss condition satisified in the granular data for short trade
    Return time in unix utc milisecond.
    '''
    lookup_data = granular_data.iloc[
        (granular_data.index>=start_time) & 
        (granular_data.index<end_time)].copy()

    first_index = lookup_data.loc[lookup_data["Low"]<target_price].index.min()
    return first_index