from random import randint
from time import time
import pandas as pd
from scipy.optimize import fsolve
from math import exp    
from config import TIMEFRAMES_IN_MS
import numpy as np

def get_name_with_kwargs(name:str,kwargs:dict):
    name_str:str=name
    for key in kwargs.keys():
        name_str += f"{key}_{kwargs[key]}"
    return name_str

def under_sampling_rebalance(data:pd.DataFrame,target_col:str):
    '''
    Under sampling. All categories have data numbers equal to the smallest one
    
    Params:
    - data: input dataframe
    - target_col: name of the target col

    Return: balanced dataframe
    '''
    val_counts = data[target_col].value_counts().sort_index()
    return fix_sampling_rebalance(cat_length=val_counts.min(),data=data,target_col=target_col)

def over_sampling_rebalance(data:pd.DataFrame,target_col:str):
    '''
    Over-sampling. All categories have data numbers equal to the largest one

    Params:
    - data: input dataframe
    - target_col: name of the target col
    
    Return: balanced dataframe
    '''
    val_counts = data[target_col].value_counts().sort_index()
    return fix_sampling_rebalance(cat_length=val_counts.max(),data=data,target_col=target_col)


def fix_sampling_rebalance(cat_length:int,data:pd.DataFrame,target_col:str):
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
        times = int(cat_length/val_counts[i])               #type: ignore
        remainer = cat_length - times * val_counts[i]       #type: ignore
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

def first_time_go_above_price(start_time:int,end_time:int,target_price:float,granular_data:pd.DataFrame):
    ''' 
    Find the time in ms that take profit and stop loss condition satisified in the granular data for short trade
    Return time in unix utc milisecond.
    '''
    lookup_data = granular_data.iloc[(granular_data.index>=start_time) & 
        (granular_data.index<end_time)].copy()

    first_index = lookup_data.loc[lookup_data["High"]>=target_price].index.min()
    return first_index

def first_time_go_below_price(start_time:int,end_time:int,target_price:float,granular_data:pd.DataFrame):
    ''' 
    Find the time in ms that take profit and stop loss condition satisified in the granular data for short trade
    Return time in unix utc milisecond.
    '''
    lookup_data = granular_data.iloc[(granular_data.index>=start_time) & 
        (granular_data.index<end_time)].copy()

    first_index = lookup_data.loc[lookup_data["Low"]<target_price].index.min()
    return first_index

def calculate_weight(y_train:str):
        counts = pd.DataFrame(y_train).value_counts().sort_index()
        weights = 1/counts * counts.sum()/(len(counts))
        return {i:weights[i] for i in counts.index.map(lambda x:x[0])}

def init_imbalanced_bias(y_train):
    """
    To handle imbalanced classification, provide initial bias list and class weight dictionary to 2 places in a tf classifier
    
    In the last layer of classifier: tf.keras.layers.Dense(..., bias_initializer = bias_init)
    
    Args:
        y_train: list of class label for train, ex. [1,2,0,1,2 ...]
    Returns:
        bias_init:list e.g. [0.3222079660508266, 0.1168690393701237, -0.43907701967633633]
    Examples:
        bias_init = init_imbalanced_class_weight_bias(df=train_df, lable=label)
    """
    
    # to deal with imbalance classification, calculate class_weight 
    d = dict(pd.DataFrame(y_train).value_counts().sort_index())

    # define classes frequency list
    frequency = list(list(d.values())/sum(d.values()))  #type: ignore

    # define equations to solve initial bias
    def eqn(x, frequency=frequency):
        sum_exp = sum([exp(x_i) for x_i in x])
        return [exp(x[i])/sum_exp - frequency[i] for i in range(len(frequency))]

    # calculate init bias
    bias_init = fsolve(func=eqn, x0=[0]*len(frequency)).tolist() #type: ignore

    return bias_init
