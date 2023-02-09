# DRL models from Stable Baselines 3
from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
from sb3_contrib import ARS, TRPO
import matplotlib.pyplot as plt
import config as cf
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

#Available models
MODELS = {
    "a2c": A2C, 
    "ppo": PPO, 
    "ars":ARS,
    "trpo":TRPO,
}

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True

class DRLTradeAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env,test_env=None):
        self.env = env
        self.trade_number = 0
        self.test_env = test_env
        self.action_memory = []
        self.cost_memory = []
        self.reward_memory = []
        self.trade_profit_memory = []
        self.latest_model = []

    def set_test_env(self, env):
        self.test_env = env

    def set_env(self,env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        model_kwargs=None,
        verbose=1,
        seed=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        return MODELS[model_name](
            policy=policy,
            env=self.env,
            verbose=verbose,
            seed=seed,
            tensorboard_log=cf.TENSORBOARD_LOGDIR,
        )

    def train_model(self, model, total_timesteps:int, reset_num_timesteps:bool=False, progress_bar:bool = True,
                    checkpoint:bool=False, save_frequency:int=cf.CHECKPOINT_CALLBACK["frequency"],
                    checkpoint_subdir_name:str="smt"):
        
        self.env.reset()
        checkpoint_callback = None
        if checkpoint:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_frequency, 
                save_path="{}{}/".format(cf.CHECKPOINT_CALLBACK["save_dir"],checkpoint_subdir_name)
            )

        model = model.learn(
            total_timesteps = total_timesteps,
            reset_num_timesteps = reset_num_timesteps,
            progress_bar = progress_bar,
            callback = [TensorboardCallback(),checkpoint_callback]
        )

        self.latest_model = model

        print(f"Total trained timestep: {model.num_timesteps}")
        return model

    def predict(self,model,environment,deterministic:bool=False, render:bool=True):
        self.test_env = environment
        self.action_memory, self.reward_memory, self.trade_profit_memory, self.cost_memory, self.trade_number = self.DRL_prediction(
            model=model,
            test_env=environment,
            deterministic=deterministic,
            render=render
        )
        self.latest_model = model
        result_df = self.make_result_data()
        self.describe_trades(result_df)
        self.plot_multiple(result_df)

    @staticmethod
    def DRL_prediction(model, test_env, deterministic:bool, render:bool):
        
        """make a prediction"""
        action_memory = []
        reward_memory = []
        cost_memory = []
        trade_profit_memory = []
        trade_number = 0
        obs = test_env.reset()
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = test_env.step(action)
            
            if (render):
                test_env.render()
            
            if done:
                action_memory = test_env.get_action_memory()
                reward_memory = test_env.get_reward_memory()
                trade_profit_memory = test_env.get_trade_profit_memory()
                cost_memory = test_env.get_cost_memory()
                trade_number = test_env.get_trade_number()
                print("hit end!")
                break

        return action_memory, reward_memory, trade_profit_memory, cost_memory, trade_number 
    
    def load_model(self,model_name:str,filename:str):
        '''
        Load a model from saved file
        
        Params:
        - model_name: name of the model
        - filepath: path to saved file

        Return: load model
        '''
        path = f"{cf.TRAINED_MODEL_DIR}{model_name}/{filename}"

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(path)
            print(f"Successfully load model from {path}")
        except BaseException:
            raise ValueError("Fail to load model!")
        return model
    
    def load_model_from_checkpoint(self,model_name:str,filename:str):
        '''
        Load a model from saved file
        
        Params:
        - model_name: name of the model
        - filepath: path to saved file

        Return: load model
        '''
        path = cf.CHECKPOINT_CALLBACK["save_dir"]+filename

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(path)
            print(f"Successfully load model from {path}")
        except BaseException:
            raise ValueError(f"Fail to load model from {path}")
        return model

    def save_model(self,model,model_name,filename:str):
        path = f"{cf.TRAINED_MODEL_DIR}/{model_name}/{filename}_{model.num_timesteps}"
        try:
            model.save(path)
            print(f"Successfully save model to {path}")
        except BaseException:
            raise ValueError("Fail to save model!")

    def plot_multiple(self,data:pd.DataFrame):
        ''' 
        Plot trading results as multiple of initial
        '''
        data["y_long"] = -0.5
        data["y_neutral"] = -0.1
        data["y_short"] = -1.5
      
        # only_first_change = data.loc[data["action"] != data["action"].shift()]
        # long = data.loc[data["action"]==1]
        # short = data.loc[data["action"]==2]
        # neutral = data.loc[data["action"]==0]
        plt.figure(figsize=(12,6),dpi=720)
        plt.title("RobotTrader Performance")
        # plt.plot(data.index,data["cumsum_asset_value_change"], linewidth = 1, color="tab:blue",label="assumed_asset_value_after_cost")
        plt.plot(data.index,data["cumsum_trade_profit"],linewidth = 1,color="tab:green",label="real_asset_value_after_cost")
        # plt.plot(data.index,data["cumsum_cost"],linewidth = 1,color="tab:red",label="trading_cost")
        plt.plot(data.index,data["relative_price"],linewidth = 1,color="tab:cyan",label="relative_price")
        # plt.scatter(long.index,long["y_long"],s=1,color="tab:green")             #type: ignore
        # plt.scatter(neutral.index,neutral["y_neutral"],s=1,color="tab:orange")       #type: ignore
        # plt.scatter(short.index,short["y_short"],s=1,color="tab:red")           #type: ignore
        plt.ylabel("Multiples")
        plt.yscale("log")
        plt.grid(True,which="both")
        plt.xlabel("Timeframe")
        plt.legend()
        plt.show()

    def describe_trades(self,result_df):
        stat_dict = {}
        for action in range(3):
            g = result_df['action'].ne(result_df['action'].shift()).cumsum()
            g = g[result_df['action'].eq(action)]
            g = g.groupby(g).count().sort_values()
            stat_dict[action] = g.describe()[["count","mean","std","min","25%","50%","75%","max"]]
        print("Trade count and duration statistics:")
        print(pd.DataFrame(stat_dict))       

    def make_result_data(self):
        result_df = pd.DataFrame({
            "reward":self.reward_memory,
            "trade_profit":self.trade_profit_memory,
            "action":self.action_memory,
            "cost": self.cost_memory
            })

        result_df["log_trade_profit"] = np.log(result_df["trade_profit"]+1)
        result_df["cumsum_trade_profit"] = np.exp(result_df["log_trade_profit"].cumsum(axis=0))
        result_df["log_cost"] = np.log(1-result_df["cost"])
        result_df["cumsum_cost"] = np.exp(result_df["log_cost"].cumsum(axis=0)) -1
        result_df["price"] = self.test_env.df["Close"].values
        result_df["relative_price"] = result_df["price"]/result_df.iloc[0,:]["price"]
        return result_df.copy()
    