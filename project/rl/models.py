# DRL models from Stable Baselines 3
from __future__ import annotations
from cv2 import transform

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import ARS, TRPO
import matplotlib.pyplot as plt
import config as cf
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

#Available models
MODELS = {
    "a2c": A2C, 
    "ppo": PPO, 
    # "ars":ARS,
    # "trpo":TRPO,
    "dqn": DQN
}

MODEL_KWARGS = {x: cf.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0]) #type: ignore
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])  #type: ignore
        return True

class DRLTradeAgent:
    """ Class of deep reinforcement learning agent for trading.
        The agent choose to long, short or stay neutral whole asset each time frame.
        The goal is to maximize the multiple of asset value

    Attributes
    ----------
        env (gym environment): gym environment used in the last training
        test_env (gym environment): gym envirnment used in the last prediction
        latest_model (): the last model used for training or prediction
        trade_number (int): number of traded made in the last prediction
        action_memory (list[int]): history of action in the last prediction
        cost_memory (list[float]): history of cost
        reward_memory (list[float]): history of reward
        trade_profit_memory (list[float]): history of trade profit
        trade_unallocated_reward_memory (list[float]): history of unallocated reward memory
        

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
        """_summary_

        Args:
            env (gym environment): trained model 
            test_env (gym environment, optional): test model. Defaults to None.
        """
        
        self.env = env
        self.test_env = test_env
        self.latest_model = None
        self.trade_number = 0
        self.action_memory = []
        self.cost_memory = []
        self.reward_memory = []
        self.trade_profit_memory = []
        self.unallocated_reward_memory = []

    def set_test_env(self, env):
        self.test_env = env

    def set_env(self,env):
        self.env = env

    def get_model(self,
                    model_name:str,
                    tensorboard_log:str="",
                    policy_kwargs=None,
                    model_kwargs=None,
                    policy:str="MlpPolicy",
                    verbose=1,
                    seed=None):
        """ Get model

        Args:
            model_name (str): name of the model, exp 'ppo'
            gamma (float): discounted factor
            tensorboard_log (str): path to tensorboard_log directory to store training log
            policy (str, optional): used policy. Defaults to "MlpPolicy".
            verbose (int, optional): verbose mode. Defaults to 1.
            seed (int, optional): random seed. Defaults to None.

        Raises:
            NotImplementedError: not supportive model

        Returns:
            _type_: the model
        """
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]
        
        tensorboard_log = cf.TENSORBOARD_LOGDIR if tensorboard_log == "" \
            else f"{cf.TENSORBOARD_LOGDIR}{tensorboard_log}/"
        
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    def train_model(self, model, total_timesteps:int, reset_num_timesteps:bool=False, 
                    progress_bar:bool = True,checkpoint:bool=False, save_frequency:int=cf.CHECKPOINT_CALLBACK["frequency"],catalog_name:str="smt"):
        
        """ Train model.

        Args:
            model (_type_): used model to learn
            total_timesteps (int): total number of timesteps to train
            reset_num_timesteps (bool, optional): reset timesteps count or not. Defaults to False.
            progress_bar (bool, optional): show progress bar or not. Defaults to True.
            checkpoint (bool, optional): save in checkpoint or not. Defaults to False.
            save_frequency (int, optional): frequency for checkpoitn save. Defaults to cf.CHECKPOINT_CALLBACK["frequency"].
            catalog_name (str, optional): name of folder for checkpoint save. Defaults to "smt".

        Returns:
            _type_: trained model
        """
        
        self.env.reset()
        checkpoint_callback = None
        callbacks = []

        callbacks.append(
            TensorboardCallback()
        )

        if checkpoint:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_frequency, 
                save_path="{}{}/".format(cf.CHECKPOINT_CALLBACK["save_dir"],catalog_name)
            )
            callbacks.append(checkpoint_callback)

        model = model.learn(
            total_timesteps = total_timesteps,
            reset_num_timesteps = reset_num_timesteps,
            progress_bar = progress_bar,
            callback = callbacks
        )

        self.latest_model = model

        print(f"Total trained timestep: {model.num_timesteps}")
        return model

    def predict(self,model,environment,deterministic:bool=False, render:bool=True):
        """ Make prediction with by model

        Args:
            model (_type_): model used for prediction
            environment (_type_): used environment
            deterministic (bool, optional): predict the action with highest prob. Defaults to False.
            render (bool, optional): print prediction and info list. Defaults to True.
        
        Returns:
            result_df (type: pd.DataFrame): result table with detail
        
        """
        
        self.test_env = environment
        self.action_memory, self.reward_memory, \
            self.unallocated_reward_memory, self.trade_profit_memory, \
                self.cost_memory, self.trade_number = self.DRL_prediction(
            model=model,
            test_env=environment,
            deterministic=deterministic,
            render=render
        )
        self.latest_model = model
        self.result_df = self.make_result_data()

    @staticmethod
    def DRL_prediction(model, test_env, deterministic:bool, render:bool):
        
        """make a prediction"""
        action_memory = []
        reward_memory = []
        unallocated_reward_memory = []
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
                unallocated_reward_memory = test_env.get_unallocated_reward_memory()
                trade_profit_memory = test_env.get_trade_profit_memory()
                cost_memory = test_env.get_cost_memory()
                trade_number = test_env.get_trade_number()
                break

        return action_memory, reward_memory, unallocated_reward_memory, trade_profit_memory, cost_memory, trade_number 
    
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


    def plot_reward(self,log:bool=False,dpi=720):
        data = self.result_df.copy()

        plt.figure(figsize=(12,6),dpi=dpi)
        
        if (self.latest_model!=None):
            timesteps = self.latest_model.num_timesteps
            gamma = self.latest_model.gamma
            plt.title(f"RobotTrader Performance: gamma={gamma}, timestep={timesteps}")
        else:
            plt.title(f"RobotTrader Performance")
            
        plt.plot(data.index,np.exp(data["reward"])-1,linewidth=1, color="tab:gray",label="reward")
        plt.plot(data.index,np.exp(data["unallocated_reward"])-1,linewidth=1, color="tab:blue",label="unallocated_reward")
        plt.plot(data.index,data["trade_profit"],linewidth=1, color="tab:green",label="trade_profit")
        
        plt.ylabel("Reward and profit")
        if log:
            plt.yscale("log")
        plt.grid(True,which="both")
        plt.xlabel("Timeframe")
        plt.legend()
        plt.show()
    
    def plot_results(self,log:bool=False,dpi=720):  
        data = self.result_df.copy()        
        plt.figure(figsize=(12,6),dpi=dpi)
        if (self.latest_model!=None):
            params_str = \
                f"num_timestep: {self.latest_model.num_timesteps}\n"\
                f"gamma: {self.latest_model.gamma}\n"\
                f"batch_size: {self.latest_model.batch_size}\n"\
                f"seed: {self.latest_model.seed}\n"\
                f"learning_rate: {self.latest_model.learning_rate}\n"
            
            plt.text(0.05,0.95, params_str,
                    fontsize=8,
                    horizontalalignment="left",
                    verticalalignment='top',
                    transform = plt.gca().transAxes)
            plt.title("Predict Result")
            plt.plot(data.index,data["cumsum_trade_profit"],linewidth = 1,color="tab:green",label="real_asset_value_after_cost")
            plt.plot(data.index,data["cumsum_cost"]+1,linewidth = 1,color="tab:red",label="cumsum_trading_cost")
            plt.plot(data.index,data["relative_price"],linewidth = 1,color="tab:cyan",label="relative_price")
            plt.ylabel("Multiple")
            plt.xlabel("Timeframe")
            plt.legend(loc="upper right")
            if log:
                plt.yscale("log")
            plt.show()
        else:
            print("No available data")

    def describe_trades(self):
        stat_dict = {}
        df = self.result_df
        for action in range(3):
            g = df['action'].ne(df['action'].shift()).cumsum()
            g = g[df['action'].eq(action)]
            g = g.groupby(g).count().sort_values()
            stat_dict[action] = g.describe()[["count","mean","std","min","25%","50%","75%","max"]]
        print("Trade count and duration statistics:")
        print(pd.DataFrame(stat_dict))       

    def make_result_data(self):
        result_df = pd.DataFrame({
            "reward":self.reward_memory,
            "unallocated_reward":self.unallocated_reward_memory,
            "trade_profit":self.trade_profit_memory,
            "action":self.action_memory,
            "cost": self.cost_memory
            })

        result_df["log_trade_profit"] = np.log(result_df["trade_profit"]+1)
        result_df["cumsum_trade_profit"] = np.exp(result_df["log_trade_profit"].cumsum(axis=0))
        result_df["log_cost"] = np.log(1+result_df["cost"])
        result_df["cumsum_cost"] = np.exp(result_df["log_cost"].cumsum(axis=0)) -1
        result_df["price"] = self.test_env.df["Close"].values
        result_df["relative_price"] = result_df["price"]/result_df.iloc[0,:]["price"]
        result_df["cumsum_reward"] = np.exp(result_df["reward"].cumsum(axis=0)) - 1
        return result_df