import tensorflow as tf
import gym
import numpy as np
from keras.layers import Input, Dense, Add
from keras import Sequential, Model
from keras.optimizers import Adam
import random
from collections import deque
import wandb
tf.keras.backend.set_floatx("float64")

class QNet:
    """ Neural network for DQN
    """
    def __init__(self,
                 state_dim:int, 
                 action_dim:int, 
                 hidden_layers:list[int],
                 learning_rate:float,
                 epsilon: float,
                 eps_decay:float,
                 eps_min:float,
                 epochs: int,
                 variant: str
                 ) -> None:
        """Initialize

        Args:
            state_dim (int): state dimension
            action_dim (int): action dimension
            variant (str): algorithm variant:'vanila','double','duel','duel-double'
            hidden_layers (list[int]): hidden layers 
            learning_rate (float): learning rate
            epsilon (float): epsilon
            eps_decay (float): epsilon decay
            eps_min (float): min epsilon
            epochs (int): number of neural network fit each update
        """
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.epochs = epochs
        self.variant = variant
        self.model = self.create_model()
        
    def create_model(self):
        """Create the model.

        Returns:
            Sequential: model
        """
        
        print(f"DQN Variant is {self.variant}")
        if self.variant == "duel" or self.variant=="duel_double":
            print("Checkpoint 1")
            input = Input((self.state_dim,))
            previous_layer = input
            for node in self.hidden_layers:
                layer = Dense(node, activation='relu')(previous_layer)
                previous_layer = layer
            value_output = Dense(1)(previous_layer)
            advantage_output = Dense(self.action_dim)(previous_layer)
            output = Add()([value_output, advantage_output])
            model = Model(input, output)
        else: 
            model = Sequential(
                    [Input((self.state_dim,))]\
                        +[Dense(i,activation = 'relu') for i in self.hidden_layers]\
                            +[Dense(self.action_dim)]               
                )
             
        model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))
        return model       
        
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self,state):
        """ Get the action given state.
            The action is taken as the one with highest q_value which predicted by the network.
            odd equals epsilon that the action taken randomly.
            Epsilon is decayed.

        Args:
            state (_type_): observed state
        Returns:
            int : action
        """
        
        state = np.reshape(state, [1,self.state_dim])
        q_value = self.predict(state)[0]
        
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim -1)
        else:
            return np.argmax(q_value)
    
    def train(self,states,targets):
        """ Train the network with 

        Args:
            states (_type_): _description_
            targets (_type_): _description_
        """
        if self.model:
            self.model.fit(x=states,y=targets,epochs=self.epochs,verbose=0)

class ReplayBuffer:
    """ Class to store the experience
        Picked randomly to train the network
    """
    def __init__(self,batch_size:int,capacity:int=10_000) -> None:
        """Initialize

        Args:
            batch_size (int): number of item sampled each time
            capacity (int, optional): buffer maximum length. Defaults to 10_000.
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def put(self,state,action,reward,next_state,done):
        self.buffer.append([state,action,reward,next_state,done])
    
    def sample(self):
        """ Pick sample experience from replay buffer
        Returns:
            _type_: _description_
        """
        sample = random.sample(self.buffer,self.batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)
        
class DQNAgent:
    """ The DQN agent for discrete action
    """
    def __init__(
        self,
        env:gym.Env,
        variant:str="Vanila",
        gamma:float=0.95,
        learning_rate:float=0.005,
        epsilon:float=1.0,
        eps_decay:float=0.99,
        eps_min:float=0.05,
        batch_size:int=4,
        replay_batch_num:int=2,
        epochs:int=1,
        hidden_layers:list[int]=[64,64]
        ) -> None:
        """Initialize

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): discount factor. Defaults to 0.95.
            variant (str): algorithm variant:'vanila','double','duel','duel-double'
            learning_rate (float, optional): Learning rate to update neural network. Defaults to 0.005.
            epsilon (float, optional): greedy parameter, if random below epsilon, the agent will take a randome action. Defaults to 1.0.
            eps_decay (float, optional): decay of epsilon. Defaults to 0.99.
            eps_min (float, optional): minimum epsilon. Defaults to 0.05.
            replay_batch_num (int): number of replays each episode. The replay will occured after ending of episode.
            batch_size (int, optional): the number of exprience pick out from buffer relay to train network each step. Defaults to 32.
            epochs (int, optional): number of epochs to train the neural network each update. Defaults to 1.
            hidden_layears (list[int], optional): hidden neural network layers and nodes. Defaults to [64,64].
        """
        self.env = env
        self.variant = variant
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n 
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.replay_batch_num = replay_batch_num
        self.epochs = epochs
        self.gamma = gamma
        self.batch_size = batch_size
        wandb.init()
        
        self.model = QNet(
            variant= self.variant,
            state_dim = self.state_dim,
            action_dim = self.action_dim,
            hidden_layers=self.hidden_layers,
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            eps_decay=self.eps_decay,
            eps_min = self.eps_min,
            epochs=self.epochs
        )
        
        self.target_model = QNet(
            variant=self.variant,
            state_dim = self.state_dim,
            action_dim = self.action_dim,
            hidden_layers=self.hidden_layers,
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            eps_decay=self.eps_decay,
            eps_min = self.eps_min,
            epochs=self.epochs
        )
        
        self.target_update()
        self.buffer = ReplayBuffer(batch_size=self.batch_size)
    
    def target_update(self):
        """Update weight of target network similar to action network
        """
        if self.model and self.target_model:
            weights = self.model.model.get_weights()
            self.target_model.model.set_weights(weights)
    
    def replay(self):
        """ Replay the experience.
            Take replay_batch_num times of sample.
            Use the target model to predict the next q-value.
            Use that q-value and temporal difference function to get
            q-value of current state. Use that as a true value to retrain
            the action network.
        """
            
        for _ in range(self.replay_batch_num):
            states, actions, rewards, next_states, dones = self.buffer.sample()
            targets = self.target_model.predict(states)
            
            if self.variant == "vanila" or self.variant == "duel":    
                next_q_values = self.target_model.predict(next_states).max(axis=1)
            else:
                next_q_values = self.target_model.predict(next_states)[
                        range(self.batch_size),np.argmax(self.model.predict(next_states),axis=1)]
            
            targets[range(self.batch_size),actions] = rewards + (1-dones) * next_q_values * self.gamma         
            self.model.train(states,targets)
    
    def train(self, max_episodes=1000):
        """Train the agent

        Args:
            max_episodes (int, optional): maximum number of episode. Defaults to 1000.
        """
        total_reward_list=[]
        for ep in range(max_episodes):
            wandb.log({'Episode':ep})
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state,action,reward,next_state,done)
                total_reward +=reward
                state = next_state
                self.epsilon = max(self.epsilon * self.eps_decay,self.eps_min)
                wandb.log({'Reward':reward})
                wandb.log({'Epsilon':self.epsilon})
                wandb.log({'LearningRate':self.learning_rate})
            if self.buffer.size() >= self.batch_size:
                self.replay()
                wandb.log({'Buffer Size':self.buffer.size()})
            self.target_update()
            wandb.log({'TotalReward':total_reward})
            total_reward_list.append(total_reward)
            wandb.log({'Mean TotalReward':np.array(total_reward_list).mean()})
            wandb.log({'Std TotalReward':np.array(total_reward_list).std()})
        